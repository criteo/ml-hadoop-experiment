import os
import json
import fcntl
import logging
import math
import pickle
from typing import Callable, Any, Dict, Tuple, List, Union, Iterator, Set

import pandas as pd
import pyspark
import psutil


_logger = logging.getLogger(__file__)

# Artifact(s) to use for inference. Could be a single model, a model + tokenizer ...
artifact_type = Any

# Function to load all artifacts to use for inference
load_fn_type = Callable[[Any], artifact_type]

CUDA_DEVICE_ENV = "CUDA_DEVICE"


class _SerializableObjWrapper(object):
    """
    We override __getstate__() and __setstate__() to allow easy serialization
    of this instance on the workers when this is used in a UDF.
    """
    def __init__(
        self, load_fn: load_fn_type, *load_fn_args: Any
    ):
        self.obj = load_fn(*load_fn_args)
        self.load_fn_args = load_fn_args
        self.load_fn = load_fn

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "load_fn": self.load_fn,
            "load_fn_args": self.load_fn_args
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.load_fn_args = state["load_fn_args"]
        self.obj = state["load_fn"](*(self.load_fn_args))


class SerializableObj(object):
    def __init__(
        self, sparkSession: pyspark.sql.SparkSession,
        load_fn: load_fn_type, *load_fn_args: Any
    ):
        self.ew = _SerializableObjWrapper(load_fn, *load_fn_args)
        self.broadcast = sparkSession.sparkContext.broadcast(self.ew)

    def __enter__(self) -> 'SerializableObj':
        return self

    def __exit__(self, *exc_details: Any) -> None:
        self.broadcast.destroy()


class Locker:
    def __init__(self, lock_file: str) -> None:
        self.lock_file = lock_file

    def __enter__(self) -> None:
        self.fp = open(self.lock_file, "w")
        fcntl.lockf(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type: Any, value: Any, tb: Any) -> None:
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def _allocate_cuda_device(
    n_gpus: int, allocation_map: Dict[int, Set[int]], all_pids: Set[int], pid: int
) -> Tuple[int, Dict[int, Set[int]]]:
    cuda_device = None
    new_allocation_map = {}
    for cuda in range(n_gpus):
        if cuda in allocation_map:
            pids = allocation_map[cuda]
            new_allocation_map[cuda] = pids.intersection(all_pids)
            if pid in pids:
                cuda_device = cuda
        else:
            new_allocation_map[cuda] = set()
    if cuda_device:
        return cuda_device, new_allocation_map

    new_sorted_allocation_map = dict(sorted(
        new_allocation_map.items(), key=lambda item: len(item[1])
    ))
    cuda, pids = list(new_sorted_allocation_map.items())[0]
    pids.add(pid)
    return cuda, new_sorted_allocation_map


def _get_cuda_device(n_gpus: int, pid: int, allocation_file: str) -> int:
    if not os.path.exists(allocation_file):
        cuda_device = 0
        new_allocation_map = {cuda_device: [pid]}
        for cuda in range(1, n_gpus):
            new_allocation_map[cuda] = []
        with open(allocation_file, 'w') as fp:
            fp.write(json.dumps(new_allocation_map))
    else:
        with open(allocation_file, 'r+') as fp:
            allocation_map = json.loads(fp.read())
            allocation_map = {int(cuda): set(pids) for cuda, pids in allocation_map.items()}
            cuda_device, _new_allocation_map = \
                _allocate_cuda_device(n_gpus, allocation_map, _get_all_pids(), pid)
            fp.seek(0)
            fp.write(json.dumps({cuda: list(pids) for cuda, pids in _new_allocation_map.items()}))
            fp.truncate()
    return cuda_device


def _get_all_pids() -> Set[int]:
    return {p.pid for p in psutil.process_iter()}


# Method used to dispatch Spark tasks on GPUs of a GPU machine
# A GPU can be shared between tasks. This method uniformly allocates GPUs to tasks
def get_cuda_device(
    n_gpus: int,
    lock_file: str = "/tmp/lockfile",
    allocation_file: str = "/tmp/allocation_cuda"
) -> int:
    # Python workers are reused by default. In that case we can allocate the GPU only
    # once by storing the allocated GPU in env var
    if CUDA_DEVICE_ENV not in os.environ:
        with Locker(lock_file):
            cuda_device = _get_cuda_device(n_gpus, os.getpid(), allocation_file)
            os.environ[CUDA_DEVICE_ENV] = str(cuda_device)
    else:
        cuda_device = int(os.environ[CUDA_DEVICE_ENV])
    return cuda_device


def split_in_batches(
    series: Tuple[pd.Series, ...],
    batch_size: int,
) -> Iterator[Tuple[pd.Series, ...]]:
    n_rows = series[0].size
    n_batches = math.ceil(n_rows / batch_size)
    print(f"Number of rows: {n_rows}")
    print(f"Number of batches: {n_batches}")
    for i in range(n_batches):
        print(f"Processing batch {i} ...")
        start = i * batch_size
        stop = (i + 1) * batch_size
        # No need to handle out of range indices for stop. Pandas handles it for us
        yield tuple(serie[start:stop] for serie in series)


def _is_serializable(obj: Any) -> bool:
    if isinstance(obj, SerializableObj):
        return True
    try:
        pickle.loads(pickle.dumps(obj))
        return True
    except Exception as e:
        _logger.info(f"Unable to serialize/deserialize object {obj}; Caught excepttion: {e}")
        return False


def broadcast(
    sc: pyspark.context.SparkContext, artifact: artifact_type
) -> Union[List[pyspark.broadcast.Broadcast], pyspark.broadcast.Broadcast]:
    if not sc:
        raise ValueError("You must provide a spark context to serialize inference_fn_args")
    if isinstance(artifact, List):
        return [_broadcast(sc, obj) for obj in artifact]
    else:
        return _broadcast(sc, artifact)


def from_broadcasted(
    broadcasted_obj: Union[List[pyspark.broadcast.Broadcast], pyspark.broadcast.Broadcast]
) -> Any:
    if isinstance(broadcasted_obj, List):
        return [_from_broadcasted(obj) for obj in broadcasted_obj]
    else:
        return _from_broadcasted(broadcasted_obj)


def _broadcast(sc: pyspark.context.SparkContext, artifact: Any) -> pyspark.broadcast.Broadcast:
    if isinstance(artifact, SerializableObj):
        return artifact.broadcast
    else:
        if _is_serializable(artifact):
            return sc.broadcast(artifact)
        else:
            raise ValueError(f"Object {artifact} is not serializable")


def _from_broadcasted(broadcasted_obj: pyspark.broadcast.Broadcast) -> Any:
    obj = broadcasted_obj.value
    if isinstance(obj, _SerializableObjWrapper):
        obj = obj.obj
    return obj
