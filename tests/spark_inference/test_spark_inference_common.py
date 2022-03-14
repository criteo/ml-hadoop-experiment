import os
import tempfile
import fcntl
from contextlib import contextmanager
import json
from typing import Dict, List

import mock
import pytest

from ml_hadoop_experiment.common.spark_inference import get_cuda_device, CUDA_DEVICE_ENV


def test_get_cuda_device_without_allocation():
    with tempfile.NamedTemporaryFile() as lock_tmp, \
            tempfile.TemporaryDirectory() as tmp_dir, \
            _file_locking_mock():
        allocation_tmp = os.path.join(tmp_dir, "allocation")
        device = get_cuda_device(
            n_gpus=3, lock_file=lock_tmp.name, allocation_file=allocation_tmp
        )
        assert device == 0


@pytest.mark.parametrize(
    "alloc_map,pid,expected_cuda_device",
    [
        ({0: [2]}, 1, 1),
        ({1: [2]}, 2, 1),
        ({0: [2], 2: [1]}, 3, 1),
        ({0: [2], 1: [3], 2: [1]}, 4, 0),
        ({0: [1, 2], 1: [3], 2: [4, 5]}, 6, 1)
    ]
)
def test_get_cuda_device_with_existing_allocations(alloc_map, pid, expected_cuda_device):
    all_pids = []
    for pids in alloc_map.values():
        all_pids.extend(pids)
    cuda_device = _run_test_get_cuda_device(
        existing_allocs=alloc_map, pid=pid, existing_pids=all_pids, n_gpus=3
    )
    assert cuda_device == expected_cuda_device


def test_get_cuda_device_reuse_allocation_of_previous_pid():
    cuda_device = _run_test_get_cuda_device(
        existing_allocs={0: [1], 1: [2], 2: [3]}, pid=4, existing_pids=[1, 3], n_gpus=3
    )
    assert cuda_device == 1


def test_get_cuda_device_caches_cuda_device():
    cleanup()
    with mock.patch("ml_hadoop_experiment.common.spark_inference._get_cuda_device") \
        as _get_cuda_device_mock, \
            mock.patch("ml_hadoop_experiment.common.spark_inference.os.getpid") as getpid_mock:
        _get_cuda_device_mock.return_value = 0
        getpid_mock.return_value = 0
        cuda_device = get_cuda_device(n_gpus=1)
        assert cuda_device == get_cuda_device(n_gpus=1)
        _get_cuda_device_mock.assert_called_once()


def _run_test_get_cuda_device(
    existing_allocs: Dict[int, int], pid: int, existing_pids: List[int],
    n_gpus: int
) -> int:
    cleanup()
    with tempfile.NamedTemporaryFile() as lock_tmp, \
            tempfile.NamedTemporaryFile() as allocation_tmp, \
            mock.patch("ml_hadoop_experiment.common.spark_inference.os.getpid") as getpid_mock, \
            mock.patch("ml_hadoop_experiment.common.spark_inference._get_all_pids") as all_pids_mock, \
            _file_locking_mock():
        with open(allocation_tmp.name, "w+") as fd:
            fd.write(json.dumps(existing_allocs))
        getpid_mock.return_value = pid
        all_pids_mock.return_value = existing_pids
        return get_cuda_device(
            n_gpus=n_gpus, lock_file=lock_tmp.name, allocation_file=allocation_tmp.name
        )


@contextmanager
def _file_locking_mock():
    with mock.patch("ml_hadoop_experiment.common.spark_inference.fcntl") as fcntl_mock:
        yield
        fcntl_mock.lockf.assert_called_once()
        fcntl_mock.lockf.call_args_list == [mock.call(mock.ANY, fcntl.LOCK_EX)]
        fcntl_mock.flock.assert_called_once()
        fcntl_mock.flock.call_args_list == [mock.call(mock.ANY, fcntl.LOCK_UN)]


def cleanup():
    if CUDA_DEVICE_ENV in os.environ:
        del os.environ[CUDA_DEVICE_ENV]
