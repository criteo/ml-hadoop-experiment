import os
from typing import List, Callable, Tuple, Any
import logging
import uuid

import torch
from torch.utils.data import DataLoader
import pandas as pd
import pyspark
from pyspark.sql import functions as sf
from pyspark.sql.types import DataType

from ml_hadoop_experiment.common.spark_inference import (
    broadcast, artifact_type, from_broadcasted, split_in_batches, get_cuda_device
)

_logger = logging.getLogger(__file__)

# User-defined function to compute inference on Pandas Series
# Inputs: artifacts, list of feature columns, device
# Output: inference columnn
pandas_inference_udf = Callable[[artifact_type, Tuple[pd.Series, ...], str], pd.Series]

# User-defined preprocessing function
# Inputs: artifacts, list of feature vectors (row-by-row), device
# Output: list of transformed feature vectors that will be
# provided to inference function
preprocessing_fn = Callable[[artifact_type, Tuple[Any, ...], str], Tuple[torch.Tensor, ...]]

# User-defined function to compute inference on tensors
# Inputs: artifacts, batches of transformed feature vectors, device
# Output: batches of results. Anything but tensors (use numpy() to transform tensors to numpy)
tensor_inference_udf = Callable[[artifact_type, Tuple[torch.Tensor, ...], str], Tuple[Any, ...]]


class PandasSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, features: Tuple[pd.Series, ...]):
        self.features = features
        self.n_features = len(features)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        return tuple(self.features[i].iloc[index] for i in range(self.n_features))

    def __len__(self) -> int:
        return len(self.features[0])


def with_inference_column_and_preprocessing(
    df: pyspark.sql.DataFrame,
    artifacts: artifact_type,
    input_cols: List[str],
    preprocessing: preprocessing_fn,
    inference_fn: tensor_inference_udf,
    output_type: DataType,
    batch_size: int = 1,
    output_col: str = "prediction",
    num_threads: int = 8,
    num_workers_preprocessing: int = 8,
    dataloader_timeout_secs: int = 60,
    dataloader_max_retry: int = 3
) -> pyspark.sql.dataframe.DataFrame:
    """
    :param df: dataframe that holds the input
    :param artifacts: Artifact (model, tokenizer ...) or list of artifact to use for inference.
    Example: model, model + tokenizer ...
    :param input_col: names of the dataframe columns that will be used as inputs for inference
    :param preprocessing: preprocessing function.
    It takes as inputs a list of artifact (model, model + tokenizer ...),
    a list of feature vectors (row-by-row) and a device (cpu, cuda:0, cuda:1 ...).
    It returns a batch of tensors that will be provided to inference function
    :param inference_fn: inference function.
    It takes as inputs a list of artifact (model, model + tokenizer ...),
    a batch of tensors (outputs of preprocessing function)
    and a device (cpu, cuda:0, cuda:1 ...).
    It returns batches of results. Anything but tensors (use numpy() to transform tensors to numpy)
    :param output_type: output type of the predictions
    :param batch_size: batch size
    :param output_col: name of the colonne of the dataframe that will hold the predictions
    :param num_threads: Number of threads to run PyTorch inter/inta-ops
    The two parameters below are a workaround for the time being as long as we don't find
    the issue causing some tasks to get stuck in the dataloader
    :param dataloader_timeout_secs: timeout applied to the dataloader used to load and preprocess
    data
    :param dataloader_max_retry: maxinum number of retry applied to the dataloader used to load
    and preprocess data
    input data
    """
    _inference_fn = _tensor_inference_udf_wrapper(
        preprocessing, inference_fn, batch_size, num_workers_preprocessing,
        dataloader_timeout_secs, dataloader_max_retry
    )
    return _with_inference_column(
        df, artifacts, input_cols, _inference_fn, output_type, output_col,
        num_threads
    )


def with_inference_column(
    df: pyspark.sql.DataFrame,
    artifacts: artifact_type,
    input_cols: List[str],
    inference_fn: pandas_inference_udf,
    output_type: DataType,
    batch_size: int = 1,
    output_col: str = "prediction",
    num_threads: int = 8,
) -> pyspark.sql.dataframe.DataFrame:
    """
    :param df: dataframe that holds the input
    :param artifacts: Artifact (model, tokenizer ...) or list of artifact to use for inference.
    Example: model, model + tokenizer ...
    :param input_col: names of the dataframe columns that will be used as inputs for inference
    :param inference_fn: function that is run to compute predictons.
    It takes as inputs a list of artifact (model, model + tokenizer ...), a list of pandas series
    and a device (cpu, cuda:0, cuda:1 ...).
    Each pandas serie represent one of the dataframe column of :param input_cols, in the same order.
    It returns a Pandas Serie
    :param output_type: output type of the predictions
    :param batch_size: batch size
    :param output_col: name of the colonne of the dataframe that will hold the predictions
    :param num_threads: Number of threads to run PyTorch inter/inta-ops
    """
    _inference_fn = _pandas_inference_udf_wrapper(inference_fn, batch_size)
    return _with_inference_column(
        df, artifacts, input_cols, _inference_fn, output_type, output_col,
        num_threads
    )


def _tensor_inference_udf_wrapper(
    preprocessing: preprocessing_fn,
    inference_fn: tensor_inference_udf,
    batch_size: int,
    num_workers_preprocessing: int,
    dataloader_timeout_secs: int,
    dataloader_max_retry: int
) -> pandas_inference_udf:

    def _wrapper(
        artifacts: artifact_type,
        features: Tuple[pd.Series, ...],
        device: str,
    ) -> pd.Series:

        def _preprocess(_features: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
            return preprocessing(artifacts, _features, device)

        def _run_udf() -> pd.Series:
            dataset = PandasSeriesDataset(features)
            dataset = dataset.map(_preprocess)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers_preprocessing,
                prefetch_factor=2,
                timeout=dataloader_timeout_secs
            )
            all_results: List[Any] = list()
            for batch in dataloader:
                results = inference_fn(artifacts, batch, device)
                all_results.extend(results)
            return pd.Series(all_results)

        return _with_retry(_run_udf, dataloader_max_retry)

    return _wrapper


def _with_retry(func: Callable[[], Any], max_retry: int) -> None:
    n_try = 1
    while n_try <= max_retry:
        n_try += 1
        try:
            return func()
        except RuntimeError as e:
            _log(f"Caught exception {e}")
            if n_try > 3:
                raise e


def _pandas_inference_udf_wrapper(
    inference_fn: pandas_inference_udf,
    batch_size: int
) -> pandas_inference_udf:

    def _wrapper(
        artifacts: artifact_type,
        features: Tuple[pd.Series, ...],
        device: str,
    ) -> pd.Series:
        results = pd.Series()
        for batch in split_in_batches(features, batch_size):
            result = inference_fn(artifacts, batch, device)
            results = results.append(result)
        return results
    return _wrapper


def _with_inference_column(
    df: pyspark.sql.DataFrame,
    artifacts: artifact_type,
    input_cols: List[str],
    inference_fn: pandas_inference_udf,
    output_type: DataType,
    output_col: str = "prediction",
    num_threads: int = 8
) -> pyspark.sql.dataframe.DataFrame:

    def _inference_fn(*features: pd.Series) -> pd.Series:
        # set_num_interop_threads and set_num_threads must be called only once
        if num_threads != torch.get_num_interop_threads():
            torch.set_num_interop_threads(num_threads)
        if num_threads != torch.get_num_threads():
            torch.set_num_threads(num_threads)

        artifacts = from_broadcasted(broadcasted_artifacts)

        with torch.no_grad():
            device = "cpu"
            if torch.cuda.is_available():
                cuda_device = get_cuda_device(
                    torch.cuda.device_count(), lock_file, allocation_file
                )
                device = f"cuda:{cuda_device}"
            _log(f"Running inference on {device}")
            return inference_fn(artifacts, features, device)

    broadcasted_artifacts = broadcast(df._sc, artifacts)
    file_id = str(uuid.uuid4())
    lock_file = f"/tmp/lockfile_{file_id}"
    allocation_file = f"/tmp/allocation_cuda_{file_id}"

    _inference_udf = sf.pandas_udf(_inference_fn, returnType=output_type)
    # In some situation, the pandas udf can be computed more than once when the
    # output column is referenced more than once,
    # (https://issues.apache.org/jira/browse/SPARK-17728).
    # One workaround is to wrap the pandas udf in sf.explode(sf.array())
    return df.withColumn(output_col, sf.explode(sf.array(_inference_udf(*input_cols))))


def _log(msg: str, level: int = logging.INFO) -> None:
    _logger.log(level, f"[{os.getpid()}] {msg}")
