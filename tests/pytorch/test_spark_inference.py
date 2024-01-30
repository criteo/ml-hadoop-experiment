from typing import Any, Tuple, Union

import mock
import pandas as pd
import pytest
import torch
from pyspark.sql.types import ArrayType, DoubleType, StringType

from ml_hadoop_experiment.common.spark_inference import SerializableObj, artifact_type
from ml_hadoop_experiment.pytorch import spark_inference
from ml_hadoop_experiment.pytorch.fixtures.test_models import (
    load_reducer,
    load_tokenizer,
    load_translator,
)
from ml_hadoop_experiment.pytorch.spark_inference import (
    with_inference_column,
    with_inference_column_and_preprocessing,
)


def test_with_inference_column_and_serializable_model(local_spark_session):
    run_with_inference_column(local_spark_session, False, False)


def test_with_inference_column_and_not_serializable_model(local_spark_session):
    run_with_inference_column(local_spark_session, True, False)


@pytest.mark.parametrize("on_gpu", [True, False])
def test_with_inference_column_on_gpus(local_spark_session, on_gpu):
    run_with_inference_column(local_spark_session, False, on_gpu)


@pytest.mark.parametrize("serialize_objs", [(True,), (False,)])
def test_with_inference_column_with_multiple_artifacts(local_spark_session, serialize_objs):
    def run_prediction(
        artifacts: artifact_type, features: Tuple[pd.Series, ...], device: str
    ) -> pd.Series:
        model, tokenizer = artifacts
        tokens = tokenizer.encode(features[0])
        predictions = model(tokens)
        return pd.Series(tokenizer.decode(predictions))

    ss = local_spark_session
    inputs = ["Hello world", "How are you"]

    df = ss.createDataFrame(inputs, "string").toDF("data")
    model: Union[torch.nn.Module, SerializableObj] = (
        SerializableObj(ss, load_translator) if serialize_objs else load_translator()
    )
    tokenizer: Union[torch.nn.Module, SerializableObj] = (
        SerializableObj(ss, load_tokenizer) if serialize_objs else load_tokenizer()
    )
    df_result = with_inference_column(
        df=df,
        artifacts=[model, tokenizer],
        input_cols=["data"],
        inference_fn=run_prediction,
        output_type=StringType(),
        batch_size=1,
        output_col="predictions",
        num_threads=1,
    )
    pdf = df_result.toPandas()

    expected = pd.Series(["bonjour tout le monde", "comment vas tu"])
    _compare(expected, pdf["predictions"])


def test_with_retry():
    def func() -> None:
        raise RuntimeError("Error")

    with mock.patch("ml_hadoop_experiment.pytorch.spark_inference.log") as logger_mock:
        n_retry = 3
        try:
            spark_inference._with_retry(func, n_retry)
        except RuntimeError:
            pass
        assert logger_mock.call_count == n_retry


@pytest.mark.parametrize("on_gpu", [True, False])
def test_with_inference_column_and_preprocessing(local_spark_session, on_gpu):
    def preprocessing_fn(
        _: artifact_type, features: Tuple[Any, ...], _device: str
    ) -> Tuple[torch.Tensor, ...]:
        assert _device == "cpu" if not on_gpu else "cuda:0"
        feature1 = torch.Tensor(features[0].tolist()) + 10
        feature2 = torch.Tensor(features[1].tolist()) + 5
        return feature1, feature2

    def inference_fn(
        model: artifact_type, features: Tuple[torch.Tensor, ...], _device: str
    ) -> Tuple[Any, ...]:
        assert _device == "cpu" if not on_gpu else "cuda:0"
        results = model(*features)
        return results.numpy()

    ss = local_spark_session
    data = [([10.0, 11.0, 12.0], [-1.0]), ([20.0, 21.0, 22.0], [-2.0]), ([1.0, 2.0, 3.0], [-3.0])]
    df = ss.createDataFrame(data, ["feature1", "feature2"])
    model: Union[torch.nn.Module, SerializableObj] = load_reducer()
    with mock.patch(
        "ml_hadoop_experiment.pytorch.spark_inference.torch.cuda.is_available"
    ) as cuda_available_mock, mock.patch(
        "ml_hadoop_experiment.pytorch.spark_inference.get_cuda_device"
    ) as get_cuda_device_mock:
        cuda_available_mock.return_value = on_gpu
        get_cuda_device_mock.return_value = 0
        df_result = with_inference_column_and_preprocessing(
            df=df,
            artifacts=model,
            input_cols=["feature1", "feature2"],
            preprocessing=preprocessing_fn,
            inference_fn=inference_fn,
            output_type=DoubleType(),
            batch_size=2,
            output_col="predictions",
            num_threads=1,
        )
        pdf = df_result.toPandas()
        expected = pd.Series([51.0, 84.0, 30.0])
        _compare(expected, pdf["predictions"])


def test_with_inference_column_and_preprocessing_computed_once(local_spark_session):

    def preprocessing_fn(
        counter: artifact_type, features: Tuple[Any, ...], device: str
    ) -> Tuple[torch.Tensor, ...]:
        return features

    def inference_fn(
        counter: artifact_type, features: Tuple[torch.Tensor, ...], device: str
    ) -> Tuple[Any, ...]:
        counter["value"] += 1
        return tuple(
            zip(features[0].numpy() + counter["value"], features[1].numpy() + counter["value"])
        )

    ss = local_spark_session
    df = ss.createDataFrame([(2.0, 3.0), (12.0, 13.0)], ["feature1", "feature2"])
    df_result = with_inference_column_and_preprocessing(
        df=df,
        artifacts={"value": 0},
        input_cols=["feature1", "feature2"],
        preprocessing=preprocessing_fn,
        inference_fn=inference_fn,
        output_type=ArrayType(DoubleType()),
        batch_size=2,
        output_col="predictions",
        num_threads=1,
    )
    df_result = (
        df_result.withColumn("predictions1", df_result["predictions"].getItem(0))
        .withColumn("predictions2", df_result["predictions"].getItem(1))
        .drop("predictions")
    )
    pdf = df_result.toPandas()
    _compare(pd.Series([3.0, 13.0]), pdf["predictions1"])
    _compare(pd.Series([4.0, 14.0]), pdf["predictions2"])


def test_with_inference_column_computed_once(local_spark_session):

    def inference_fn(
        counter: artifact_type, features: Tuple[pd.Series, ...], device: str
    ) -> pd.Series:
        counter["value"] += 1
        return pd.Series(zip(features[0] + counter["value"], features[1] + counter["value"]))

    ss = local_spark_session
    df = ss.createDataFrame([(2.0, 3.0), (12.0, 13.0)], ["feature1", "feature2"])
    df_result = with_inference_column(
        df=df,
        artifacts={"value": 0},
        input_cols=["feature1", "feature2"],
        inference_fn=inference_fn,
        output_type=ArrayType(DoubleType()),
        batch_size=2,
        output_col="predictions",
        num_threads=1,
    )
    df_result = (
        df_result.withColumn("predictions1", df_result["predictions"].getItem(0))
        .withColumn("predictions2", df_result["predictions"].getItem(1))
        .drop("predictions")
    )
    pdf = df_result.toPandas()
    _compare(pd.Series([3.0, 13.0]), pdf["predictions1"])
    _compare(pd.Series([4.0, 14.0]), pdf["predictions2"])


def run_with_inference_column(local_spark_session, serialize_model: bool, on_gpu: bool):
    def run_prediction(
        model: artifact_type, features: Tuple[pd.Series, ...], device: str
    ) -> pd.Series:
        assert device == "cpu" if not on_gpu else "cuda:0"
        predictions = model(torch.Tensor(features[0].tolist()), torch.Tensor(features[1].tolist()))
        return pd.Series(predictions.tolist())

    ss = local_spark_session
    data = [([10.0, 11.0, 12.0], [-1.0]), ([20.0, 21.0, 22.0], [-2.0]), ([1.0, 2.0, 3.0], [-3.0])]

    df = ss.createDataFrame(data, ["feature1", "feature2"])
    model: Union[torch.nn.Module, SerializableObj] = (
        SerializableObj(ss, load_reducer) if serialize_model else load_reducer()
    )
    with mock.patch(
        "ml_hadoop_experiment.pytorch.spark_inference.torch.cuda.is_available"
    ) as cuda_available_mock, mock.patch(
        "ml_hadoop_experiment.pytorch.spark_inference.get_cuda_device"
    ) as get_cuda_device_mock:
        cuda_available_mock.return_value = on_gpu
        get_cuda_device_mock.return_value = 0
        df_result = with_inference_column(
            df=df,
            artifacts=model,
            input_cols=["feature1", "feature2"],
            inference_fn=run_prediction,
            output_type=DoubleType(),
            batch_size=1,
            output_col="predictions",
            num_threads=1,
        )
        pdf = df_result.toPandas()

        expected = pd.Series([36.0, 69.0, 15.0])
        _compare(expected, pdf["predictions"])  # type: ignore


def _compare(expected: pd.Series, actual: pd.Series) -> None:
    assert expected.size == actual.size
    for i in range(expected.size):
        assert expected[i] == actual[i]
