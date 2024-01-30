import logging
from typing import Any, Callable, Dict, List, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import pyspark
import tensorflow as tf
from pyspark.sql import functions as sf
from pyspark.sql.column import Column
from pyspark.sql.types import DataType, FloatType

from ml_hadoop_experiment.common.spark_inference import (
    SerializableObj,
    artifact_type,
    broadcast,
    from_broadcasted,
    get_cuda_device,
    log,
    split_in_batches,
)
from ml_hadoop_experiment.tensorflow import tfrecords
from ml_hadoop_experiment.tensorflow.predictor import Predictor, feeds_type, fetches_type

_logger = logging.getLogger(__file__)

features_specs_type = Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]]
postprocessing_fn_type = Callable[[Dict[str, np.ndarray]], pd.Series]
estimator_type = Callable[[List[Dict]], List[Union[float, List[float]]]]
_default_signature = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# User-defined function to compute inference with TensorFlow models
# Inputs: artifacts, list of feature columns
# Output: inference columnn
inference_udf = Callable[[artifact_type, Tuple[pd.Series, ...]], pd.Series]


def _canned_linear_classifier_extract_prediction_fn(fetch_tensors: Dict[str, List]) -> List[float]:
    """
    fetch_tensors: output of canned LinearClassifier estimator:
        {
            'scores': [
                (proba_class1_example1, proba_class2_example1, proba_class3_example1, ...),
                (proba_class1_example2, proba_class2_example2, proba_class3_example2, ...),
                (proba_class1_example3, proba_class2_example3, proba_class3_example3, ...),
                ...
            ]
        }

    Return: probabilities of class 1 (which is the positive class for binary classfiers)
    """
    return [float(s[1]) for s in fetch_tensors["scores"]]


def _default_extract_fn(fetch_tensors: Dict[str, np.ndarray]) -> pd.Series:
    """
    Extract probabilities of the first class, which corresponds to the positive class
    for binary classifiers.

    fetch_tensors:
        {
            'scores': [
                (proba_class1_example1, proba_class2_example1, proba_class3_example1, ...),
                (proba_class1_example2, proba_class2_example2, proba_class3_example2, ...),
                (proba_class1_example3, proba_class2_example3, proba_class3_example3, ...),
                ...
            ]
        }

    Return: probabilities of class 1
    """
    return pd.Series(fetch_tensors["scores"][:, 1])


def estimator_model(sparkSession: pyspark.sql.SparkSession, export_model_path: str) -> SerializableObj:
    """
    Wrap a model built with Estimator API for inference with Spark
    Wrapped model is guaranteed to be broadcastable
    """
    return SerializableObj(sparkSession, tf.compat.v1.saved_model.load_v2, export_model_path)


def keras_model(sparkSession: pyspark.sql.SparkSession, export_model_path: str) -> SerializableObj:
    """
    Wrap a Keras model for inference with Spark
    Wrapped model is guaranteed to be broadcastable
    """
    return SerializableObj(sparkSession, tf.keras.models.load_model, export_model_path)


def graph_model(
    sparkSession: pyspark.sql.SparkSession,
    export_model_path: str,
    feeds: feeds_type,
    fetches: fetches_type,
) -> SerializableObj:
    """
    Wrap a graph model for inference with Spark
    Wrapped model is guaranteed to be broadcastable
    """
    return SerializableObj(sparkSession, Predictor.from_graph, export_model_path, feeds, fetches)  # type: ignore


def with_graph_inference_column(
    df: pyspark.sql.dataframe.DataFrame,
    model: SerializableObj,
    output_column_name: str = "prediction",
    output_column_type: pyspark.sql.types.DataType = FloatType(),
    postprocessing_fn: postprocessing_fn_type = lambda x: pd.Series(x["score"][:, 0]),
) -> pyspark.sql.dataframe.DataFrame:
    """
    Runs inference on the input dataframe and adds a column 'output_column_name' with
    your postprocessed model outputs.
    Method to use if your model is a graph.

    :param df: the dataframe to add a prediction to
    :param model: instance of SerializableObj wrapping your model
    :param output_column_name: name of the newly-created inference column
    :param output_column_type: type of the newly-created inference column
    postprocessing_fn: postprocessing function called on your model outputs
    The primary purpose of this functon is to extract the relevant scores/predictions of
    your model outputs but it is not limited to this use case.
    :return: a new dataframe with a new column 'output_column_name'
    """

    feature_names = list(model.ew.obj.feed_tensors.keys())
    for feature_name in feature_names:
        if feature_name not in df.columns:
            raise ValueError(f"{feature_name} not found in columns {df.columns}")

    def _inference_fn(model: artifact_type, series: Tuple[pd.Series, ...]) -> pd.Series:
        batch_size = series[0].size

        def input_fn() -> tf.data.Dataset:
            _series = []
            for serie in series:
                if len(serie.values[0].shape) == 0:
                    _series.append(serie.values.reshape(batch_size, 1))  # type: ignore
                elif isinstance(serie.values[0], np.ndarray):
                    _series.append([e.tolist() for e in serie.values])
                else:
                    _series.append(serie)
            return tf.data.Dataset.from_tensor_slices(
                {feature_name: serie for (feature_name, serie) in zip(feature_names, _series)}  # type: ignore
            ).batch(batch_size)

        outputs = next(model.predict(input_fn))
        return postprocessing_fn(outputs)

    return with_inference(df, model, _inference_fn, feature_names, output_column_type, output_column_name)


def with_inference_column(
    df: pyspark.sql.dataframe.DataFrame,
    tfrecords_col: Union[pyspark.sql.Column, str],
    model: artifact_type,
    output_column_name: str = "prediction",
    feed_tensor_key: str = "inputs",
    postprocessing_fn: postprocessing_fn_type = _default_extract_fn,
) -> pyspark.sql.dataframe.DataFrame:
    """
    Runs inference on the input dataframe and adds a column 'output_column_name' with
    your postprocessed model outputs.
    Method to use if your model has been built with the Estimator API

    :param df: the dataframe to add a prediction to
    :paramm tfrecords_col: column of your dataframe containing the tfrecords to use as inputs
    of your model
    :param model: model to use for inference
    :param output_column_name: name of the newly-created inference column
    :param feed_tensor_key: feed tensor key to use to feed your model with inputs
    postprocessing_fn: postprocessing function called on your model outputs
    The primary purpose of this functon is to extract the relevant scores/predictions of
    your model outputs but it is not limited to this use case.
    :return: a new dataframe with a new column 'output_column_name'
    """

    def _inference_fn(model: artifact_type, series: Tuple[pd.Series, ...]) -> pd.Series:
        import tensorflow as tf

        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        outputs = model.signatures[_default_signature](**{feed_tensor_key: tf.constant(series[0])})
        return postprocessing_fn(outputs)

    return with_inference(df, model, _inference_fn, [tfrecords_col], FloatType(), output_column_name)


def with_inference(
    df: pyspark.sql.DataFrame,
    model: artifact_type,
    inference_fn: inference_udf,
    input_column_names: List[Union[str, pyspark.sql.Column]],
    output_column_type: DataType,
    output_column_name: str = "prediction",
    batch_size: int = 1,
    num_threads: int = 8,
) -> pyspark.sql.dataframe.DataFrame:
    """
    Runs 'inference_fn' on the input dataframe and adds a column 'output_column_name' with
    outputs of 'inference_fn'

    :param df: dataframe that holds the input
    :param model: model to use for inference
    :param inference_fn: function run for inference.
    It takes as inputs a model and a list of pandas series. Each pandas serie represent
    one of the dataframe column of :param input_column_names, in the same order.
    It returns a Pandas Serie containing outputs of 'inference_fn'
    :param input_column_names: columns used for inference
    :param output_column_type: type of the newly-created inference column
    :param output_column_name: name of the newly-created inference column
    :param batch_size: batch size used to during inference
    :param num_threads: Number of threads for inter/inta-ops used during inference
    """
    broadcasted_artifacts = broadcast(df._sc, model)

    def _inference_fn(*rows: pd.Series) -> pd.Series:
        import tensorflow as tf

        # set_num_interop_threads and set_num_threads must be called only once
        _n_threads = tf.config.threading.get_inter_op_parallelism_threads()
        if _n_threads != num_threads:
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        _n_threads = tf.config.threading.get_intra_op_parallelism_threads()
        if _n_threads != num_threads:
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)

        artifacts = from_broadcasted(broadcasted_artifacts)

        def _run_inference():
            results = pd.Series()
            for batch in split_in_batches(rows, batch_size):
                results = results.append(inference_fn(artifacts, batch))
            return results

        tf.debugging.set_log_device_placement(True)
        gpu_devices = [d for d in tf.config.experimental.list_physical_devices() if "XLA_GPU" in d]
        n_gpu_devices = len(gpu_devices)
        if n_gpu_devices > 0:
            file_id = str(uuid4())
            lock_file = f"/tmp/lockfile_{file_id}"
            allocation_file = f"/tmp/allocation_cuda_{file_id}"
            cuda_device = get_cuda_device(n_gpu_devices, lock_file, allocation_file)
            log(_logger, f"Running inference on GPU {cuda_device}")
            with tf.device(f"/device:GPU:{cuda_device}"):
                return _run_inference()
        else:
            log(_logger, "Running inference on CPU")
            return _run_inference()

    inference_udf = sf.pandas_udf(_inference_fn, returnType=output_column_type)  # type: ignore
    # In some situation, the pandas udf can be computed more than once when the
    # output column is referenced more than once,
    # (https://issues.apache.org/jira/browse/SPARK-17728).
    # One workaround is to wrap the pandas udf in sf.explode(sf.array())
    return df.withColumn(output_column_name, sf.explode(sf.array(inference_udf(*input_column_names))))


def predict_with_tfr(
    features_specs: features_specs_type,
    model_path: str,
    postprocessing_fn: Callable = _canned_linear_classifier_extract_prediction_fn,
    feed_tensor_key: str = "inputs",
) -> estimator_type:
    """
    features_specs: specifications of your model input features
    model_path: path to your model
    postprocessing_fn: postprocessing function called on your model outputs
    The primary purpose of this functon is to extract the relevant scores/predictions of
    your model outputs but it is not limited to this use case.
    feed_tensor_key: feed tensor key to feed your model with inputs
    """
    estimator = tf.compat.v1.saved_model.load_v2(model_path)

    def _predict(inputs: List[Dict]) -> List[Any]:
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        serialized_tfrecords = [tfrecords.to_tf_proto(e, features_specs).SerializeToString() for e in inputs]
        results = estimator.signatures[_default_signature](**{feed_tensor_key: tf.constant(serialized_tfrecords)})
        return postprocessing_fn(results)

    return _predict


def filtered_columns(df: pyspark.sql.dataframe.DataFrame, specs: features_specs_type) -> List[Column]:
    return [df[x] for x in df.columns if x in specs.keys()]
