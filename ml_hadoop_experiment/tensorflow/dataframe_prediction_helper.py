from typing import (
    Dict,
    List,
    Union,
    Callable,
    Tuple
)

import tensorflow as tf
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType, DataType
from pyspark.sql.column import Column

from ml_hadoop_experiment.tensorflow import tfrecords
from ml_hadoop_experiment.tensorflow.predictor import Predictor, feeds_type, fetches_type
from ml_hadoop_experiment.common.spark_inference import SerializableObj, \
    broadcast, from_broadcasted, artifact_type, split_in_batches

features_specs_type = Dict[
        str,
        Union[
            tf.io.FixedLenFeature,
            tf.io.VarLenFeature
        ]
    ]
extract_fn_type = Callable[[Dict[str, np.ndarray]], pd.Series]
estimator_type = Callable[[List[Dict]], List[Union[float, List[float]]]]

_default_signature = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# User-defined function to compute inference with PyTorch models
# Inputs: artifacts, list of feature columns
# Output: inference columnn
inference_udf = Callable[[artifact_type, Tuple[pd.Series, ...]], pd.Series]


def _canned_linear_classifier_extract_prediction_fn(
    fetch_tensors: Dict[str, List]
) -> List[float]:
    '''
    fetch_tensors: inference result with canned LinearClassifier estimator.
        The result is a dictionary mapping keyword 'scores' to a list of tuples,
        as many tuples as examples given to the estimator for inference.
        Each tuple consists of two probabilities, one per class

    Return: List of predictions of positive class, as many as examples given to
        the estimator for inference.
    '''
    return [float(s[1]) for s in fetch_tensors['scores']]


def predict_with_tfr(
    features_specs: features_specs_type,
    model_path: str,
    extract_prediction_fn: Callable = _canned_linear_classifier_extract_prediction_fn,
    feed_tensor_key: str = "inputs"
) -> estimator_type:
    estimator = tf.compat.v1.saved_model.load_v2(model_path)

    def _predict(
        inputs: List[Dict]
    ) -> List[Union[float, List[float]]]:
        import tensorflow as tf
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        serialized_tfrecords = [
            tfrecords.to_tf_proto(
                e,
                features_specs
            ).SerializeToString()
            for e in inputs]
        results = estimator.signatures[_default_signature](
            **{feed_tensor_key: tf.constant(serialized_tfrecords)})
        return extract_prediction_fn(results)
    return _predict


def default_extract_fn(outputs: Dict[str, np.ndarray]) -> pd.Series:
    return pd.Series(outputs["scores"][:, 1])


def filtered_columns(
    df: pyspark.sql.dataframe.DataFrame,
    specs: features_specs_type
) -> List[Column]:
    return [df[x] for x in df.columns if x in specs.keys()]


def estimator_model(
    sparkSession: pyspark.sql.SparkSession, export_model_path: str
) -> SerializableObj:
    """
    Wrap an model built with Estimator API for inference with Spark
    """
    return SerializableObj(sparkSession, tf.compat.v1.saved_model.load_v2, export_model_path)


def keras_model(
    sparkSession: pyspark.sql.SparkSession, export_model_path: str
) -> SerializableObj:
    """
    Wrap a Keras model for inference with Spark
    """
    return SerializableObj(sparkSession, tf.keras.models.load_model, export_model_path)


def graph_model(
    sparkSession: pyspark.sql.SparkSession, export_model_path: str,
    feeds: feeds_type, fetches: fetches_type
) -> SerializableObj:
    """
    Wrap a graph model for inference with Spark
    """
    return SerializableObj(
        sparkSession, Predictor.from_graph, export_model_path, feeds, fetches)  # type: ignore


def with_graph_inference_column(
    df: pyspark.sql.dataframe.DataFrame,
    predictor_model: SerializableObj,
    output_column_name: str = "prediction",
    output_type: pyspark.sql.types.DataType = FloatType(),
    extract_fn: extract_fn_type = lambda x: pd.Series(x["score"][:, 0])
) -> pyspark.sql.dataframe.DataFrame:
    """
    Add a column in the dataframe that is the result of the inference of a model.

    :param df: the dataframe to add a prediction to
    :param predictor_model: instance of SerializableObj
    :param output_column_name: name of the newly-created inference column
    :param output_type: output type of the predictor model
    :param extract_fn: method to extract the result from the estimator output dictionary and
    transform it as a float
    :return: a new dataframe with a column named 'column_named' that is the prediction value
    """

    feature_names = list(predictor_model.ew.obj.feed_tensors.keys())
    for feature_name in feature_names:
        if feature_name not in df.columns:
            raise ValueError(f"{feature_name} not found in columns {df.columns}")

    def _inference_fn(model: artifact_type, series: Tuple[pd.Series, ...]) -> pd.Series:
        batch_size = series[0].size

        def input_fn() -> tf.data.Dataset:
            _series = []
            for serie in series:
                if len(serie.values[0].shape) == 0:
                    _series.append(serie.values.reshape(batch_size, 1))
                elif isinstance(serie.values[0], np.ndarray):
                    _series.append([e.tolist() for e in serie.values])
                else:
                    _series.append(serie)
            return tf.data.Dataset.from_tensor_slices({
                feature_name: serie
                for (feature_name, serie) in zip(feature_names, _series)
            }).batch(batch_size)

        outputs = next(model.predict(input_fn))
        return extract_fn(outputs)

    return with_inference(
        df, predictor_model, feature_names, _inference_fn, output_type,
        output_col=output_column_name
    )


def with_inference_column(
    df: pyspark.sql.dataframe.DataFrame,
    tfrecords_col: Union[pyspark.sql.Column, str],
    estimator_model: artifact_type,
    column_name: str = "prediction",
    input_key: str = "inputs",
    extract_fn: extract_fn_type = default_extract_fn
) -> pyspark.sql.dataframe.DataFrame:
    """
    Add a column in the dataframe that is the result of the inference of a model.
    This is computed line-by-line with a UDF.
    The estimator is required to take as input a TFR example

    :param df: the dataframe to add a prediction to
    :param tfrecords_col: column containing your model inputs in tfrecord format
    :param estimator_model: instance of SerializableObj
    :param column_name: name of the newly-created inference column
    :param input_key: key where the estimator function is expecting to find the TFR record
    :param extract_fn: method to extract the result from the estimator output dictionary and
    transform it as a float
    :return: a new dataframe with a column named 'column_named' that is the prediction value
    """
    def _inference_fn(model: artifact_type, series: Tuple[pd.Series, ...]) -> pd.Series:
        import tensorflow as tf
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        outputs = model.signatures[_default_signature](
            **{input_key: tf.constant(series[0])}
        )
        return extract_fn(outputs)

    return with_inference(
        df, estimator_model, [tfrecords_col], _inference_fn, FloatType(),
        output_col=column_name
    )


def with_inference(
    df: pyspark.sql.DataFrame,
    model: artifact_type,
    input_cols: List[Union[str, pyspark.sql.Column]],
    inference_fn: inference_udf,
    output_type: DataType,
    batch_size: int = 1,
    output_col: str = "prediction",
    num_threads: int = 8,
) -> pyspark.sql.dataframe.DataFrame:
    """
    :param ss: Spark sesson used to create the dataframe
    :param df: dataframe that holds the input
    :param model: model to use for inference
    :param input_col: names of the dataframe columns that will be used as inputs for inference
    :param inference_fn: function that is run to compute predictons.
    It takes as inputs a model and a list of pandas series. Each pandas serie represent
    one of the dataframe column of :param input_cols, in the same order.
    It returns a Pandas Serie
    :param output_type: output type of the predictions
    :param batch_size: batch size
    :param output_col: name of the colonne of the dataframe that will hold the predictions
    :param num_threads: Number of threads to run inter/inta-ops
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
        results = pd.Series()
        for batch in split_in_batches(rows, batch_size):
            results = results.append(inference_fn(artifacts, batch))
        print(f"FINAL RESULT: {results}")
        return results

    inference_udf = sf.pandas_udf(_inference_fn, returnType=output_type)
    # In some situation, the pandas udf can be computed more than once when the
    # output column is referenced more than once,
    # (https://issues.apache.org/jira/browse/SPARK-17728).
    # One workaround is to wrap the pandas udf in sf.explode(sf.array())
    return df.withColumn(output_col, sf.explode(sf.array(inference_udf(*input_cols))))
