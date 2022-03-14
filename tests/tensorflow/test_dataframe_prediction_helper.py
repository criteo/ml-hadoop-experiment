import os
import pickle
import tempfile

from pyspark.sql.types import ArrayType, FloatType
from tensorflow import keras
from tensorflow.keras.layers import Input, Add, Multiply
import tensorflow as tf
import pandas as pd
import numpy as np

from ml_hadoop_experiment.tensorflow.dataframe_prediction_helper import \
    keras_model, graph_model, with_graph_inference_column, with_inference
from ml_hadoop_experiment.common.spark_inference import _SerializableObjWrapper


def _loader_mock(model_path):
    return model_path


def test_serde_estimator_wrapper():
    estimator_wrapper = _SerializableObjWrapper(_loader_mock, "my path")
    pickle.loads(pickle.dumps(estimator_wrapper))


def test_keras_inference_mono_head(local_spark_session):
    # User-define inference function
    def inference_fn(model, series):
        dataset = tf.data.Dataset.from_tensor_slices({
            "feature1": series[0],
            "feature2": series[1]
        })
        predictions = model.predict(dataset)
        return pd.Series(predictions["score"])

    # Model definition
    inputs = [
        Input(shape=(1,), dtype="int64", name="feature1"),
        Input(shape=(1,), dtype="int64", name="feature2")
    ]
    add_layer = Add()(inputs)
    dummy_model = keras.Model(inputs=inputs, outputs={"score": add_layer})

    # inference
    with tempfile.TemporaryDirectory() as tmp:
        dummy_model.save(tmp)
        ss = local_spark_session
        df = ss.createDataFrame([[3, 113], [33, 333]], ["feature1", "feature2"])
        with keras_model(ss, tmp) as model:
            pdf = with_inference(
                df, model, ["feature1", "feature2"], inference_fn, FloatType()
            ).toPandas()
            assert pdf["prediction"].equals(
                (pdf["feature1"] + pdf["feature2"]).astype('float32'))


def test_keras_inference_multi_head(local_spark_session):

    # user-defined function
    def inference_fn(model, series):
        dataset = tf.data.Dataset.from_tensor_slices({
            "feature1": series[0],
            "feature2": series[1]
        })
        predictions = model.predict(dataset)
        scores = np.array([v for k, v in predictions.items() if k in ["score_add", "score_mul"]])
        scores = np.squeeze(np.dstack(scores), axis=1)
        return pd.Series(scores.tolist())

    # model definition
    inputs = [
        Input(shape=(1,), dtype="int64", name="feature1"),
        Input(shape=(1,), dtype="int64", name="feature2")
    ]
    add_layer = Add()(inputs)
    mul_layer = Multiply()(inputs)
    dummy_model = keras.Model(
        inputs=inputs, outputs={"score_add": add_layer, "score_mul": mul_layer}
    )

    # inference
    with tempfile.TemporaryDirectory() as tmp:
        dummy_model.save(tmp)
        ss = local_spark_session
        df = ss.createDataFrame([[3, 113], [33, 333]], ["feature1", "feature2"])
        with keras_model(ss, tmp) as model:
            pdf = with_inference(
                df, model, ["feature1", "feature2"], inference_fn, ArrayType(FloatType())
            ).toPandas()
            for row in pdf.itertuples():
                assert (row.feature1 + row.feature2) == row.prediction[0]
                assert (row.feature1 * row.feature2) == row.prediction[1]


def test_graph_inference_mono_head(local_spark_session):
    df = local_spark_session.createDataFrame([[3, 113], [33, 333]], ["partnerid", "contextid"])
    graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_graph.pb")
    with graph_model(
        local_spark_session, graph_path, ["partnerid", "contextid"], ["add/add"]
    ) as model:
        pdf = with_graph_inference_column(
            df, model, extract_fn=lambda x: pd.Series(x["add/add"][:, 0])
        ).toPandas()
        assert pdf["prediction"].equals((pdf["partnerid"] + pdf["contextid"]).astype('float32'))


def test_with_inference_computed_once(local_spark_session):
    # User-define inference function
    def inference_fn(counter, features):
        counter["value"] += 1
        return pd.Series(zip(features[0] + counter["value"], features[1] + counter["value"]))

    ss = local_spark_session
    df = ss.createDataFrame([(2., 12.), (8., 18.)], ["feature1", "feature2"])
    df_result = with_inference(
        df, {"value": 0}, ["feature1", "feature2"], inference_fn, ArrayType(FloatType()),
        batch_size=100
    )
    df_result = df_result \
        .withColumn("predictions1", df_result["prediction"].getItem(0)) \
        .withColumn("predictions2", df_result["prediction"].getItem(1)) \
        .drop("prediction")
    pdf = df_result.toPandas()
    _compare(pd.Series([3., 9.]), pdf["predictions1"])
    _compare(pd.Series([13., 19.]), pdf["predictions2"])


def _compare(expected: pd.Series, actual: pd.Series) -> None:
    assert expected.size == actual.size
    for i in range(expected.size):
        assert expected[i] == actual[i]
