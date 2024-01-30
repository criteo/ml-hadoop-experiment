import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from ml_hadoop_experiment.tensorflow import pandas_to_tensors


@pytest.mark.parametrize(
    "init_data,result_data,default_value,dtype",
    [
        ([1, 2, None, 4], [1, 2, 3, 4], 3, np.int32),
        ([1.0, 2.0, None, 4.0], [1, 2, 3, 4], 3, np.int32),
        ([1.5, 2.5, None, 4.5], [1.5, 2.5, 3.5, 4.5], 3.5, np.float64),
        (["a", "b", None, "d"], ["a", "b", "c", "d"], "c", str),
    ],
)
def test_make_feature_list_scalar(init_data, result_data, default_value, dtype):
    fun = pandas_to_tensors._make_feature_list_scalar("toto", default_value, dtype)
    list_ = []
    d = {"toto": init_data}
    df = pd.DataFrame(data=d)
    fun(df, list_)

    assert len(list_) == 1
    assert list_[0][0] == "toto"
    assert np.array_equal(list_[0][1], result_data)


@pytest.mark.parametrize(
    "init_data,type",
    [([1, 2, None, 4], np.int32), ([1.5, 2.5, None, 4.5], np.float64), (["v", None], str)],
)
def test_make_feature_list_scalar_no_default(init_data, type):
    fun = pandas_to_tensors._make_feature_list_scalar("toto", None, type)
    list_ = []
    d = {"toto": init_data}
    df = pd.DataFrame(data=d)

    with pytest.raises(ValueError):
        fun(df, list_)


def test_make_feature_list_varlen():
    fun = pandas_to_tensors._make_feature_list_varlen("toto", str)
    list_ = []
    d = {"toto": [["a", "b"], ["c", "d"], None, ["e"]]}
    df = pd.DataFrame(data=d)
    fun(df, list_)

    assert len(list_) == 3

    assert list_[0][0] == "toto/shape"
    assert np.array_equal(list_[0][1], [4, 2])

    assert list_[1][0] == "toto/indices"
    assert np.array_equal(list_[1][1], [[0, 0], [0, 1], [1, 0], [1, 1], [3, 0]])

    assert list_[2][0] == "toto/values"
    assert np.array_equal(list_[2][1], ["a", "b", "c", "d", "e"])


def test_make_feature_list_varlen_empty():
    fun = pandas_to_tensors._make_feature_list_varlen("toto", str)
    list_ = []
    d = {"toto": [[], []]}
    df = pd.DataFrame(data=d)
    fun(df, list_)

    assert len(list_) == 3

    assert list_[0][0] == "toto/shape"
    assert np.array_equal(list_[0][1], [2, 0])

    assert list_[1][0] == "toto/indices"
    assert np.array_equal(list_[1][1], np.empty((0, 2), dtype=np.int64))

    assert list_[2][0] == "toto/values"
    assert np.array_equal(list_[2][1], np.array([], dtype=str))


def test_generate_create_tensor_fn():
    specs = {"dim": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)}
    d = {"dim": [1, 2]}
    df = pd.DataFrame(data=d)

    make_tensors = pandas_to_tensors.generate_create_tensor_fn(specs)

    result = make_tensors(df)

    assert len(result) == 1
    assert isinstance(result["dim"], np.ndarray)
    assert np.array_equal(result["dim"], [1, 2])
