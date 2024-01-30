import pytest
import tensorflow as tf
import pyspark

from ml_hadoop_experiment.tensorflow.dataframe_to_tf_helper import (
    is_datatype_compatible_with_feature_spec,
    is_structfield_compatible_with_feature_spec,
)


@pytest.mark.parametrize(
    "column_type,feature_spec,expected_result",
    [
        # test integer -> integer scalar conversions
        (pyspark.sql.types.LongType(), tf.io.FixedLenFeature([], dtype=tf.int64), True),
        (pyspark.sql.types.LongType(), tf.io.FixedLenFeature([], dtype=tf.int32), False),
        (pyspark.sql.types.IntegerType(), tf.io.FixedLenFeature([], dtype=tf.int64), True),
        (pyspark.sql.types.IntegerType(), tf.io.FixedLenFeature([], dtype=tf.int32), True),
        # test floating point -> floating point scalar conversions
        (pyspark.sql.types.DoubleType(), tf.io.FixedLenFeature([], dtype=tf.float32), False),
        (pyspark.sql.types.FloatType(), tf.io.FixedLenFeature([], dtype=tf.float32), True),
        (pyspark.sql.types.DoubleType(), tf.io.FixedLenFeature([], dtype=tf.float64), True),
        (pyspark.sql.types.FloatType(), tf.io.FixedLenFeature([], dtype=tf.float64), True),
        # test scalar column -> tensor conversions
        (pyspark.sql.types.LongType(), tf.io.FixedLenFeature([1], dtype=tf.int64), False),
        (pyspark.sql.types.LongType(), tf.io.FixedLenFeature([2], dtype=tf.int64), False),
        (pyspark.sql.types.LongType(), tf.io.FixedLenFeature([2, 2], dtype=tf.int64), False),
        # test array column -> tensor conversions
        (
            pyspark.sql.types.ArrayType(pyspark.sql.types.LongType(), False),
            tf.io.FixedLenFeature([1], dtype=tf.int64),
            True,
        ),
        (
            pyspark.sql.types.ArrayType(pyspark.sql.types.LongType(), False),
            tf.io.FixedLenFeature([2], dtype=tf.int64),
            True,
        ),
        (
            pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType(), False),
            tf.io.FixedLenFeature([2], dtype=tf.int64),
            True,
        ),
        (
            pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType(), False),
            tf.io.FixedLenFeature([2, 2], dtype=tf.int64),
            True,
        ),
        (
            pyspark.sql.types.ArrayType(
                pyspark.sql.types.ArrayType(pyspark.sql.types.LongType(), False), False
            ),
            tf.io.FixedLenFeature([2], dtype=tf.int64),
            False,
        ),
        (
            pyspark.sql.types.ArrayType(
                pyspark.sql.types.ArrayType(pyspark.sql.types.LongType(), False), False
            ),
            tf.io.FixedLenFeature([2, 2], dtype=tf.int64),
            True,
        ),
        # test conversion to VarLenFeature
        (pyspark.sql.types.IntegerType(), tf.io.VarLenFeature(tf.int64), True),
        (
            pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType(), False),
            tf.io.VarLenFeature(tf.int64),
            True,
        ),
    ],
)
def test_is_datatype_compatible_with_feature_spec(column_type, feature_spec, expected_result):
    result = is_datatype_compatible_with_feature_spec(feature_spec, column_type)
    assert expected_result == result


@pytest.mark.parametrize(
    "structfield,name,feature_spec,expected_result",
    [
        # different name
        (
            pyspark.sql.types.StructField("tata", pyspark.sql.types.LongType()),
            "toto",
            tf.io.FixedLenFeature([], dtype=tf.int64),
            False,
        ),
        # incompatible data type
        (
            pyspark.sql.types.StructField("toto", pyspark.sql.types.LongType()),
            "toto",
            tf.io.FixedLenFeature([], dtype=tf.int32),
            False,
        ),
        # incompatible nullability
        (
            pyspark.sql.types.StructField("toto", pyspark.sql.types.LongType(), nullable=True),
            "toto",
            tf.io.FixedLenFeature([], dtype=tf.int64),
            False,
        ),
        # OK
        (
            pyspark.sql.types.StructField("toto", pyspark.sql.types.LongType(), nullable=False),
            "toto",
            tf.io.FixedLenFeature([], dtype=tf.int64),
            True,
        ),
        (
            pyspark.sql.types.StructField("toto", pyspark.sql.types.LongType(), nullable=True),
            "toto",
            tf.io.FixedLenFeature([], dtype=tf.int64, default_value=1),
            True,
        ),
    ],
)
def test_is_structfield_compatible_with_feature_spec(
    structfield, name, feature_spec, expected_result
):
    result = is_structfield_compatible_with_feature_spec(structfield, name, feature_spec)
    assert expected_result == result
