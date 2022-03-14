import logging
from typing import (
    Union,
    Any
)
from functools import singledispatch

import tensorflow as tf
import pyspark

from ml_hadoop_experiment.tensorflow.tfrecords import features_specs_type


_logger = logging.getLogger(__name__)

features_spec_type = Union[
    tf.io.FixedLenFeature,
    tf.io.VarLenFeature
]


def get_exact_sparksql_type(tensorflow_type: tf.DType) -> pyspark.sql.types.DataType:
    exact_conversions = {
        tf.int32: pyspark.sql.types.IntegerType(),
        tf.int64: pyspark.sql.types.LongType(),
        tf.float32: pyspark.sql.types.FloatType(),
        tf.float64: pyspark.sql.types.DoubleType(),
        tf.string: pyspark.sql.types.StringType(),
    }
    return exact_conversions[tensorflow_type]


def can_convert_x_to_y(x: pyspark.sql.types.DataType, y: tf.DType) -> bool:
    allowed_conversions = {
        pyspark.sql.types.IntegerType(): [tf.int32, tf.int64],
        pyspark.sql.types.LongType(): [tf.int64],
        pyspark.sql.types.FloatType(): [tf.float32, tf.float64],
        pyspark.sql.types.DoubleType(): [tf.float64],
        pyspark.sql.types.StringType(): [tf.string],
    }
    return y in allowed_conversions.get(x, [])


@singledispatch
def exact_data_type_for_feature_spec(feature_spec: Any) -> pyspark.sql.types.DataType:
    raise TypeError(f"Feature spec type {feature_spec} not supported.")


@exact_data_type_for_feature_spec.register(tf.io.FixedLenFeature)
def exact_data_type_for_fixed_feature_spec(feature_spec: tf.io.FixedLenFeature
                                           ) -> pyspark.sql.types.DataType:
    df_type = get_exact_sparksql_type(feature_spec.dtype)
    order = len(feature_spec.shape)
    # if order > 0, feature spec is a real tensor, create nested array types:
    for _ in range(order):
        df_type = pyspark.sql.types.ArrayType(df_type, False)
    return df_type


@exact_data_type_for_feature_spec.register(tf.io.VarLenFeature)
def exact_data_type_for_varlen_feature_spec(feature_spec: tf.io.VarLenFeature
                                            ) -> pyspark.sql.types.DataType:
    df_type = get_exact_sparksql_type(feature_spec.dtype)
    return pyspark.sql.types.ArrayType(df_type, False)


def exact_structfield_for_feature_spec(name: str, feature_spec: features_spec_type
                                       ) -> pyspark.sql.types.StructField:
    datatype = exact_data_type_for_feature_spec(feature_spec)
    is_nullable = feature_spec.default_value is not None
    return pyspark.sql.types.StructField(name, datatype, is_nullable)


@singledispatch
def is_datatype_compatible_with_feature_spec(feature_spec: Any,
                                             datatype: pyspark.sql.types.DataType) -> bool:
    raise NotImplementedError(f'Unsupported type for feature_spec {feature_spec} : '
                              f'{type(feature_spec)}')


@is_datatype_compatible_with_feature_spec.register(tf.io.FixedLenFeature)
def is_datatype_compatible_with_fixed_feature_spec(feature_spec: tf.io.FixedLenFeature,
                                                   datatype: pyspark.sql.types.DataType) -> bool:
    if datatype == exact_data_type_for_feature_spec(feature_spec):
        return True
    df_rank = 0
    datatype_tmp = datatype
    while isinstance(datatype_tmp, pyspark.sql.types.ArrayType):
        datatype_tmp = datatype_tmp.elementType
        df_rank += 1
    if df_rank == len(feature_spec.shape) or (df_rank == 1 and len(feature_spec.shape) > 1):
        if can_convert_x_to_y(datatype_tmp, feature_spec.dtype):
            return True
        else:
            _logger.info(f"No conversion from {datatype} to {feature_spec.dtype} can be "
                         f"performed ")
            return False
    else:
        _logger.info(f"Rank of schema {datatype} differs from {feature_spec}")
        return False


@is_datatype_compatible_with_feature_spec.register(tf.io.VarLenFeature)
def is_datatype_compatible_with_varlen_feature_spec(feature_spec: tf.io.VarLenFeature,
                                                    datatype: pyspark.sql.types.DataType) -> bool:
    if datatype == exact_data_type_for_feature_spec(feature_spec):
        return True
    # allow scalar columns to be sent to VarLenFeature
    if can_convert_x_to_y(datatype, feature_spec.dtype):
        return True

    # allow compatibility with 1-D array if inner types are the same
    if isinstance(datatype, pyspark.sql.types.ArrayType) and can_convert_x_to_y(
            datatype.elementType, feature_spec.dtype):
        return True

    _logger.info(f"No conversion found from {datatype} to {feature_spec}")
    return False


def is_structfield_compatible_with_feature_spec(structfield: pyspark.sql.types.StructField,
                                                name: str,
                                                feature_spec: features_spec_type) -> bool:
    if structfield.name != name:
        _logger.info(f"Mismatched names between structField {structfield.name} and feature spec "
                     f"{name}.")
        return False

    compat = is_datatype_compatible_with_feature_spec(feature_spec, structfield.dataType)
    if not compat:
        return False

    if isinstance(feature_spec, tf.io.FixedLenFeature) and \
            structfield.nullable is True and feature_spec.default_value is None:
        _logger.info("Column is nullable but feature spec doesn't handle missing values")
        return False

    return True


def is_dataframe_compatible_with_feature_spec(df: pyspark.sql.DataFrame,
                                              name: str,
                                              feature_spec: features_spec_type) -> bool:
    try:
        field = df.schema[name]
    except KeyError:
        _logger.info(f"No column named {name} in dataframe")
        return False

    return is_structfield_compatible_with_feature_spec(field, name, feature_spec)


def is_dataframe_compatible_with_feature_specs(df: pyspark.sql.DataFrame,
                                               feature_spec_dict: features_specs_type) -> bool:
    """Check if the schema of a dataframe is compatible with a model's feature specs

    If verbose is set to True, we will print the list of incompatible columns and the reason of
    the incompatibility.
    """
    global_compatibility = True
    for name, feature_spec in feature_spec_dict.items():
        compat = is_dataframe_compatible_with_feature_spec(df, name, feature_spec)
        if not compat:
            global_compatibility = False
            _logger.info(f"Dataframe has no column compatible with feature spec {name}: "
                         f"{feature_spec}.")
    return global_compatibility
