from typing import (
    Dict,
    Union,
    Tuple,
    Callable,
    List,
    Optional,
    Any,
    Generator
)

import tensorflow as tf
import numpy as np
import pandas as pd

from ml_hadoop_experiment.tensorflow.numpy_to_sparse_tensors import \
    create_sparse_np_stacked


add_to_list_type = Callable[
    [pd.DataFrame, List[Tuple[str, np.array]]],
    None
]

features_specs_type = Dict[
    str,
    Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
]


def _make_feature_list_scalar(key: str, default_value: Optional[Union[int, str, float]],
                              dtype: Any) -> add_to_list_type:

    if default_value is not None:
        if (isinstance(default_value, int) and dtype != np.int32 and dtype != np.int64) or \
                (isinstance(default_value, str) and dtype != np.str) or \
                (isinstance(default_value, float) and dtype != np.float32 and dtype != np.float64):
            raise ValueError(f"default_value {default_value} of type {type(default_value)} "
                             f"incompatible with feature of type {dtype}")

    def add_tensors(pandas_df: pd.DataFrame, tensors: List[Tuple[str, tf.Tensor]]) -> None:
        # WARNING we have to call astype(dtype) because the from_record method may have generated
        # an incorrect type for this column. If we call astype with the same type it will be a
        # no-op anyway.

        features: np.array = None
        if default_value is not None:
            features = pandas_df[key].fillna(default_value).astype(dtype).values
        else:
            if pandas_df[key].isnull().values.any():
                raise ValueError(f"For key {key} some inputs are null in the dataframe, "
                                 f"and no default value was provided")
            else:
                features = pandas_df[key].astype(dtype).values
        tensors.append((key, features))

    return add_tensors


def _make_feature_list_varlen(key: str, dtype: Any) -> add_to_list_type:
    def add_tensors(pandas_df: pd.DataFrame, list_: List[Tuple[str, np.array]]) -> None:

        def iter_() -> Generator:
            for v in pandas_df[key].values:
                if v is None:  # pandas will have parsed missing feature list as None: convert to []
                    yield np.array([], dtype)
                else:
                    yield np.array(v, dtype)

        feature_list = list(iter_())
        indices, values, dense_shape = create_sparse_np_stacked(feature_list, dtype)
        list_.append((key + "/shape", dense_shape))
        list_.append((key + "/indices", indices))
        list_.append((key + "/values", values))

    return add_tensors


def generate_create_tensor_fn(feature_spec: features_specs_type) -> Callable[[pd.DataFrame],
                                                                             Dict[str, np.array]]:
    """
    From a feature_spec, generate all the necessary converters that will be able to transform
    a pandas dataframe to a container of tensors.
    Return a method that, when called on a dataframe, will generate all the "raw" tensors
    for all the dimensions in the feature_spec. This return value can then be directly sent
    to the tensorflow inference function.
    """
    generators: List[add_to_list_type] = []

    tf_to_np = {tf.int32: np.int32,
                tf.int64: np.int64,
                tf.float32: np.float32,
                tf.float64: np.float64,
                tf.string: np.str}

    for key, value in feature_spec.items():

        if isinstance(value, tf.io.VarLenFeature):
            if value.dtype in tf_to_np:
                gen = _make_feature_list_varlen(key, tf_to_np[value.dtype])
            else:
                raise NotImplementedError(f'{key} has unknown type: {value.dtype}')
        elif isinstance(value, tf.io.FixedLenFeature):
            if len(value.shape) == 0 or (len(value.shape) == 1 and value.shape[0] == 1):
                if value.dtype in tf_to_np:
                    gen = _make_feature_list_scalar(key, value.default_value, tf_to_np[value.dtype])
                else:
                    raise NotImplementedError(f'{key} has unknown type: {value.dtype}')
            else:
                raise NotImplementedError(f"spec for FixedLenFeature of non-scalar shape not"
                                          f"supported (got {value.shape} for key {key})")
        else:
            raise NotImplementedError(f'{key} has unknown type: {type(value)}')

        generators.append(gen)

    def make_tensors_from_pandas_dataframe(pandas_df: pd.DataFrame) -> Dict[str, np.array]:
        tensors: List[Tuple[str, np.array]] = []
        for generator in generators:
            generator(pandas_df, tensors)

        # sanity check that all tensors have been expanded to the same size:
        items_count = pandas_df.shape[0]
        for k, v in tensors:
            if "/" not in k:  # numpy array representing a dense tensor
                assert items_count == v.shape[0]
            elif k.endswith("/shape"):  # numpy array representing shape of a sparse array
                assert items_count == v[0]

        return dict(tensors)

    return make_tensors_from_pandas_dataframe
