from typing import (
    Dict,
    Union,
    Tuple,
    Any
)

import tensorflow as tf
import numpy as np


features_specs_type = Dict[
    str,
    Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
]


# generates lists starting from 0 to k-1 for each item "k" in the list
# https://stackoverflow.com/questions/53422006/how-to-do-this-operation-in-numpy-chaining-of
# -tiling-operation/
# eg [2,3] --> [0,1,0,1,2]
def _generate_increments(sizes: Any) -> np.array:
    if len(sizes) == 0:
        return np.array([], np.int64)

    # find index of last non-0:
    idx = len(sizes) - 1
    while sizes[idx] == 0:
        idx -= 1
        if idx < 0:
            return np.zeros(0, np.int64)

    steps = sizes[:idx + 1]
    cumulative_steps = steps.cumsum()

    reset_range = np.zeros(cumulative_steps[-1], np.int64)
    reset_range[np.flip(cumulative_steps[:-1])] = np.flip(steps[:-1])
    return np.arange(cumulative_steps[-1]) - reset_range.cumsum()


# Generate a sparse tensor by stacking variable-length features.
# This happens when each feature is an array
def create_sparse_np_stacked(features: Any, dtype: Any) -> \
        Tuple[np.array, np.array, np.array]:
    feature_lengths = np.array([len(f) for f in features])
    max_feature_length = np.max(feature_lengths)

    dense_shape = np.array([len(features), max_feature_length])

    if max_feature_length == 0:
        # every list is empty, return empty sparse tensor representation
        # Indices is supposed to be a N x 2 array, where N is the number of items. Here because
        # we have no items I'm creating an (empty) 0 x 2 array.
        indices = np.empty((0, 2), dtype=np.int64)
        values = np.array([], dtype=dtype)
        return indices, values, dense_shape

    # stack all lists in a numpy array.
    # Removing empty array beforehand is an optim that make hstack roughly 20% faster.
    values = np.hstack([f for f in features if len(f) > 0])

    x = np.repeat(np.arange(len(features)), feature_lengths, axis=0)
    y = _generate_increments(feature_lengths)

    indices = np.dstack([x, y]).reshape([-1, 2])

    return indices, values, dense_shape
