import numpy as np
import pytest

from ml_hadoop_experiment.tensorflow import numpy_to_sparse_tensors


@pytest.mark.parametrize(
    "sizes,expected", [([2, 3], [0, 1, 0, 1, 2]), ([2, 0, 3], [0, 1, 0, 1, 2]), ([2, 1, 0], [0, 1, 0]), ([0, 0, 0], [])]
)
def test_generate_increments(sizes, expected):
    result = numpy_to_sparse_tensors._generate_increments(np.array(sizes))
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "features,expected_indices,expected_values,expected_shape",
    [
        ([[7, 8], [10, 11, 12]], [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], [7, 8, 10, 11, 12], [2, 3]),
        ([[], []], np.empty((0, 2), dtype=np.int64), [], [2, 0]),
    ],
)
def test_create_sparse_np_stacked_int(features, expected_indices, expected_values, expected_shape):
    indices, values, shape = numpy_to_sparse_tensors.create_sparse_np_stacked(features, np.int64)
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(shape, expected_shape)


@pytest.mark.parametrize(
    "features,expected_indices,expected_values,expected_shape",
    [
        ([['a', 'b'], ['c', 'd', 'e']], [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], ['a', 'b', 'c', 'd', 'e'], [2, 3]),
        ([[], []], np.empty((0, 2), dtype=np.int64), np.array([], dtype=str), [2, 0]),
    ],
)
def test_create_sparse_np_stacked_str(features, expected_indices, expected_values, expected_shape):
    indices, values, shape = numpy_to_sparse_tensors.create_sparse_np_stacked(features, str)
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(shape, expected_shape)
