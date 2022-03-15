import typing

import pytest

from ml_hadoop_experiment.tensorflow.evaluator_conf_helper import (
    Eval_config,
    get_eval_params
)


class User_params(typing.NamedTuple):
    nb_workers: int
    nb_evaluators: int
    nb_examples_before_eval: int
    max_eval_batch_size: int


tests_with_good_params = [
    (User_params(50, 1, 50000, 100000), Eval_config(5, 50000, 1000, 1)),
    (User_params(50, 1, 5000000, 50000), Eval_config(5, 5000000, 50000, 2))
]

tests_with_bad_params = [
    User_params(0, 1, 5000000, 50000),
    User_params(50, 0, 5000000, 50000),
    User_params(50, 1, 0, 50000),
    User_params(50, 1, 5000000, 0),
    User_params(-50, 1, 5000000, 50000),
    User_params(50, -1, 5000000, 50000),
    User_params(50, 1, -5000000, 50000),
    User_params(50, 1, 5000000, -50000)
]


@pytest.mark.parametrize("test_set,expected_result", tests_with_good_params)
def test_get_eval_params_with_good_params(test_set, expected_result):
    assert get_eval_params(*test_set) == expected_result


@pytest.mark.parametrize("test_set", tests_with_bad_params)
def test_get_eval_params_with_bad_params(test_set):
    with pytest.raises(ValueError):
        get_eval_params(*test_set)
