import math
import typing


class Eval_config(typing.NamedTuple):
    throttle_secs: int
    save_checkpoints_steps: int
    evaluation_batch_size: int
    steps: int


def get_eval_params(
    nb_workers: int,
    nb_evaluators: int,
    nb_examples_before_eval: int,
    max_eval_batch_size: int = 100000,
) -> "Eval_config":
    """
    Helper to compute coherent parameters for model evaluation
    """
    for var, val in {
        "nb_examples_before_eval": nb_examples_before_eval,
        "nb_workers": nb_workers,
        "nb_evaluators": nb_evaluators,
        "max_eval_batch_size": max_eval_batch_size,
    }.items():
        if val <= 0:
            raise ValueError(f"{var} can't be <= 0. Got {val}")

    save_checkpoints_steps = nb_examples_before_eval

    # We assure workers process examples as fast as evaluators
    # In order to try to process every checkoint as possible,
    # evaluation time of a checkpoint must be ~ time to generate a checkpoint
    # We don't take into account of vcores/inter_op/intra_op_parallelism as
    # we don't know how they impact training speed
    evaluation_batch_size = (save_checkpoints_steps / nb_workers) * nb_evaluators
    steps = 1
    if evaluation_batch_size > max_eval_batch_size:
        steps = math.ceil(evaluation_batch_size / max_eval_batch_size)
        evaluation_batch_size = evaluation_batch_size / steps
    # Don't really know how to compute this parameter so le'ts be consevative
    throttle_secs = 5
    return Eval_config(throttle_secs, save_checkpoints_steps, int(evaluation_batch_size), steps)
