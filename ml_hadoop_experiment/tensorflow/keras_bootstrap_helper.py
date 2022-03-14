from typing import Callable, Dict, List, Union, Optional, Any

import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras import Input, layers, Model

try:
    import keras as legacy_keras
except ImportError:
    pass


def build_eval_only_model(model: Union[Model, legacy_keras.Model],
                          metrics: List[Any] = None) -> Model:
    """builds a model for evaluation only.

    Arguments:
        model: Keras Model from which we will get the definition of output layers
        and loss.
        metrics: any collection of metrics such as described in keras.model.compile().
    Returns:
        a Keras model allowing to compute loss of the input model and provided metrics
        based on previously computed predictions fed as inputs
    """
    inputs = []
    outputs = []
    for name, shape, out in zip(model.output_names, model.output_shape, model.outputs):
        # we strip the first dimension of the tensor, as Keras Input must not contain batch size
        in_layer = Input(shape=tuple(shape[1:]), name=f"pred_{name}", dtype=out.dtype)
        inputs.append(in_layer)
        outputs.append(layers.Lambda(lambda x: x, name=name)(in_layer))

    eval_model = Model(inputs=inputs, outputs=outputs)

    optimizer = 'adam'  # will not be used, the model is not trainable
    eval_model.compile(optimizer=optimizer, loss=model.loss, weighted_metrics=metrics)
    return eval_model


def evaluate_bootstrap(model: Union[Model, legacy_keras.Model],
                       df: pd.DataFrame,
                       nb_bootstrap: int,
                       input_transform: Callable[[pd.DataFrame], List[np.ndarray]],
                       label_transform: Callable[[pd.DataFrame], List[np.ndarray]],
                       metrics: List[Any] = None,
                       weight_transform: Callable[[pd.DataFrame], List[np.ndarray]] = None,
                       seed: Optional[int] = None) -> Dict[str, List[float]]:
    """Boostrap the evaluation of loss and metrics from a model

    Arguments:
        model: Keras Model on which to compute bootstraps
        df: pandas dataframe containing inputs and labels
        nb_bootstrap: number of bootstrap iterations
        input_transform: function converting the input dataframe into list of numpy arrays
            with sizes corresponding to the input of the model
        label_transform: function converting the input dataframe into list of numpy arrays
            with sizes corresponding to the outputs of the model
        metrics: any collection of metrics such as described in keras.model.compile()
        weight_transform: function converting the dataframe into as many weight columns as inputs
            if this is set, all metrics will be weighted
        seed: optional int allowing to reset the random bootstrap sampler to a fixed seed
    Returns:
        A dict of metric_name -> list of values of size nb_bootstrap

    """
    eval_only = build_eval_only_model(model, metrics)
    results = []

    predictions = model.predict(input_transform(df), verbose=0)
    labels = label_transform(df)
    weight_columns = None if weight_transform is None else weight_transform(df)

    n = len(df)

    if seed is not None:
        np.random.seed(seed)

    for _ in tqdm(range(nb_bootstrap)):
        bootstrap_indexes = np.random.randint(n, size=n)
        (indexes, counts) = np.unique(bootstrap_indexes, return_counts=True)
        bootstrap_weights = np.zeros(n)
        bootstrap_weights[indexes] = counts

        if weight_columns is None:
            sample_weights = {name: bootstrap_weights for name in eval_only.output_names}
        else:
            sample_weights = {name: bootstrap_weights * column
                              for name, column in zip(eval_only.output_names, weight_columns)}

        results.append(eval_only.evaluate(predictions,
                                          labels,
                                          sample_weight=sample_weights,
                                          verbose=0))
    metrics_names = eval_only.metrics_names if weight_transform is not None \
        else map(lambda s: s.replace('weighted_', ''), eval_only.metrics_names) \
        # this is needed for tensoflow 1.15 which adds a 'weighted_' to the metrics name

    return {metric: values for (metric, values) in zip(metrics_names,
                                                       np.array(results).T.tolist())}
