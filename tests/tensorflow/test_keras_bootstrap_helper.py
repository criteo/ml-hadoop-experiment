import pytest
import pandas as pd
import numpy as np
from tensorflow import keras
import keras as legacy_keras

from ml_hadoop_experiment.tensorflow.keras_bootstrap_helper import build_eval_only_model,\
    evaluate_bootstrap


def compile_model(model, weighted):
    if weighted:
        model.compile(optimizer='adam', loss='mse',
                      weighted_metrics=['RootMeanSquaredError'])
    else:
        model.compile(optimizer='adam', loss='mse',
                      metrics=['RootMeanSquaredError'])


def keras_model_fn(weighted_metrics: bool):
    feature1 = keras.Input(shape=(1,), name='feature1')
    feature2 = keras.Input(shape=(1,), name='feature2')

    sum_inputs = keras.layers.Add()([feature1, feature2])
    prod = keras.layers.Multiply()([feature1, feature2])

    model = keras.Model(inputs=[feature1, feature2], outputs=[sum_inputs, prod])
    compile_model(model, weighted_metrics)
    return model


def legacy_keras_model_fn(weighted_metrics):
    feature1 = legacy_keras.Input(shape=(1,), name='feature1')
    feature2 = legacy_keras.Input(shape=(1,), name='feature2')

    sum_inputs = legacy_keras.layers.Add()([feature1, feature2])
    prod = legacy_keras.layers.Multiply()([feature1, feature2])

    model = legacy_keras.Model(inputs=[feature1, feature2], outputs=[sum_inputs, prod])
    compile_model(model, weighted_metrics)
    return model


@pytest.fixture
def df():
    df = pd.DataFrame({'feature1': [13, 2, 6, 33, 5],
                       'feature2': [3, 1, 7, 3, 9]})
    df['expected_sum'] = df['feature1'] + df['feature2']
    df['expected_prod'] = df['feature1'] * df['feature2']
    df['noise'] = np.random.random(len(df))
    df['weight_add'] = [1.0, 2.0, 3.0, 4.0, 5.0]
    df['weight_multiply'] = [5.0, 4.0, 3.0, 2.0, 1.0]
    return df


def input_format_fn(df):
    return [np.stack(df['feature1']), np.stack(df['feature2'])]


@pytest.mark.parametrize("model_build_fn", [keras_model_fn, legacy_keras_model_fn])
def test_build_eval_only_model(model_build_fn, df):
    keras_model = model_build_fn(weighted_metrics=False)
    predictions = keras_model.predict(input_format_fn(df))

    def label_format_fn(df):
        # errors on labels to check correctness of loss/metrics computation
        return [np.stack(df['expected_sum'] + 2), np.stack(df['expected_prod'] + 4)]

    results = keras_model.evaluate(input_format_fn(df), label_format_fn(df))

    eval_only_model = build_eval_only_model(keras_model, ['RootMeanSquaredError'])
    eval_only = eval_only_model.evaluate(predictions, label_format_fn(df))

    # losses and metrics: total loss, loss_add, loss_multiply, RMSE_add, RMSE_multiply
    assert(results == [20.0, 4.0, 16.0, 2.0, 4.0])
    assert(eval_only == [20.0, 4.0, 16.0, 2.0, 4.0])


@pytest.mark.parametrize("model_build_fn", [keras_model_fn, legacy_keras_model_fn])
def test_evaluate_bootstrap(model_build_fn, df):
    def label_format_fn(df):
        return [np.stack(df['expected_sum'] + df['noise']),
                np.stack(df['expected_prod'] + 2 * df['noise'])]

    keras_model = model_build_fn(weighted_metrics=False)
    n = len(df)
    np.random.seed(0)
    boot1 = df.iloc[np.random.randint(n, size=n)]
    results_1 = keras_model.evaluate(input_format_fn(boot1), label_format_fn(boot1))
    boot2 = df.iloc[np.random.randint(n, size=n)]
    results_2 = keras_model.evaluate(input_format_fn(boot2), label_format_fn(boot2))

    results = evaluate_bootstrap(
        keras_model, df, 2, input_format_fn, label_format_fn, ['RootMeanSquaredError'], seed=0)
    for (idx, (name, values)) in enumerate(results.items()):
        assert np.isclose(values[0], results_1[idx], rtol=1e-09, atol=1e-06)
        assert np.isclose(values[1], results_2[idx], rtol=1e-09, atol=1e-06)


@pytest.mark.parametrize("model_build_fn", [keras_model_fn, legacy_keras_model_fn])
def test_evaluate_bootstrap_weight_metrics(model_build_fn, df):
    def label_format_fn(df):
        return [np.stack(df['expected_sum'] + df['noise']),
                np.stack(df['expected_prod'] + 2 * df['noise'])]

    def weight_format_fn(df):
        return [np.stack(df['weight_add']), np.stack(df['weight_multiply'])]

    keras_model = model_build_fn(weighted_metrics=True)
    n = len(df)
    np.random.seed(0)
    boot1 = df.iloc[np.random.randint(n, size=n)]
    results_1 = keras_model.evaluate(input_format_fn(boot1), label_format_fn(boot1),
                                     sample_weight=weight_format_fn(boot1))
    boot2 = df.iloc[np.random.randint(n, size=n)]
    results_2 = keras_model.evaluate(input_format_fn(boot2), label_format_fn(boot2),
                                     sample_weight=weight_format_fn(boot2))

    results = evaluate_bootstrap(
        keras_model, df, 2, input_format_fn, label_format_fn,
        ['RootMeanSquaredError'], weight_format_fn, seed=0)
    for (idx, (name, values)) in enumerate(results.items()):
        assert np.isclose(values[0], results_1[idx], rtol=1e-09, atol=1e-06)
        assert np.isclose(values[1], results_2[idx], rtol=1e-09, atol=1e-06)
