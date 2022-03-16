import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_hadoop_experiment.sklearn.spark_inference import with_inference_column

@pytest.fixture()
def sklearn_lr_model():
    """Returns a simple trained LR model, with test data and expected predictions
    """
    df_train = pd.DataFrame([{'a': 1.5, 'b': 2.5}, {'a': .5, 'b': 4.5}])
    targets = pd.Series([True, False])

    lr = LogisticRegression()
    lr.fit(df_train, targets)

    return lr


def test_with_inference_column_pos(local_spark_session, sklearn_lr_model):
    """Test default column name with extract fn
    """
    df_test = pd.DataFrame([{'a': 2.5, 'b': 6.5}, {'a': 1.5, 'b': 3.5}])
    expected_pos = [0.19610801, 0.53911309]

    def extract_pos(probas):
        return pd.Series(np.array(probas[:, 1]))

    pred_df = with_inference_column(local_spark_session.createDataFrame(df_test),
                                    sklearn_lr_model,
                                    extract_pos).toPandas()

    for pred, exp in zip(pred_df.prediction.values, expected_pos):
        assert pred == pytest.approx(exp)


def test_with_inference_column_neg(local_spark_session, sklearn_lr_model):
    """Test custom column name with another extract fn not returning a pd.Series
    """
    df_test = pd.DataFrame([{'a': 2.5, 'b': 6.5}, {'a': 1.5, 'b': 3.5}])
    expected_neg = [0.80389199, 0.46088691]

    def extract_neg(probas):
        return probas[:, 0]

    col = "my_column"
    pred_df = with_inference_column(local_spark_session.createDataFrame(df_test),
                                    sklearn_lr_model,
                                    extract_neg,
                                    col).toPandas()

    for pred, exp in zip(pred_df[col].values, expected_neg):
        assert pred == pytest.approx(exp)


def test_with_inference_column_params(local_spark_session, sklearn_lr_model):
    """Test mandatory paramns error handling
    """
    with pytest.raises(ValueError):
        with_inference_column("not a df", sklearn_lr_model)
    with pytest.raises(ValueError):
        with_inference_column(
            local_spark_session.createDataFrame(pd.DataFrame({'a': 1})), "not a model"
        )
