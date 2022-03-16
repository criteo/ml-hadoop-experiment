from typing import Any, Optional, Callable

import numpy as np
import pandas as pd
import pyspark.sql as sp
from pyspark.sql.types import FloatType
from pyspark.sql import functions as sf


def with_inference_column(df: sp.DataFrame,
                          model: Any,
                          extract_fn: Optional[Callable[[Any], pd.Series]] = None,
                          column_name: str = "prediction",
                          ) -> sp.DataFrame:
    """Return a new dataframe with an added column of inference results on the model

    Example usage:
        def predict_proba_extract(result):
[            return pd.Series(result[:, 1])

        model = mlflow.pyfunc.load_model(f'{run.info.artifact_uri}/{model_path}')
        prediction_df = with_inference_column(test_df, model, predict_proba_extract, "prob")

    :param df: dataframe on which to add results columns
    :param model: model instance
    :param extract_fn: optional method to extract results from inference output
    :param column_name: name of new column
    """
    if df is None or not isinstance(df, sp.DataFrame):
        raise ValueError("Missing or invalid dataframe.")
    if model is None or getattr(model, "predict_proba", None) is None:
        raise ValueError("Missing or invalid model.")

    columns = df.columns

    def inference(*data: pd.Series) -> pd.Series:
        """Called by executors, recreates a Pandas DataFrame from a batch of data split
        into column and passes it to the model to return a column of predictions.

        :param data: list of columns corresponding to a batch of rows
        """
        pdf = pd.DataFrame({c: d for c, d in zip(columns, data)})

        # TODO: this is an example sklearn.Pipeline call,
        # implement 'universal' solution for model flavors
        result = model.predict_proba(pdf)

        if extract_fn is not None:
            result = extract_fn(result)

        if not isinstance(result, pd.Series):
            result = pd.Series(data=np.ascontiguousarray(result))

        return result

    udf = sf.pandas_udf(inference, FloatType())(*columns)

    return df.withColumn(column_name, udf)
