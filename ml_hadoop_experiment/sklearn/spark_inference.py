from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyspark.sql as sp
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType


def with_inference_column(
    df: sp.DataFrame,
    model: Any,
    output_column_name: str = "prediction",
    output_column_type: sp.types.DataType = FloatType(),
    postprocessing_fn: Optional[Callable[[Any], pd.Series]] = None,
) -> sp.DataFrame:
    """
    Runs inference on the input dataframe and adds a column 'output_column_name' with
    your postprocessed model outputs.
    Your model/sklearn.Pipeline must have a method predict_proba

    Example usage:
        def predict_proba_extract(model_outputs):
            return pd.Series(model_outputs[:, 1])

        model = mlflow.pyfunc.load_model(f'{run.info.artifact_uri}/{model_path}')
        prediction_df = with_inference_column(test_df, model, "prob", predict_proba_extract)

    :param df: the dataframe to add a prediction to
    :param model: model to use for inference
    :param output_column_name: name of the newly-created inference column
    :param output_column_type: type of the newly-created inference column
    :param postprocessing_fn: postprocessing function called on your model outputs
    The primary purpose of this functon is to extract the relevant scores/predictions of
    your model outputs but it is not limited to this use case.
    """
    if df is None or not isinstance(df, sp.DataFrame):
        raise ValueError("Missing or invalid dataframe.")
    if model is None or getattr(model, "predict_proba", None) is None:
        raise ValueError("Missing or invalid model.")

    columns = df.columns

    def inference(*data: pd.Series) -> pd.Series:
        pdf = pd.DataFrame({c: d for c, d in zip(columns, data)})
        result = model.predict_proba(pdf)
        if postprocessing_fn is not None:
            result = postprocessing_fn(result)
        if not isinstance(result, pd.Series):
            result = pd.Series(data=np.ascontiguousarray(result))
        return result

    udf = sf.pandas_udf(inference, output_column_type)(*columns)  # type: ignore

    return df.withColumn(output_column_name, udf)
