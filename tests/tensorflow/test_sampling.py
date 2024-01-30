from unittest import mock

import pandas as pd
import pyspark.sql.functions as sf
import pytest
from pytest import approx

from ml_hadoop_experiment.tensorflow.sampling import sample_with_predicate


@pytest.mark.parametrize(
    "global_sampling,pos_sampling,neg_sampling,count_result,label_is_null",
    [
        (1.0, 1.0, 1.0, 6, False),
        (0.0, 1.0, 1.0, 0, False),
        (1.0, 0.0, 1.0, 2, False),
        (1.0, 1.0, 0.0, 4, False),
        (1.0, 0.5, 1.0, 4, False),
        (1.0, 0.5, 0.5, 2, False),
        (1.0, 0.3, 0.3, 1, False),
        (1.0, 1.0, 1.0, 6, True),
        (0.0, 1.0, 1.0, 0, True),
        (1.0, 0.0, 1.0, 2, True),
        (1.0, 1.0, 0.0, 4, True),
        (1.0, 0.5, 1.0, 4, True),
        (1.0, 0.5, 0.5, 2, True),
        (1.0, 0.3, 0.3, 1, True),
    ],
)
def test_sampling(local_spark_session, global_sampling, pos_sampling, neg_sampling, count_result, label_is_null):

    def add_deterministic_sampling_col_mock(df, _):
        # mock: don't add sampling_hash column
        # so the sampling algo will directly refer to column "sampling_hash" in the DataFrame
        return "sampling_hash", df

    with mock.patch("ml_hadoop_experiment.tensorflow.sampling.add_deterministic_sampling_col") as mock_sampling_col:
        mock_sampling_col.side_effect = add_deterministic_sampling_col_mock

        ldf = pd.DataFrame.from_records(
            [
                (0.2, 13, 3, 1),
                (0.4, 13, 3, 1),
                (0.6, 2, 1, 0),
                (0.8, 6, 7, 0),
                (0.85, 33, 3, 1),
                (0.9, 5, 9, 1),
            ],
            columns=["sampling_hash", "feature1", "feature2", "label"],
        )

        df = local_spark_session.createDataFrame(ldf)

        if label_is_null:
            df = df.withColumn("label", sf.expr("IF(label == 0, null, label)"))

        df_result = sample_with_predicate(
            df,
            global_sampling,
            pos_sampling,
            neg_sampling,
            df["label"] > 0,
            ["feature1", "feature2"],
        )

        assert len(df_result.collect()) == count_result


def test_sampling_weight_column(local_spark_session):

    def add_deterministic_sampling_col_mock(df, _):
        # mock: don't add sampling_hash column
        # so the sampling algo will directly refer to column "sampling_hash" in the DataFrame
        return "sampling_hash", df

    with mock.patch("ml_hadoop_experiment.tensorflow.sampling.add_deterministic_sampling_col") as mock_sampling_col:
        mock_sampling_col.side_effect = add_deterministic_sampling_col_mock

        ldf = pd.DataFrame.from_records(
            [
                # these lines will be sampled out:
                (0.9, 13, 3, 1, 10.0),
                (0.9, 6, 7, 0, 10.0),
                (0.9, 33, 3, 1, 10.0),
                (0.9, 5, 9, 1, 10.0),
                # these lines will be kept:
                (0.1, 13, 3, 1, 10.0),
                (0.1, 13, 30, 1, 20.0),
                (0.1, 2, 1, 0, 10.0),
                (0.1, 2, 10, 0, 20.0),
            ],
            columns=["sampling_hash", "feature1", "feature2", "label", "weight"],
        )

        df = local_spark_session.createDataFrame(ldf)

        df_result = sample_with_predicate(df, 0.9, 0.5, 0.9, df["label"] > 0, ["feature1", "feature2"])

        # total positive sampling is 0.9*0.5 = 0.45
        # total negative sampling is 0.9*0.9 = 0.81
        # relatively we are sampling the positives more than the negatives at a rate of
        # 0.9/0.5 = 1.8

        rows = df_result.collect()

        assert rows[0]["weight"] == approx(10.0 * 0.9 / 0.5)
        assert rows[1]["weight"] == approx(20.0 * 0.9 / 0.5)
        assert rows[2]["weight"] == approx(10.0)
        assert rows[3]["weight"] == approx(20.0)

        assert len(rows) == 4
