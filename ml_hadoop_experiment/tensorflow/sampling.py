from typing import (
    List,
    Tuple
)

import pyspark
import pyspark.sql.functions as sf

from ml_hadoop_experiment.tensorflow.constant import WEIGHT_COLUMN_NAME


# randomized sampling, non-deterministic
def add_random_sampling_col(df: pyspark.sql.DataFrame) -> Tuple[str, pyspark.sql.DataFrame]:
    return "sampling_rand", df.withColumn("sampling_rand", pyspark.sql.functions.rand())


# deterministic sampling based on hash of columns_for_sample
def add_deterministic_sampling_col(df: pyspark.sql.DataFrame, columns_for_sample: List[str]
                                   ) -> Tuple[str, pyspark.sql.DataFrame]:
    # hash() method returns a scala.Int so it is uniformly distributed in [-2**31; 2**31-1]
    df = df.withColumn("hash", sf.hash(
        *[df[x] for x in columns_for_sample]))
    # "sampling_hash" is uniformly distributed in [0; 1[
    return "sampling_hash", df.withColumn("sampling_hash", 0.5 + df.hash.cast(
        pyspark.sql.types.DoubleType()) / (sf.lit(float(2 ** 32))))


# simplify filters if they are trivial
def get_filter_sampling_ratio(column: pyspark.sql.column.Column, sampling_ratio: float
                              ) -> pyspark.sql.column.Column:
    if sampling_ratio <= 0.0:
        return sf.lit(False)
    elif sampling_ratio >= 1.0:
        return sf.lit(True)
    else:
        # we actually really return a column even though the IDE may think it's a bool
        return column < sampling_ratio


def sample_with_predicate(df: pyspark.sql.DataFrame,
                          global_sampling: float,
                          positive_sampling: float,
                          negative_sampling: float,
                          positive_predicate: pyspark.sql.Column,
                          columns_for_sample: List[str] = []) -> pyspark.sql.DataFrame:
    """
     Sample the dataframe in a single pass according to various criteria.
     global_sampling: ratio over which the full dataset should be sampled
     positive_sampling: ratio over which the positive examples should be sampled
     negative_sampling: ratio over which the negative examples should be sampled
     positive_predicate: a pyspark.sql.Column of Boolean type that defines which examples are
     positives.
     columns_for_sample: list of columns that define the source of entropy for the random sampling.
     If this is not defined, then the random sampling will be random but non-deterministic.
     If this is defined, then the random sampling will be deterministic, and based on the hash of
     all the provided columns. This means that rows having the same values for the columns in
     columns_for_sample will be sampled together.
     Thus if you define this you should define enough columns so that their values provide a good
     source of entropy (like the combination of timestamp, uid, etc).

     This method will return a new dataframe correctly sampled and will add a "weight" column
     that will have its values adjusted to take into account the re-sampling.
    """

    if len(columns_for_sample) == 0:
        sampling_col, df = add_random_sampling_col(df)
    else:
        sampling_col, df = add_deterministic_sampling_col(
            df, columns_for_sample)

    global_pos_sampling = global_sampling * positive_sampling
    global_neg_sampling = global_sampling * negative_sampling
    max_sampling = max(global_pos_sampling, global_neg_sampling)

    pos_filter = get_filter_sampling_ratio(
        df[sampling_col], global_pos_sampling)
    neg_filter = get_filter_sampling_ratio(
        df[sampling_col], global_neg_sampling)

    # Like in prediction code, we assume predicate is false if underlying
    # columns are missing for this row
    pos_pred = positive_predicate & positive_predicate.isNotNull()
    df = df.filter((pos_pred & pos_filter) | (~pos_pred & neg_filter))

    # adjust weight column with sampling information:
    if WEIGHT_COLUMN_NAME not in df.columns:
        weight_col = sf.lit(1.0)
    else:
        weight_col = df[WEIGHT_COLUMN_NAME]

    # We apply relative weighting only. That is, if positive sampling is 0.2 and negative
    # sampling is 0.1, then the weights are as if the samplings were 1.0 and 0.5, ie the weights
    # will be 1.0 and 2.0.

    df = df.withColumn(WEIGHT_COLUMN_NAME,
                       pyspark.sql.functions.when(positive_predicate,
                                                  weight_col * max_sampling / global_pos_sampling).
                       otherwise(weight_col * max_sampling / global_neg_sampling))

    return df
