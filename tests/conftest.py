import os
import sys

import pytest
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def local_spark_session():
    if "SPARK_HOME" in os.environ.keys():
        del os.environ["SPARK_HOME"]
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
    SparkSession.builder._options = {}
    ssb = SparkSession.builder.master("local[1]").config("spark.submit.deployMode", "client")
    ss = ssb.getOrCreate()
    yield ss
    ss.stop()


@pytest.fixture()
def sklearn_lr_model():
    """Returns a simple trained LR model, with test data and expected predictions
    """
    df_train = pd.DataFrame([{'a': 1.5, 'b': 2.5}, {'a': .5, 'b': 4.5}])
    targets = pd.Series([True, False])

    lr = LogisticRegression()
    lr.fit(df_train, targets)

    return lr
