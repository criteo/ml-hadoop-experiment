import os
import sys

import pytest
from pyspark.sql import SparkSession


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
