from contextlib import contextmanager
from io import BytesIO
from unittest import mock
import os

import pytest
import pyspark

from ml_hadoop_experiment.tensorflow import vocabulary


@pytest.mark.parametrize("inputs,threshold,num_partitions,expected", [
    ([
        pyspark.Row(feature1=1, feature2=10, feature3=20),
        pyspark.Row(feature1=1, feature2=13, feature3=25),
        pyspark.Row(feature1=3, feature2=10, feature3=20),
        pyspark.Row(feature1=33, feature2=100, feature3=20)
    ], 1, 1, {
        'feature1': [1, 3, 33],
        'feature2': [10, 13, 100],
        'feature3': [20, 25]
    }),
    ([
        pyspark.Row(feature1=1, feature2=10, feature3=None),
        pyspark.Row(feature1=1, feature2=13, feature3=None),
        pyspark.Row(feature1=3, feature2=11, feature3=20),
        pyspark.Row(feature1=33, feature2=100, feature3=20)
    ], 2, 6, {'feature1': [1], 'feature3': [20]}),
    ([
        pyspark.Row(my_feature=[1]),
        pyspark.Row(my_feature=[2, 4]),
        pyspark.Row(my_feature=[10, 100]),
        pyspark.Row(my_feature=[10, 200])
    ], 1, 1, {
        'my_feature': [1, 2, 4, 10, 100, 200]
    })
])
def test_get_vocab_values(
    local_spark_session,
    inputs,
    threshold,
    num_partitions,
    expected
):
    rdd = local_spark_session.sparkContext.parallelize(inputs, num_partitions)
    col_names = list(inputs[0].asDict().keys())
    dictionary = {}
    for col_name in col_names:
        dictionary[col_name] = [col_name]
    vocab_values = vocabulary._get_vocab_values(rdd, dictionary, threshold)

    assert vocab_values.keys() == expected.keys()
    for key, value in expected.items():
        assert set(value) == set(vocab_values[key])


@pytest.mark.parametrize("inputs,threshold,num_partitions,column_mapping,expected", [
    ([
        pyspark.Row(feature1=1, feature2=10, feature3=20),
        pyspark.Row(feature1=1, feature2=13, feature3=25),
        pyspark.Row(feature1=3, feature2=10, feature3=20),
        pyspark.Row(feature1=33, feature2=100, feature3=20)
    ], 1, 1, {
        "my_key": ["feature1", "feature2"],
        "my_key_2": ["feature2", "feature3"]
    }, {
        'my_key': [1, 3, 10, 13, 33, 100],
        'my_key_2': [10, 13, 20, 25, 100]
    })
])
def test_get_vocab_values_merged(
    local_spark_session,
    inputs,
    threshold,
    num_partitions,
    column_mapping,
    expected
):
    rdd = local_spark_session.sparkContext.parallelize(inputs, num_partitions)
    vocab_values = vocabulary._get_vocab_values(rdd, column_mapping, threshold)

    assert vocab_values.keys() == expected.keys()
    for key, value in expected.items():
        assert set(value) == set(vocab_values[key])


@mock.patch('ml_hadoop_experiment.tensorflow.vocabulary.filesystem')
@pytest.mark.parametrize("inputs,path,col_names", [
    (
        {
            'feature1': [1, 3, 33],
            'feature2': [10, 13, 100],
            'feature3': [20, 25]
        },
        '/home/me/my_path',
        ['feature1', 'feature2', 'feature3', 'userid']
    ),
    (
        {
            'country': ['FR', 'US', '']
        },
        '/home/me/my_path',
        ['country']
    )
])
def test_write_vocab_files(fs_provider_mock, inputs, path, col_names):
    ios = dict()

    @contextmanager
    def get_hdfs_open(*args, **kwargs):
        ios[args[0]] = BytesIO()
        io_mock = mock.Mock()
        io_mock.write = ios[args[0]].write
        yield io_mock

    fs_mock = mock.Mock()
    fs_provider_mock.resolve_filesystem_and_path.return_value = (fs_mock, mock.Mock())
    fs_mock.exists.return_value = True
    fs_mock.open = get_hdfs_open
    dictionary = {}
    for col_name in col_names:
        dictionary[col_name] = [col_name]
    voc_files_list = vocabulary._write_vocab_files(inputs, path, dictionary)

    expected_voc_files_list = {
        col_name: os.path.join(path, f'{col_name}.voc') for col_name in col_names
    }
    assert set(voc_files_list) == set(expected_voc_files_list.values()) == set(ios.keys())
    for col_name in col_names:
        assert ios[expected_voc_files_list[col_name]].getvalue() == \
            '\n'.join([str(e) for e in inputs.get(col_name, '') if e != '']).encode()
