import getpass
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import protobuf_examples as data
from ml_hadoop_experiment.tensorflow import tfrecords as tffr


user = getpass.getuser()


@pytest.mark.parametrize("feature_mappings,specs", [
    (data.feature_mappings, data.features_specs)  
])
def test_to_tf_proto(feature_mappings, specs, tmpdir):
    tf_proto = tffr.to_tf_proto(data.mappings_sample(feature_mappings), specs)
    expected = data.mappings_protobuf(feature_mappings)
    assert tf_proto == expected, \
        "{0} should be equal to {1}".format(tf_proto, expected)

    # write tfRecords from proto
    records_path = f'{tmpdir.realpath()}/simulated.tfrecord'
    with tf.io.TFRecordWriter(records_path) as writer:
        writer.write(tf_proto.SerializeToString())

    # test reading generated TFRecords with the same spec
    dataset = tf.data.TFRecordDataset([records_path]).batch(1)
    dataset = dataset.map(lambda x: tf.io.parse_example(serialized=x, features=specs))
    it = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        read_sample = sess.run(it)
        print(f"read_sample: {read_sample}")
        expected_reading = data.mappings_tfreading(feature_mappings)
        for feature in read_sample:
            read_value = read_sample[feature]
            if isinstance(read_value, tf.compat.v1.SparseTensorValue):
                read_value = read_value.values
            expected_value = expected_reading[feature]
            spec = specs[feature]
            if spec.dtype == tf.string:
                np.testing.assert_array_equal(read_value, expected_value)
            else:
                np.testing.assert_array_almost_equal(read_value, expected_value)


@pytest.mark.parametrize("feature_mappings,specs", [
    (data.feature_mappings_with_inconsistent_size, data.features_specs_with_defaults)
])
def test_to_tf_proto_inconsistent_content(feature_mappings, specs):
    with pytest.raises(ValueError):
        tffr.to_tf_proto(data.mappings_sample(feature_mappings), specs)


@pytest.mark.parametrize("tfrecords,export_path,index", [
    ([data.mappings_protobuf(data.feature_mappings)], f"viewfs://root/{user}/tf_record_test", 1),
    ([data.mappings_protobuf(m) for m in data.all_mappings],
     f"viewfs://root/{user}/tf_record_test", 2)
])
def test_write_example_partition(tfrecords, export_path, index):
    tfwriter_mock = mock.Mock()

    @contextmanager
    def get_tfrecordwriter(*args, **kwargs):
        yield tfwriter_mock

    with mock.patch('ml_hadoop_experiment.tensorflow.tfrecords.tf.io') as tfpython_io_mock:
        tfpython_io_mock.TFRecordWriter = get_tfrecordwriter
        results = tffr.write_example_partition(tfrecords, index, export_path)
        expected_call_args_list =\
            [((tfrecord.SerializeToString(),),) for tfrecord in tfrecords]
        assert tfwriter_mock.write.call_args_list == expected_call_args_list
        assert results[0][0] ==\
            "{0}/part-{1:05d}".format(export_path, index)
