import os
import logging
from collections.abc import Sized
from typing import (
    Dict,
    Union,
    Iterator,
    List,
    Tuple,
    Callable,
    Iterable,
    Optional,
    cast,
    Generator
)

import deprecation
import tensorflow as tf
from pyspark.sql.functions import rand
import pyspark
from cluster_pack import filesystem

from ml_hadoop_experiment.tensorflow import vocabulary, dataframe_prediction_helper
from ml_hadoop_experiment.common.paths import check_full_hdfs_path

_logger = logging.getLogger(__name__)


TF_RECORD_DIR = "tf_records"
COL_CARDINALITIES_DIR = "col_cardinalities"


features_specs_type = Dict[
    str,
    Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
]


transfo_fn_type = Callable[
    [Dict[str, tf.Tensor]],
    Dict[str, tf.Tensor]
]


@deprecation.deprecated(deprecated_in="0.2.0",
                        details="Use serving_input_receiver_fn_makers\
                        .make_default_serving_input_receiver_fn instead")
def serving_input_receiver_fn_factory(
    features_specs: features_specs_type,
    feature_transfo_fn: transfo_fn_type = None,
    input_name: str = 'inputs'
) -> Callable[[], tf.estimator.export.ServingInputReceiver]:
    def serving_input_receiver_fn(
    ) -> tf.estimator.export.ServingInputReceiver:
        serialized_tfr_example = tf.compat.v1.placeholder(
            dtype=tf.string,
            shape=[None],
            name=input_name
        )
        parsed_features = tf.io.parse_example(
            serialized=serialized_tfr_example, features=features_specs
        )
        if feature_transfo_fn:
            parsed_features = feature_transfo_fn(parsed_features)
        return tf.estimator.export.ServingInputReceiver(
            parsed_features,
            # Passing a dict is required to override the default single value: `input`
            {input_name: serialized_tfr_example}
        )

    return serving_input_receiver_fn


def read_parsed_tfr(
    files: Iterable[str],
    features_specs: features_specs_type,
    compression_type: str = "GZIP"
) -> Iterator[Dict]:
    dataset = tf.data.TFRecordDataset(files, compression_type=compression_type)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(serialized=x, features=features_specs)
    )
    return run_with_one_shot_iterator(dataset)


def read_parsed_sequence_tfr(
    files: Iterable[str],
    context_features: features_specs_type,
    sequence_features: features_specs_type,
    compression_type: str = "GZIP"
) -> Iterator[Dict]:
    dataset = tf.data.TFRecordDataset(files, compression_type=compression_type)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_sequence_example(
            x,
            context_features=context_features,
            sequence_features=sequence_features)
    )
    return run_with_one_shot_iterator(dataset)


def run_with_one_shot_iterator(dataset: tf.data.Dataset) -> Generator:
    next_tfr = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                yield sess.run(next_tfr)
        except tf.errors.OutOfRangeError:
            _logger.info("end of dataset")
            pass


def run_with_initializable_iterator(dataset: tf.data.Dataset) -> Generator:
    iterator = dataset.make_initializable_iterator()
    next_value = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(iterator.initializer)
        try:
            while True:
                yield sess.run(next_value)
        except tf.errors.OutOfRangeError:
            _logger.info("end of dataset")
            pass


features_primitive = Union[int, float, str, bytes]
features_type = Optional[Union[List[features_primitive], features_primitive]]


def _as_list(value: features_type) -> Optional[List[features_primitive]]:
    return [value] if value is not None and not isinstance(value, List) else value


def _int64_feature(value: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=value
    ))


def _float_feature(value: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=value
    ))


def _string_feature(value: List[Union[str, bytes]]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.encode() if isinstance(x, str) else x for x in value]))


def _get_feature_default_value(
    spec: tf.io.FixedLenFeature
) -> List[features_primitive]:
    value: features_primitive
    if spec.dtype.is_integer:
        value = 0
    elif spec.dtype.is_floating:
        value = 0.0
    elif spec.dtype == tf.string:
        value = b''
    else:
        raise ValueError(f'No default value for type {spec.dtype}')
    return [value] * spec.shape[0]


def _preprocess_feature_value(
    value: features_type,
    spec: Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
) -> Optional[Union[List[Union[int, float, str, bytes]]]]:
    try:
        # tf.io.FixedLenFeature has attribute default_value, tf.io.VarLenFeature hasn't
        # we can not check FixedLenFeature with isinstance due to pickling issues
        if hasattr(spec, 'default_value'):
            # Interpret an empty list as None
            if isinstance(value, Sized) and not isinstance(value, str) and\
                    not isinstance(value, bytes) and len(value) == 0:
                value = None

            if value is None:
                if spec.default_value is not None:
                    value = None
                else:
                    value = _get_feature_default_value(spec)

        return _as_list(value)
    except TypeError as ex:
        raise ValueError(f"{type(value)} {str(value)} {ex}")


def _value_to_feature(
    value: List[features_primitive],
    spec: Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
) -> tf.train.Feature:
    if spec.dtype.is_integer:
        for val in value:
            if not isinstance(val, int):
                raise ValueError(f"{str(val)} in {value} is not integer as required by {spec}")
        return _int64_feature(cast(List[int], value))
    elif spec.dtype.is_floating:
        for val in value:
            if not isinstance(val, int) and not isinstance(val, float):
                raise ValueError(f"{str(val)} in {value} is not a number as required by {spec}")
        return _float_feature(cast(List[float], value))
    elif spec.dtype == tf.string:
        for val in value:
            if not isinstance(val, str) and not isinstance(val, bytes):
                raise ValueError(f"{str(val)} in {value} is not str or bytes as required by {spec}")
        return _string_feature(cast(List[Union[str, bytes]], value))
    else:
        raise ValueError(f'Type {type(value)} of spec {spec} is not supported')


def to_tf_proto(
    x: Dict[str, features_type],
    features_specs: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]]
) -> tf.train.Example:
    """ Transform values into tensorflow Examples for TFRecords
    If default values are included in the feature spec then we keep null entries for the
    Tfrecord, assuming the same spec will be used to read the TFrecords. If no default value
    is included then we add one, otherwise the incomplete feature spec will create a failure
    during reading time. This function only covers FixedLenFeature and VarLenFeature.
    """
    feature = {}
    for name, spec in features_specs.items():
        # Some components of x have value None
        value = _preprocess_feature_value(x.get(name), spec)
        # We do not store None values in the feature dictionary
        if value is None:
            continue

        if hasattr(spec, 'shape') and len(value) != spec.shape[0]:
            raise ValueError(
                f"value {value} does not correspond to expected shape in spec {spec}")
        feature[name] = _value_to_feature(value, spec)

    features = tf.train.Features(feature=feature)
    return tf.train.Example(features=features)


def write_example_partition(
    tfrecords: Iterable[tf.train.Example],
    index: int,
    export_path: str,
    compression_type: tf.compat.v1.io.TFRecordCompressionType =
    tf.compat.v1.io.TFRecordCompressionType.GZIP
) -> List[Tuple[str, int]]:
    remote_path = "{0}/part-{1:05d}".format(export_path, index)
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    num_tfr_records = 0
    with tf.io.TFRecordWriter(remote_path, options=options) \
            as writer:
        for tfr in tfrecords:
            writer.write(tfr.SerializeToString())
            num_tfr_records += 1
    return [(remote_path, num_tfr_records)]


def write_example_rdd(
    tfrecords: pyspark.RDD,
    export_path: str,
    compression_type: tf.compat.v1.io.TFRecordCompressionType =
    tf.compat.v1.io.TFRecordCompressionType.GZIP
) -> List[Tuple[str, int]]:
    """ Save a list of TF records on HDFS"""
    if not check_full_hdfs_path(export_path):
        raise ValueError(f"{export_path} is not a full hdfs path")
    return tfrecords.mapPartitionsWithIndex(
        lambda idx, tfrecords:
            write_example_partition(tfrecords, idx, export_path, compression_type)).collect()


def df_to_tf_record(
    df: pyspark.sql.DataFrame,
    features_specs: features_specs_type,
    base_dir: str,
    vocab_columns: List[str] = None,
    threshold: int = 0
) -> List[str]:
    tf_record_dir = f"{base_dir}/{TF_RECORD_DIR}"

    if vocab_columns is not None:
        col_cardinalities_dir = f"{base_dir}/{COL_CARDINALITIES_DIR}"
        _logger.info("writing vocab files ..")
        vocab_files = vocabulary.gen_vocab_files_from_list(
            vocab_columns,
            df.select(vocab_columns).rdd,
            col_cardinalities_dir,
            threshold=threshold
        )
        _logger.info(f"vocab files: {vocab_files}")

    _df = df.select(dataframe_prediction_helper.filtered_columns(df, features_specs)).\
        orderBy(rand()).\
        persist(pyspark.StorageLevel.DISK_ONLY)

    _logger.info("writing tf_record files ..")
    _df.write.\
        format("tfrecords").\
        option("codec", "org.apache.hadoop.io.compress.GzipCodec").\
        save(tf_record_dir)

    fs, _ = filesystem.resolve_filesystem_and_path(tf_record_dir)
    files = [f for f in fs.ls(tf_record_dir) if not os.path.basename(f).startswith("_")]
    _logger.info(f"Generated tf records in {len(files)} files")
    return files
