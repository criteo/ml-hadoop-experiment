from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

# feature_key: (sample from glup, value in protobuf, value read from tf_record)
# for VarLenFeature when reading we inspect SparseTensorValue.values that is of shape (n,)
# In contrast to FixedLenFeature that is of shape (1, n) when read
feature_mappings = {
    'feature1': ([0.01], tf.train.Feature(float_list=tf.train.FloatList(value=[0.01])), np.array([[0.01]])),
    'feature2': ([1], tf.train.Feature(int64_list=tf.train.Int64List(value=[1])), np.array([[1]])),
    'feature3': (["value"], tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"value"])), np.array([[b"value"]])),
    'feature4': ([0, 1], tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 1])), np.array([[0, 1]])),
    'feature5': ([1, 1, 2], tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 1, 2])), np.array([1, 1, 2])),
}

feature_mappings_without_lists = {
    'feature1': (0.01, tf.train.Feature(float_list=tf.train.FloatList(value=[0.01])), np.array([[0.01]])),
    'feature2': (1, tf.train.Feature(int64_list=tf.train.Int64List(value=[1])), np.array([[1]])),
    'feature3': ("value", tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"value"])), np.array([[b"value"]])),
    'feature4': ([0, 1], tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 1])), np.array([[0, 1]])),
    'feature5': ([0, 1], tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 1])), np.array([0, 1])),
}

feature_mappings_null_without_defaults = {
    'feature1': (None, tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])), np.array([[0.0]])),
    'feature2': (None, tf.train.Feature(int64_list=tf.train.Int64List(value=[0])), np.array([[0]])),
    'feature3': (None, tf.train.Feature(bytes_list=tf.train.BytesList(value=[b""])), np.array([[b""]])),
    'feature4': (None, tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 0])), np.array([[0, 0]])),
    'feature5': (None, None, np.array([])),
}

feature_mappings_empty_without_defaults = {  # type: ignore
    'feature1': ([], tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])), np.array([[0.0]])),
    'feature2': ([], tf.train.Feature(int64_list=tf.train.Int64List(value=[0])), np.array([[0]])),
    'feature3': ([], tf.train.Feature(bytes_list=tf.train.BytesList(value=[b""])), np.array([[b""]])),
    'feature4': ([], tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 0])), np.array([[0, 0]])),
    'feature5': ([], tf.train.Feature(int64_list=tf.train.Int64List(value=[])), np.array([])),
}

feature_mappings_null_with_defaults = {
    'feature1': (None, None, np.array([[1.0]])),
    'feature2': (None, None, np.array([[1]])),
    'feature3': (None, None, np.array([[b"a"]])),
    'feature4': (None, None, np.array([[1, 1]])),
    'feature5': (None, None, np.array([])),
}

feature_mappings_with_inconsistent_size = {'feature4': ([1, 2, 3], None, None)}

all_mappings = [
    feature_mappings,
    feature_mappings_without_lists,
    feature_mappings_null_without_defaults,
    feature_mappings_empty_without_defaults,
    feature_mappings_null_with_defaults,
]


def mappings_sample(mappings):
    return {name: values[0] for name, values in mappings.items()}


def mappings_protobuf(mappings):
    return tf.train.Example(
        features=tf.train.Features(
            feature={name: values[1] for name, values in mappings.items() if values[1] is not None}
        )
    )


def mappings_tfreading(mappings):
    return {name: values[2] for name, values in mappings.items()}


features_specs = {
    'feature1': tf.io.FixedLenFeature([1], tf.float32),
    'feature2': tf.io.FixedLenFeature([1], tf.int64),
    'feature3': tf.io.FixedLenFeature([1], tf.string),
    'feature4': tf.io.FixedLenFeature([2], tf.int64),
    'feature5': tf.io.VarLenFeature(tf.int64),
}

features_specs_with_defaults = {
    'feature1': tf.io.FixedLenFeature([1], tf.float32, default_value=1.0),
    'feature2': tf.io.FixedLenFeature([1], tf.int64, default_value=1),
    'feature3': tf.io.FixedLenFeature([1], tf.string, default_value=b'a'),
    'feature4': tf.io.FixedLenFeature([2], tf.int64, default_value=[1, 1]),
    'feature5': tf.io.VarLenFeature(tf.int64),
}
