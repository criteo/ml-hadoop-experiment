import tensorflow as tf

from ml_hadoop_experiment.tensorflow import serving_input_receiver_fn_makers


def test_featurespec_to_input_placeholders():

    feature_spec = {
        "dim1": tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
        "dim2": tf.io.VarLenFeature(dtype=tf.string),
    }

    raw_tensors, prediction_input_tensors = (
        serving_input_receiver_fn_makers.featurespec_to_input_placeholders(feature_spec)
    )

    assert len(raw_tensors) == 4
    assert "dim1" in raw_tensors
    assert "dim2/shape" in raw_tensors
    assert "dim2/indices" in raw_tensors
    assert "dim2/values" in raw_tensors

    assert len(prediction_input_tensors) == 2
    assert "dim1" in prediction_input_tensors
    assert "dim2" in prediction_input_tensors

    # check that we actually have the same tensor references in both structures
    assert prediction_input_tensors["dim1"] is raw_tensors["dim1"]
    assert prediction_input_tensors["dim2"].indices is raw_tensors["dim2/indices"]
    assert prediction_input_tensors["dim2"].values is raw_tensors["dim2/values"]
    assert prediction_input_tensors["dim2"].dense_shape is raw_tensors["dim2/shape"]
