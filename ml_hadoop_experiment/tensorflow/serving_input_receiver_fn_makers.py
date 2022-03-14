from typing import (
    Dict,
    Union,
    Tuple,
    Callable,
    Any
)

import tensorflow as tf

from ml_hadoop_experiment.tensorflow.tfrecords import transfo_fn_type, features_specs_type


def featurespec_to_input_placeholders(
    feature_spec: features_specs_type,
    batched_predictions: bool = True
) -> Tuple[Dict[str, tf.Tensor], Dict[str, Union[tf.Tensor, tf.SparseTensor]]]:
    """
    From a featurespec, generate all the necessary placeholder tensors that will be needed
    for inference.
    We generate two dictionaries:
    - raw_tensors is the dictionary of tensors that the client will have to provide
    - prediction_input_tensors is the dictionary of tensors that tensorflow will receive as input
      for the training/inference
    """
    def make_placeholder_name(key: str) -> str:
        return key

    tf.compat.v1.disable_eager_execution()
    raw_tensors = {}
    prediction_input_tensors = {}
    for k, v in feature_spec.items():
        if "/" in k:
            raise ValueError(f"'/' character is not allowed in dimension name ({k})")
        if isinstance(v, tf.io.FixedLenFeature):
            if len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1):
                dense_placeholder = tf.compat.v1.placeholder(
                    v.dtype,
                    shape=([None] + v.shape) if batched_predictions else v.shape,
                    name=make_placeholder_name(k))
                raw_tensors[k] = dense_placeholder
                prediction_input_tensors[k] = dense_placeholder
            else:
                raise NotImplementedError(f"spec for FixedLenFeature of non-scalar shape not"
                                          f"supported (got {v.shape}")
        elif isinstance(v, tf.io.VarLenFeature):
            # Can't use sparse_placeholder, cf https://github.com/tensorflow/tensorflow/issues/22396
            # I'm assuming this VarLenFeature is used to hold lists, so the rank of the
            # sparse tensor is just 2.
            shape_key = k + "/shape"
            indices_key = k + "/indices"
            values_key = k + "/values"

            shape_placeholder = tf.compat.v1.placeholder(
                tf.int64,
                shape=[2],
                name=make_placeholder_name(shape_key)
            )
            indices_placeholder = tf.compat.v1.placeholder(
                tf.int64,
                shape=[None, 2],
                name=make_placeholder_name(indices_key)
            )
            values_placeholder = tf.compat.v1.placeholder(
                v.dtype,
                shape=[None],
                name=make_placeholder_name(values_key)
            )

            raw_tensors[shape_key] = shape_placeholder
            raw_tensors[indices_key] = indices_placeholder
            raw_tensors[values_key] = values_placeholder

            prediction_input_tensors[k] = tf.SparseTensor(indices=indices_placeholder,
                                                          values=values_placeholder,
                                                          dense_shape=shape_placeholder)
        else:
            raise NotImplementedError(f"Unknown feature spec type {v}")

    return raw_tensors, prediction_input_tensors


def make_raw_serving_input_receiver_fn(
    feature_spec: features_specs_type,
    transform_input_tensor: Callable[[Dict[str, tf.Tensor]], None],
    is_model_canned_estimator: bool = False,
    batched_predictions: bool = True
) -> Callable[[], tf.estimator.export.ServingInputReceiver]:
    """
    Build the serving_input_receiver_fn used for serving/inference.
    transform_input_tensor: method that takes the input tensors and will mutate them so prediction
    will have its correct input. For instance, it could be to generate feature transfo from
    "raw dimensions" tensors.
    is_model_canned_estimator: if the model you want to serve is a canned estimator, the serving
    function has to be generated differently
    """

    def serving_input_receiver_fn() -> Any:
        # generate all tensor placeholders:
        raw_tensors, prediction_input_tensors = featurespec_to_input_placeholders(
            feature_spec, batched_predictions)

        # Add transformations (for instance, feature transfos) to prediction_input_tensors
        transform_input_tensor(prediction_input_tensors)

        if is_model_canned_estimator:
            return tf.estimator.export.ServingInputReceiver(
                features=prediction_input_tensors, receiver_tensors={},
                receiver_tensors_alternatives={"raw_input": raw_tensors})
        else:
            return tf.estimator.export.ServingInputReceiver(
                features=prediction_input_tensors, receiver_tensors=raw_tensors)

    return serving_input_receiver_fn


def make_default_serving_input_receiver_fn(
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
