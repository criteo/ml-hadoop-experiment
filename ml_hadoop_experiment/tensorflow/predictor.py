from types import TracebackType
from typing import List, Dict, Union, Callable, Optional, Iterator, Type

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np


feeds_type = Optional[List[str]]
fetches_type = Optional[List[str]]


def import_graph_def(grap_def_path: str) -> List[Union[tf.Operation, tf.Tensor]]:
    with gfile.FastGFile(grap_def_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, name="")


def get_node_by_name(graph: tf.Graph, name: str) -> Optional[Union[tf.Operation, tf.Tensor]]:
    for node in graph.as_graph_def().node:
        if node.name == name:
            return graph.as_graph_element(node.name)
    return None


def get_tensors(graph: tf.Graph, names: feeds_type) -> Dict[str, tf.Tensor]:
    tensors = dict()
    if names is not None:
        for name in names:
            op_or_tensor = graph.as_graph_element(name)
            if isinstance(op_or_tensor, tf.Tensor):
                tensors[name] = op_or_tensor
            else:
                if len(op_or_tensor.outputs) > 1:
                    raise ValueError(f"Found more than one tensor for operation {op_or_tensor}")
                tensors[name] = op_or_tensor.outputs[0]
    return tensors


def get_feedable_tensors(graph: tf.Graph, names: feeds_type) -> Dict[str, tf.Tensor]:
    feedable_tensors = get_tensors(graph, names)
    for name, tensor in feedable_tensors.items():
        if not graph.is_feedable(tensor):
            raise ValueError(f"{name} should be feedable but is not")
    return feedable_tensors


def get_fetchable_tensors(graph: tf.Graph, names: fetches_type) -> Dict[str, tf.Tensor]:
    fetchable_tensors = get_tensors(graph, names)
    for name, tensor in fetchable_tensors.items():
        if not graph.is_fetchable(tensor):
            raise ValueError(f"{name} should be fetchable but is not")
    return fetchable_tensors


class Predictor:
    def __init__(self, session: tf.compat.v1.Session, feeds: feeds_type, fetches: fetches_type):
        self.session = session
        self.feed_tensors = get_feedable_tensors(self.session.graph, feeds)
        self.fetch_tensors = get_fetchable_tensors(self.session.graph, fetches)

    @staticmethod
    def from_graph(path: str, feeds: feeds_type, fetches: fetches_type) -> "Predictor":
        session = tf.compat.v1.Session(graph=tf.Graph())
        with session.graph.as_default():
            import_graph_def(path)

            # Run Table initializer if present in the graph
            init_all_tables = get_node_by_name(session.graph, "init_all_tables")
            if init_all_tables:
                session.run(init_all_tables)
        return Predictor(session, feeds, fetches)

    def __enter__(self) -> "Predictor":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.session.close()

    def predict(
        self, inputs: Union[Dict[str, np.array], Callable[[], tf.data.Dataset]]
    ) -> Union[Iterator, Dict[str, tf.Tensor]]:
        if isinstance(inputs, dict):
            if not set(self.feed_tensors) <= set(inputs):
                raise KeyError(
                    "Missing keys in inputs: "
                    f"{set(self.feed_tensors) - set(inputs)} (inputs = {inputs})"
                )
            return self.session.run(
                self.fetch_tensors,
                feed_dict={tensor: inputs[name] for name, tensor in self.feed_tensors.items()}
            )
        elif callable(inputs):

            def _input_gen() -> Iterator[Dict[str, tf.Tensor]]:
                with self.session.graph.as_default():
                    dataset = inputs()  # type: ignore
                    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
                    next_element = iterator.get_next()
                    self.session.run(tf.compat.v1.tables_initializer())
                    try:
                        while True:
                            input_dict = self.session.run(next_element)
                            output_dict = self.predict(input_dict)
                            yield {**input_dict, **output_dict}
                    except tf.errors.OutOfRangeError:
                        pass

            return _input_gen()
        else:
            raise TypeError(f"Expected type dict or tf.data.Dataset but got {type(inputs)}")
