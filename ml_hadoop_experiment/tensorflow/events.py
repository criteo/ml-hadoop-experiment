import typing
import os

import pandas
import tensorflow as tf
from cluster_pack import filesystem

from ml_hadoop_experiment.common.paths import check_full_hdfs_path


def test_is_event_file(filename: str) -> bool:
    return os.path.basename(filename).startswith('events.out')


def gen_events_iterator(model_path: str) -> typing.Iterator:
    if not check_full_hdfs_path(model_path):
        raise ValueError(f"{model_path} is not a full hdfs path")
    fs, _ = filesystem.resolve_filesystem_and_path(model_path)
    event_file = next((filename for filename in fs.ls(model_path)
                       if test_is_event_file(filename)))
    assert isinstance(event_file, str)
    return tf.compat.v1.train.summary_iterator(event_file)


def get_all_metrics(model_path: str) -> pandas.DataFrame:
    events = gen_events_iterator(model_path)
    dataframe: typing.Dict = {
        'step': list(),
        'name': list(),
        'value': list()
    }
    for event in events:
        summary = event.summary
        if summary:
            for value in summary.value:
                if value.simple_value:
                    dataframe['step'].append(event.step)
                    dataframe['name'].append(value.tag)
                    dataframe['value'].append(value.simple_value)
    return pandas.DataFrame(data=dataframe)
