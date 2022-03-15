import typing
from unittest import mock

import pandas

from ml_hadoop_experiment.tensorflow import events


MODULE_TO_TEST = "ml_hadoop_experiment.tensorflow.events"


class Value(typing.NamedTuple):
    simple_value: typing.Optional[float]
    tag: str


class Summary(typing.NamedTuple):
    value: typing.List[Value]


class Event(typing.NamedTuple):
    step: int
    summary: Summary


def gen_events(step: int, tag: str, value: typing.Optional[float]):
    val = Value(simple_value=value, tag=tag)
    return Event(step=step, summary=Summary(value=[val]))


def generate_events():
    yield gen_events(42, "metric0", 32.4)
    yield gen_events(44, "metric0", 33.8)
    yield gen_events(44, "metric1", 23.3)  # Columns may have differents sizes
    yield gen_events(48, "metric0", None)  # Should not be in final dataframe
    yield gen_events(48, "metric2", None)  # Should not create a new column


EXPECTED_DATAFRAME = pandas.DataFrame.from_records(
    [{"step": 42, "name": "metric0", "value": 32.4},
     {"step": 44, "name": "metric0", "value": 33.8},
     {"step": 44, "name": "metric1", "value": 23.3}])


def test_parse_tf_events():
    with mock.patch(f'{MODULE_TO_TEST}.gen_events_iterator') as mock_events_iterator:
        mock_events_iterator.side_effect = lambda path: list(generate_events())
        df = events.get_all_metrics("mypath")
        assert df.equals(EXPECTED_DATAFRAME.reindex(columns=df.columns))


def test_parse_without_summary():
    with mock.patch(f'{MODULE_TO_TEST}.gen_events_iterator') as mock_events_iterator:
        mock_events_iterator.side_effect = lambda path: [Event(step=0, summary=None)]
        df = events.get_all_metrics("mypath")
        assert df.empty
