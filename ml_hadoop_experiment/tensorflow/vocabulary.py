from collections import defaultdict
from functools import singledispatch
import math
import os
from typing import Dict, Iterator, Tuple, List, Any

import pyspark
from cluster_pack import filesystem


def _get_columns_values(row: pyspark.Row, column_to_keys: Dict[str, List[str]]
                        ) -> Iterator[Tuple[Tuple, int]]:
    '''
    col_names: names of columns of interest
    Return: a list of tuple composed of:
    - The name of a column of interest
    - An iterator associating cardinalities to
    doublet(column name, value) found in the column
    of interest
    '''
    for col_name, keys in column_to_keys.items():
        if col_name in row:
            col_value = row[col_name]
            # dispatch this value to all the relevant keys
            if col_value is not None:
                for key in keys:
                    if isinstance(col_value, list):
                        yield from (((key, val), 1) for val in col_value)
                    else:
                        yield ((key, col_value), 1)


def _get_vocab_values(
    rdd: pyspark.RDD,
    col_dict: Dict[str, List[str]],
    threshold: int
) -> Dict[str, List[Any]]:

    column_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, values in col_dict.items():
        for column_name in values:
            column_to_keys[column_name].append(key)

    vocab_values_rdd = rdd.flatMap(
        lambda row: _get_columns_values(row, column_to_keys)
    ).reduceByKey(
        lambda x, y: x + y, numPartitions=math.ceil(rdd.getNumPartitions()/4)
    )

    # _get_columns_values() will always return a count of 1 for each modality. As a result, a
    # threshold of 0 or 1 on the aggregated count is always going to return true.
    # In this case, we don't call filter() as the predicate would be trivial and it would
    # create a job for nothing.
    if threshold > 1:
        vocab_values_rdd = vocab_values_rdd.filter(
            lambda x: x[1] >= threshold
        )

    vocab_values = vocab_values_rdd.collect()

    vocab_values_dict: Dict[str, List[Any]] = defaultdict(list)
    for (col_name, value), _ in vocab_values:
        vocab_values_dict[col_name].append(value)
    return vocab_values_dict


def _write_vocab_files(
    vocab_values_dict: Dict[str, List[Any]],
    path: str,
    col_names: Dict[str, List[str]]
) -> List[str]:
    voc_files_list = []
    fs, _ = filesystem.resolve_filesystem_and_path(path)
    if not fs.exists(path):
        fs.mkdir(path)
    for key_name in col_names.keys():
        voc_file_path = os.path.join(path, f'{key_name}.voc')
        with fs.open(voc_file_path, "wb") as fd:
            voc_files_list.append(voc_file_path)
            if key_name in vocab_values_dict:
                first_elem_written = False
                for val in vocab_values_dict[key_name]:
                    value = f'{val}'
                    # This is to avoid empty string modalities
                    # as they're not supported by Tensorflow
                    if value != "":
                        if first_elem_written:
                            fd.write('\n'.encode())
                        fd.write(value.encode())
                        first_elem_written = True
    return voc_files_list


@singledispatch
def gen_vocab_files(
    columns: Dict[str, List[str]],
    rdd: pyspark.RDD,
    path: str,
    threshold: int = 0
) -> List[str]:
    raise NotImplementedError('Unsupported type')


@gen_vocab_files.register(list)
def gen_vocab_files_from_list(
    columns: List[str],
    rdd: pyspark.RDD,
    path: str,
    threshold: int = 0
) -> List[str]:
    '''
    columns: names of columns for which to create a vocabulary file (1 file per column)
    path: path where vocabulary files are written
    threshold: given a column of interest, any value with a cardinality
    smaller than the threshold is ignored
    NB: lists columns will generate vocabulary files based on the items within the list,
    it won't generate a vocabulary file of lists.
    '''

    columns_dict: Dict[str, List[str]] = {}
    for col_name in columns:
        columns_dict[col_name] = [col_name]

    return gen_vocab_files_from_dict(columns_dict, rdd, path, threshold)


@gen_vocab_files.register(dict)
def gen_vocab_files_from_dict(
    columns: Dict[str, List[str]],
    rdd: pyspark.RDD,
    path: str,
    threshold: int = 0
) -> List[str]:
    '''
    columns: a dictionary containing which columns to merge and put in a dictionary file
    path: path where vocabulary files are written
    threshold: given a column of interest, any value with a cardinality
    smaller than the threshold is ignored
    NB: lists columns will generate vocabulary files based on the items within the list,
    it won't generate a vocabulary file of lists.
    '''
    vocab_values_dict = _get_vocab_values(rdd, columns, threshold)
    return _write_vocab_files(vocab_values_dict, path, columns)
