# coding=utf-8
#
# Data point comparison module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.


import functools

import numpy

from face_amnesia.base.data_set import DataSet


def memory_address_comparator(first_item: tuple, second_item: tuple) -> int:
    """
    Custom comparator using the memory address of each item's vector.
    :param first_item: tuple(numpy.ndarray, set) - First item.
    :param second_item: tuple(numpy.ndarray, set) - Second item.
    :return: int - Indicate if first item
        is strictly smaller than (-1), equal
        to (0) or greater than (1) second one.

    Note that this approach assumes that each item's
    NumPy array is only created once and is never
    deeply copied afterwards (i.e. its memory address
    always stays the same). However, NumPy creates
    deep copies of given arrays and matrices on
    creation of its inherent data structure which may
    be taken into account before using this approach.
    """
    first_vector = first_item[0]
    second_vector = second_item[0]
    if id(first_vector) < id(second_vector):
        return -1
    elif id(first_vector) == id(second_vector):
        return 0
    else:
        return 1


def vector_entry_comparator(first_item: tuple, second_item: tuple) -> int:
    """
    Custom comparator using the entries of each item's
    vector. In other words: vectors are compared according
    to their first entry. In case of equality, they are
    compared according to the second one, etc.
    :param first_item: tuple(numpy.ndarray, set) - First item.
    :param second_item: tuple(numpy.ndarray, set) - Second item.
    :return: int - Indicate if first item
        is strictly smaller than (-1), equal
        to (0) or greater than (1) second one.
    """
    first_vector = first_item[0]
    second_vector = second_item[0]
    for i in range(first_vector.shape[0]):
        if numpy.less(first_vector[i], second_vector[i]):
            return -1
        elif numpy.greater(first_vector[i], second_vector[i]):
            return 1
    return 0


def compare(first_data_set: DataSet, second_data_set: DataSet) -> tuple:
    """
    Compare given data sets by computing the
    number of metadata entries present in first
    but not in second data set and vice versa.
    Return both values as a tuple of integers.
    Intuitively, both numbers are 0 in case
    of strictly identical metadata structures.
    :param first_data_set: DataSet - Data points
        and corresponding metadata structures.
    :param second_data_set: DataSet - Data points
        and corresponding metadata structures.
    :return: (int, int) - Number of metadata
        elements present in first but not
        in second data set (and vice versa).
    :raise ValueError: In case one
        DataSet instance is missing.
    """
    # Check if first data structure is
    # actually not a DataSet instance.
    if not isinstance(first_data_set, DataSet):
        raise ValueError

    # Check if second data structure is
    # actually not a DataSet instance.
    if not isinstance(second_data_set, DataSet):
        raise ValueError

    # Collect metadata entries of first
    # data set in a single set structure.
    first_metadata = set()
    for i in range(len(first_data_set)):
        metadata = first_data_set.get_metadata_at_index(i)
        first_metadata.update(metadata)

    # Collect metadata entries of second
    # data set in a single set structure.
    second_metadata = set()
    for i in range(len(second_data_set)):
        metadata = second_data_set.get_metadata_at_index(i)
        second_metadata.update(metadata)

    num_unique_elements_in_first_metadata = len(first_metadata.difference(second_metadata))
    num_unique_elements_in_second_metadata = len(second_metadata.difference(first_metadata))

    return num_unique_elements_in_first_metadata, num_unique_elements_in_second_metadata


def is_equal(first_data_set: DataSet, second_data_set: DataSet) -> bool:
    """
    Check if both data sets contain the same data
    entries, i.e. the same (data point, metadata)
    pairs (regardless of the specific ordering).
    :param first_data_set: DataSet - Data points
        and corresponding metadata structures.
    :param second_data_set: DataSet - Data points
        and corresponding metadata structures.
    :return: bool - True if both data sets
        contain the same (data point, metadata)
        pairs and False otherwise.
    :raise ValueError: In case no
        DataSet instance is provided.
    """
    # Check if first data structure is
    # actually not a DataSet instance.
    if not isinstance(first_data_set, DataSet):
        raise ValueError

    # Check if second data structure is
    # actually not a DataSet instance.
    if not isinstance(second_data_set, DataSet):
        raise ValueError

    # Check if number of data points is matching.
    if len(first_data_set) != len(second_data_set):
        return False

    # Convert each data set to a sequence
    # of (data point, metadata) tuples.
    first_data = zip(first_data_set.data_points, first_data_set.metadata)
    second_data = zip(second_data_set.data_points, second_data_set.metadata)

    # Sort tuple sequences according
    # to corresponding vector entries.
    sorted_first_data = sorted(first_data, key=functools.cmp_to_key(vector_entry_comparator))
    sorted_second_data = sorted(second_data, key=functools.cmp_to_key(vector_entry_comparator))

    # Compare both structures element by element.
    for i in range(len(sorted_first_data)):
        # Compare vector of current elements.
        result = numpy.allclose(sorted_first_data[i][0], sorted_second_data[i][0])
        if not result:
            return False
        # Compare metadata of current element.
        result = sorted_first_data[i][1] == sorted_second_data[i][1]
        if not result:
            return False
    return True
