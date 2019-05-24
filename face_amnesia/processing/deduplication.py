# coding=utf-8
#
# Data point deduplication module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import functools

from face_amnesia.base.data_set import DataSet
from face_amnesia.processing import comparison
from face_amnesia.processing import parallelism


def _deduplicate_by_sorting(data_set: DataSet, callable_comparator) -> DataSet:
    """
    Find and remove duplicates in given data
    set merging corresponding metadata structures.
    Sort data set with specified comparator in
    order to identify duplicated entries.
    :param data_set: DataSet - Data points and
        corresponding metadata structures.
    :param callable_comparator: Callable
        comparator function.
    :return: DataSet - Data set without duplicates.
    """
    # Use Python's zip() function to concatenate the vector
    # and metadata lists to a single list of tuples
    # mapping each vector to its metadata structure.
    data = zip(data_set.data_points, data_set.metadata)
    # Sort this list using specified comparator.
    sorted_data = sorted(data, key=functools.cmp_to_key(callable_comparator))
    # Create new lists to store unique data
    # points and corresponding metadata structures.
    new_data_points = list()
    new_metadata = list()
    # Iterate through sorted list
    # considering equal entries only once.
    last_entry = None
    for entry in sorted_data:
        if last_entry is None or callable_comparator(entry, last_entry) != 0:
            # Current data point was never seen before,
            # so append it to list of unique data points.
            new_data_points.append(entry[0])
            new_metadata.append(entry[1])
            last_entry = entry
        else:
            # Current data point is already present,
            # therefore only merge its metadata.
            new_metadata[-1] = DataSet.merge_metadata(new_metadata[-1], entry[1])
    # Return data set without duplicates.
    return DataSet(new_data_points, new_metadata)


def _deduplicate(all_data_sets: list,
                 selected_indices: list,
                 sort_by_memory_address: float) -> list:
    """
    Iteratively process data sets at
    specified indices in given list.
    :param all_data_sets: list - DataSet instances.
    :param selected_indices: list - Indices
        specifying selected data sets.
    :param sort_by_memory_address: bool - Use
        sorting according to memory address
        (in case of True) or according to
        vector entries (otherwise) in order
        to identify duplicates.
    :return: list - Selected data sets without duplicates.
        Return empty list if no data set is selected.
    """
    selected_data_sets = []
    if sort_by_memory_address:
        for index in selected_indices:
            selected_data_sets.append(
                _deduplicate_by_sorting(all_data_sets[index], comparison.memory_address_comparator))
    else:
        for index in selected_indices:
            selected_data_sets.append(
                _deduplicate_by_sorting(all_data_sets[index], comparison.vector_entry_comparator))
    return selected_data_sets


def deduplicate(data_set: DataSet,
                sort_by_memory_address: bool = False,
                use_lsh: bool = False) -> DataSet:
    """
    Find and remove duplicates in given data set merging
    corresponding metadata structures. Use either sorting
    by memory address or by vector entries (specified by
    corresponding flag) to identify duplicates. If desired,
    use a single LSH table to presort data points and
    process each of its created buckets in parallel.
    :param data_set: DataSet - Data points and
        corresponding metadata structures.
    :param sort_by_memory_address: bool - Use
        sorting according to memory address
        (in case of True) or according to
        vector entries (otherwise) in order
        to identify duplicates (default: False).
    :param use_lsh: bool - Indicate if LSH
        structure should be used to presort
        given data in buckets and process
        them in parallel (default: False).
    :return: DataSet - Data set without duplicates.
        Return given data set directly in case it is empty.
    :raise ValueError: In case of an invalid data set.

    Note that the elimination of duplicates is
    processed in-place without deeply copying the
    data. In case this is not the desired behavior,
    create a deep copy of your data before using
    this function.
    """
    # Check if given data structure is
    # actually not a DataSet instance.
    if not isinstance(data_set, DataSet):
        raise ValueError

    # Check if data set is empty.
    if len(data_set.data_points) <= 0:
        return data_set

    if use_lsh:
        return parallelism.process_in_parallel(data_set, _deduplicate, (sort_by_memory_address,))
    else:
        selected_data_sets = _deduplicate([data_set], [0], sort_by_memory_address)
        return selected_data_sets[0]
