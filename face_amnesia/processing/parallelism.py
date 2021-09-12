# coding=utf-8
#
# Module to manage parallel tasks.
#
# Copyright: C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy

from face_amnesia.base import retrieval
from face_amnesia.base.data_set import DataSet


def process_in_parallel(data_set: DataSet,
                        callable_function,
                        args: tuple) -> DataSet:
    """
    Process given data set in parallel by
    presorting its vectors with a single LSH
    table and executing specified callable
    on each bucket using the available number
    of CPU cores. Concatenate resulting DataSet
    instances to a single one before returning.
    :param data_set: DataSet - Data points and
        corresponding metadata structures.
    :param callable_function: Callable function
        to be executed on each LSH bucket
        returning a DataSet instance.
    :param args: tuple - Further arguments
        for callable function.
    :return: DataSet - Concatenated result.

    The callable function returns a DataSet
    instance and assumes a list of data sets
    as its first and a list of indices as its
    second argument. Further arguments can
    be provided via the args parameter.
    """
    # Get number of available CPUs.
    num_cpus = cpu_count()
    # Create LSH table and feed
    # corresponding buckets.
    lsh = retrieval.Lsh(use_pca=False, num_hash_tables=1)
    lsh.append(data_set)
    hash_table = list(lsh.hash_tables[0].values())
    # Compute possible number
    # of buckets per CPU core.
    num_buckets = len(hash_table)
    buckets_per_thread = int(numpy.ceil(num_buckets / num_cpus))
    # Compute necessary
    # number of threads.
    num_threads = int(numpy.ceil(num_buckets / buckets_per_thread))
    # Prepare threads.
    thread_pool = ThreadPool()
    result_pool = list()
    # Start threads.
    pointer = 0
    for _i in range(num_threads - 1):
        indices = [i for i in range(pointer, pointer + buckets_per_thread)]
        current_args = (hash_table, indices) + args
        async_result = thread_pool.apply_async(callable_function, current_args)
        result_pool.append(async_result)
        pointer += buckets_per_thread
    last_indices = [i for i in range(pointer, num_buckets)]
    current_args = (hash_table, last_indices) + args
    async_result = thread_pool.apply_async(callable_function, current_args)
    result_pool.append(async_result)
    # Wait for threads to be terminated
    # and concatenate the results.
    new_data_set = None
    for async_result in result_pool:
        received_data_sets = async_result.get()
        for ds in received_data_sets:
            if new_data_set is None:
                new_data_set = ds
            else:
                new_data_set.add_data_set(ds)
    return new_data_set
