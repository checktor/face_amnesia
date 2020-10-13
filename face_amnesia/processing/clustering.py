# coding=utf-8
#
# Data point clustering module.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.


import numpy

from face_amnesia import settings
from face_amnesia.base.data_set import DataSet
from face_amnesia.processing import parallelism
from face_amnesia.utils import calculation


def _get_clustered_data_set(data_set: DataSet, clusters: list) -> DataSet:
    """
    Combine vectors within each cluster to a single
    data point and corresponding metadata structure.
    Use mean vector as the cluster's representative
    and all metadata structures excluding duplicates.
    :param data_set: DataSet - Data points
        and corresponding metadata structures.
    :param clusters: list(list) - Data point
        indices contained in each cluster.
    :return: DataSet - Clustered data points
        and corresponding metadata structures.
    """
    # Create lists to store compressed vectors
    # and corresponding metadata structures.
    new_data_points = list()
    new_metadata = list()

    for index_list in clusters:
        # Compress data points in current cluster
        # by computing corresponding mean vector.
        current_data_points = [data_set.get_vector_at_index(i) for i in index_list]
        mean_vector = calculation.get_mean(current_data_points)
        new_data_points.append(mean_vector)

        # Compress metadata structures in current
        # cluster by using corresponding merge function.
        current_metadata_structure = set()
        for i in index_list:
            current_metadata_structure = DataSet.merge_metadata(current_metadata_structure,
                                                                data_set.get_metadata_at_index(i))
        new_metadata.append(current_metadata_structure)

    # Return compressed data.
    return DataSet(new_data_points, new_metadata)


def _chinese_whispers_clustering(data_set: DataSet,
                                 distance_threshold: float,
                                 num_iterations: int = 30) -> DataSet:
    """
    Cluster given data set in a way that only a
    single data point within groups of nearby points
    is kept. Use a temporary graph structure with
    one node per data point and add an unweighted
    edge between every pair of nodes with specified
    maximum distance. Search for groups of nodes
    with many pairwise edges using chinese whispers
    graph clustering. This algorithm works in three
    steps:
        1. Assign each node a different label.
        2. Iterate through all nodes assigning
            each node the label used for the
            majority of its neighbors.
        3. Repeat second step a specified number
            of times (use 30 iterations as
            indicated in original paper).
    See Chris Biemann: "Chinese Whispers - an Efficient Graph
    Clustering Algorithm and its Application to Natural
    Language Processing Problems" (2006) for details.
    :param data_set: DataSet - Data points
        and corresponding metadata structures.
    :param distance_threshold: float - Maximum
        distance between two data points in
        order to consider them as candidates
        for the same cluster.
    :param num_iterations: int - Maximum number of
        passes through all data points (default: 30).
    :return: DataSet - Clustered data points
        and corresponding metadata structures.

    Compare with dlib implementation:
    http://dlib.net/python/index.html#dlib.chinese_whispers_clustering
    """
    # Create adjacency list representation
    # of given data set only considering
    # edges connecting data points within
    # specified distance threshold.
    edges = [[] for _i in range(len(data_set))]
    for i in range(len(data_set)):
        for j in range(i + 1, len(data_set)):
            # Get distance from vector at
            # index i to vector at index j.
            distance = calculation.get_distance(data_set.get_vector_at_index(i), data_set.get_vector_at_index(j))
            # Compare current distance to specified threshold.
            if numpy.less_equal(distance, distance_threshold):
                edges[i].append(j)
                edges[j].append(i)

    # Initialize clustering labels so that every
    # data point is assigned a different cluster.
    labels = [i for i in range(len(data_set))]

    # Initialize list to specify current data
    # point sequence, i.e. the order in which
    # data point labels are adjusted.
    sequence = [i for i in range(len(data_set))]

    for _i in range(num_iterations):
        # Create flag to track if
        # any label was adjusted.
        label_adjusted = False
        # Create random ordering of data points.
        numpy.random.shuffle(sequence)
        # Iterate through data points in this order.
        label_counter = dict()
        for current_index in sequence:
            # Count all occurrences of the
            # same label in direct neighborhood
            # of currently chosen data point.
            label_counter.clear()
            for neighbor_index in edges[current_index]:
                neighbor_label = labels[neighbor_index]
                if neighbor_label in label_counter:
                    label_counter[neighbor_label] += 1
                else:
                    label_counter[neighbor_label] = 1
            # Get the most common label.
            best_label = -1
            best_count = -1
            for label, count in label_counter.items():
                if count > best_count:
                    best_label = label
                    best_count = count
            # Assign most common label to current data
            # point only if it exists and is different.
            if (best_label >= 0) and (best_label != labels[current_index]):
                labels[current_index] = best_label
                label_adjusted = True
        # Check if any label was adjusted. If not, the
        # process has converged and can be stopped.
        if not label_adjusted:
            break

    # Reconstruct corresponding clusters.
    clusters = dict()
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]
    clusters = list(clusters.values())

    # Return data set containing
    # one representative description
    # vector with corresponding metadata
    # structures for each cluster.
    return _get_clustered_data_set(data_set, clusters)


def _cluster(all_data_sets: list,
             selected_indices: list,
             distance_threshold: float) -> list:
    """
    Iteratively process data sets at
    specified indices in given list.
    :param all_data_sets: list - DataSet instances.
    :param selected_indices: list - Indices
        specifying selected data sets.
    :param distance_threshold: float - Maximum
        distance between two data points in
        order to consider them as candidates
        for the same cluster.
    :return: list - Selected data sets properly clustered.
        Return empty list if no data set is selected.
    """
    selected_data_sets = []
    for index in selected_indices:
        selected_data_sets.append(_chinese_whispers_clustering(all_data_sets[index], distance_threshold))
    return selected_data_sets


def cluster(data_set: DataSet,
            distance_threshold: float = 0,
            use_lsh: bool = False) -> DataSet:
    """
    Cluster given data set in a way that only a
    single data point within groups of nearby points
    is kept. Use specified distance threshold as
    maximum distance between two data points in
    order to consider them as candidates for the
    same cluster. If desired, use a single LSH
    table to presort data points and process
    each of its created buckets in parallel.
    :param data_set: DataSet - Data points
        and corresponding metadata structures.
    :param distance_threshold: float - Maximum
        distance between two data points in
        order to consider them as candidates
        for the same cluster. Defaults to
        value specified in settings module.
    :param use_lsh: bool - Indicate if LSH
        structure should be used to presort
        given data in buckets and process
        them in parallel (default: False).
    :return: DataSet - Clustered data points
        and corresponding metadata structures.
        Return given data set in case it is empty.
    :raise TypeError: In case of an invalid data set.
    """
    # Check if given data is actually a DataSet instance.
    if not isinstance(data_set, DataSet):
        raise TypeError

    # Check if matrix is empty.
    if len(data_set.data_points) <= 0:
        return data_set

    # Validate distance threshold.
    if numpy.less_equal(distance_threshold, 0):  # numpy.isnan(distance_threshold):
        # Distance threshold is not provided
        # or invalid, so fall back to default.
        distance_threshold = settings.DISTANCE_THRESHOLD_CLUSTERING

    if use_lsh:
        return parallelism.process_in_parallel(data_set, _cluster, (distance_threshold,))
    else:
        selected_data_sets = _cluster([data_set], [0], distance_threshold)
        return selected_data_sets[0]
