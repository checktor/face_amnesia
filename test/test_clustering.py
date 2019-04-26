# coding=utf-8
#
# Unit test for clustering module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import unittest

import numpy

from face_amnesia.base.data_set import DataSet
from face_amnesia.processing import clustering
from test import source


class TestClustering(unittest.TestCase):
    # TODO: Unit tests do not use presorting of data points via LSH.

    def _test_cluster(self,
                      source_matrix,
                      source_metadata,
                      threshold,
                      target_matrix,
                      target_metadata):
        # Test data as it is.
        given_data_set = DataSet(source_matrix, source_metadata)
        received_data_set = clustering.cluster(given_data_set, threshold)
        self.assertIsInstance(received_data_set, DataSet)
        numpy.testing.assert_allclose(received_data_set.data_points, target_matrix)
        for i in range(len(received_data_set)):
            self.assertSetEqual(received_data_set.get_metadata_at_index(i), target_metadata[i])

        # Test empty metadata.
        received_data_set = clustering.cluster(DataSet(source_matrix, []), threshold)
        self.assertIsInstance(received_data_set, DataSet)
        numpy.testing.assert_allclose(received_data_set.data_points, target_matrix)
        for i in range(len(received_data_set)):
            self.assertSetEqual(received_data_set.get_metadata_at_index(i), set())

    def test_cluster(self):
        # Test valid matrices and metadata.
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           5,
                           source.MATRIX_ONE_CLUSTER_NUMPY,
                           source.MATRIX_ONE_CLUSTER_METADATA)
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           2,
                           source.MATRIX_TWO_CLUSTER_NUMPY,
                           source.MATRIX_TWO_CLUSTER_METADATA)
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           0.5,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA)

        # Test matrix with duplicates.
        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL,
                           5,
                           source.MATRIX_WITH_DUPLICATES_ONE_CLUSTER_NUMPY,
                           source.MATRIX_ONE_CLUSTER_METADATA)
        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL,
                           2,
                           source.MATRIX_WITH_DUPLICATES_TWO_CLUSTER_NUMPY,
                           source.MATRIX_TWO_CLUSTER_METADATA)
        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL,
                           0.5,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA)

        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL,
                           5,
                           source.MATRIX_WITH_DUPLICATES_ONE_CLUSTER_NUMPY,
                           source.MATRIX_WITH_DUPLICATES_ONE_CLUSTER_METADATA_UNEQUAL)
        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL,
                           2,
                           source.MATRIX_WITH_DUPLICATES_TWO_CLUSTER_NUMPY,
                           source.MATRIX_WITH_DUPLICATES_TWO_CLUSTER_METADATA_UNEQUAL)
        self._test_cluster(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL,
                           0.5,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_UNEQUAL)

        # Test matrix containing only one vector.
        self._test_cluster(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS,
                           source.MATRIX_ONE_ENTRY_METADATA,
                           0.5,
                           source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS,
                           source.MATRIX_ONE_ENTRY_METADATA)

        # Test empty matrix.
        self._test_cluster([], [], 0.5, [], [])

        # Test case of invalid distance threshold.
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           0,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA)
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           -1,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA)
        self._test_cluster(source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA,
                           numpy.NAN,
                           source.MATRIX_LIST_OF_NUMPYS,
                           source.MATRIX_METADATA)

        # Test other data types and None values.
        self.assertRaises(TypeError, clustering.cluster,
                          source.CUSTOM_DICT,
                          0.5)
        self.assertRaises(TypeError, clustering.cluster,
                          source.CUSTOM_STR,
                          0.5)
        self.assertRaises(TypeError, clustering.cluster,
                          None,
                          0.5)
        self.assertRaises(TypeError, clustering.cluster,
                          DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA),
                          source.CUSTOM_DICT)
        self.assertRaises(TypeError, clustering.cluster,
                          DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA),
                          source.CUSTOM_STR)
        self.assertRaises(TypeError, clustering.cluster,
                          DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA),
                          None)
