# coding=utf-8
#
# Unit test for deduplication module.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import copy
import unittest

from face_amnesia.base.data_set import DataSet
from face_amnesia.processing import comparison, deduplication
from test import source


class TestDeduplication(unittest.TestCase):
    # TODO: Unit tests do not use presorting of data points via LSH.

    def _test_deduplicate(self,
                          source_matrix,
                          source_metadata,
                          target_matrix,
                          target_metadata,
                          sort_by_memory_address):
        source_data_set = DataSet(source_matrix, source_metadata)
        copy_source_data_set = copy.deepcopy(source_data_set)

        target_data_set = DataSet(target_matrix, target_metadata)

        result_data_set = deduplication.deduplicate(copy_source_data_set, sort_by_memory_address=sort_by_memory_address)

        if len(source_data_set) > 0:
            self.assertIsNot(copy_source_data_set, result_data_set)
            self.assertIsNot(copy_source_data_set.data_points, result_data_set.data_points)
            self.assertIsNot(copy_source_data_set.metadata, result_data_set.metadata)
            self.assertTrue(comparison.is_equal(target_data_set, result_data_set))
        else:
            self.assertEqual(len(result_data_set), 0)

    def test_deduplicate(self):
        # Test data sets without duplicates.
        self._test_deduplicate(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA,
                               source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA,
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA,
                               source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA,
                               sort_by_memory_address=False)
        self._test_deduplicate(source.MATRIX_LIST_OF_NUMPYS, [],
                               source.MATRIX_LIST_OF_NUMPYS, [],
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_LIST_OF_NUMPYS, [],
                               source.MATRIX_LIST_OF_NUMPYS, [],
                               sort_by_memory_address=False)

        self._test_deduplicate(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA,
                               source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA,
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA,
                               source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA,
                               sort_by_memory_address=False)
        self._test_deduplicate(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, [],
                               source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, [],
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, [],
                               source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, [],
                               sort_by_memory_address=False)

        self._test_deduplicate([], [],
                               [], [],
                               sort_by_memory_address=True)
        self._test_deduplicate([], [],
                               [], [],
                               sort_by_memory_address=False)

        # Test data sets with duplicates.
        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL,
                               source.MATRIX_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_EQUAL,
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL,
                               source.MATRIX_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_EQUAL,
                               sort_by_memory_address=False)

        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL,
                               source.MATRIX_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_UNEQUAL,
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL,
                               source.MATRIX_LIST_OF_NUMPYS,
                               source.MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_UNEQUAL,
                               sort_by_memory_address=False)

        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS, [],
                               source.MATRIX_LIST_OF_NUMPYS, [],
                               sort_by_memory_address=True)
        self._test_deduplicate(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS, [],
                               source.MATRIX_LIST_OF_NUMPYS, [], sort_by_memory_address=False)

        # Test other data types and None values.
        self.assertRaises(ValueError, deduplication.deduplicate, source.CUSTOM_DICT)
        self.assertRaises(ValueError, deduplication.deduplicate, source.CUSTOM_STR)
        self.assertRaises(ValueError, deduplication.deduplicate, None)
