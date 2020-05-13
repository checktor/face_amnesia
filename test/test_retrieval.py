# coding=utf-8
#
# Unit test for data point retrieval module.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
import shutil
import unittest

import numpy

from face_amnesia import settings
from face_amnesia.base import storage
from face_amnesia.base.data_set import DataSet
from face_amnesia.base.retrieval import LINEAR_FILE_NAME_PATTERN, PARAMETER_FILE_NAME_PATTERN
from face_amnesia.base.retrieval import LINEAR_STRUCTURE_FOLDER_NAME, LSH_STRUCTURE_FOLDER_NAME
from face_amnesia.base.retrieval import Linear, Lsh
from face_amnesia.base.storage import METADATA_FILE_NAME_EXTENSION
from face_amnesia.processing import comparison
from test import source


class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create folders.
        os.makedirs(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH), exist_ok=True)
        os.makedirs(os.path.join(source.RETRIEVAL_DATA_SUBFOLDER_PATH), exist_ok=True)

        # Write test data.
        storage.write_data_set(os.path.join(source.RETRIEVAL_DATA_BASE_PATH, source.FILE_DATA_NAME),
                               DataSet(source.FILE_MATRIX, source.FILE_METADATA))

        storage.write_data_set(os.path.join(source.RETRIEVAL_DATA_FOLDER_PATH, source.FOLDER_DATA_NAME),
                               DataSet(source.FOLDER_MATRIX, source.FOLDER_METADATA))

        storage.write_data_set(os.path.join(source.RETRIEVAL_DATA_SUBFOLDER_PATH, source.SUBFOLDER_FIRST_DATA_NAME),
                               DataSet(source.SUBFOLDER_FIRST_MATRIX, source.SUBFOLDER_FIRST_METADATA))
        storage.write_data_set(os.path.join(source.RETRIEVAL_DATA_SUBFOLDER_PATH, source.SUBFOLDER_SECOND_DATA_NAME),
                               DataSet(source.SUBFOLDER_SECOND_MATRIX, source.SUBFOLDER_SECOND_METADATA))

    def _add_first_batch_of_data_points(self, handler):
        # Execute appending operations in random order.
        operations = [0, 1, 2, 3, 4]
        numpy.random.shuffle(operations)
        for i in operations:
            if i == 0:
                # Append single file.
                handler.append_file(os.path.join(source.RETRIEVAL_DATA_BASE_PATH, source.FILE_DATA_NAME))
            elif i == 1:
                # Append data points currently in memory.
                handler.append(DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
            elif i == 2:
                # Append empty data set.
                handler.append(DataSet([], []))
            elif i == 3:
                # Append invalid data.
                self.assertRaises(ValueError, handler.append_file, source.CUSTOM_DICT)
                self.assertRaises(ValueError, handler.append_file, source.CUSTOM_STR)
                self.assertRaises(ValueError, handler.append_file, "")
                self.assertRaises(ValueError, handler.append_file, None)
                self.assertRaises(ValueError, handler.append_folder, source.CUSTOM_DICT)
                self.assertRaises(ValueError, handler.append_folder, source.CUSTOM_STR)
                self.assertRaises(ValueError, handler.append_folder, "")
                self.assertRaises(ValueError, handler.append_folder, None)
                self.assertRaises(ValueError, handler.append, source.CUSTOM_DICT)
                self.assertRaises(ValueError, handler.append, source.CUSTOM_STR)
                self.assertRaises(ValueError, handler.append, "")
                self.assertRaises(ValueError, handler.append, None)
            elif i == 4:
                # Append single file once again.
                handler.append_file(os.path.join(source.RETRIEVAL_DATA_BASE_PATH, source.FILE_DATA_NAME))

        # Try to append data of wrong dimension.
        self.assertRaises(ValueError, handler.append, DataSet([source.VECTOR_3D_NUMPY], [source.VECTOR_3D_METADATA]))

    def _add_second_batch_of_data_points(self, handler):
        # Execute appending operations in random order.
        operations = [0, 1, 2]
        numpy.random.shuffle(operations)
        for i in operations:
            if i == 0:
                # Append whole directory.
                handler.append_folder(source.RETRIEVAL_DATA_FOLDER_PATH)
            elif i == 1:
                # Append empty data set.
                handler.append(DataSet([], []))
            elif i == 2:
                j = numpy.random.randint(2)
                if j == 0:
                    # Append subdirectory file once again.
                    handler.append_file(
                        os.path.join(source.RETRIEVAL_DATA_SUBFOLDER_PATH, source.SUBFOLDER_FIRST_DATA_NAME))
                elif j == 1:
                    # Append whole directory once again.
                    handler.append_folder(source.RETRIEVAL_DATA_FOLDER_PATH)

        # Try to append data of wrong dimension.
        self.assertRaises(ValueError, handler.append, DataSet([source.VECTOR_3D_NUMPY], [source.VECTOR_3D_METADATA]))

    def _query_first_batch_of_data_points(self, handler):
        query_vector = numpy.array([1.5, 1], dtype=settings.VECTOR_ENTRY_DATA_TYPE)

        expected_data_points = [source.FIRST_VECTOR_2D_NUMPY,
                                source.FIRST_VECTOR_2D_REVERSED_NUMPY]
        expected_metadata = [source.FIRST_VECTOR_2D_METADATA,
                             source.FIRST_VECTOR_2D_REVERSED_METADATA]
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, (numpy.sqrt(5) / 2))
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        expected_data_points = [source.FIRST_VECTOR_2D_REVERSED_NUMPY]
        expected_metadata = [source.FIRST_VECTOR_2D_REVERSED_METADATA]
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, numpy.sqrt(5) / 2 - 0.001)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        query_vector = numpy.array([5.5, 5.5], dtype=settings.VECTOR_ENTRY_DATA_TYPE)

        expected_data_points = source.FILE_MATRIX
        expected_metadata = source.FILE_METADATA
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, (numpy.sqrt(2) / 2))
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        expected_data_points = []
        expected_metadata = []
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, (numpy.sqrt(2) / 2) - 0.001)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        retrieved_result = handler.query(query_vector, 0)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))
        retrieved_result = handler.query(query_vector, -1)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        query_vector = source.FIRST_VECTOR_2D_NUMPY

        expected_data_points = [source.FIRST_VECTOR_2D_NUMPY]
        expected_metadata = [source.FIRST_VECTOR_2D_METADATA]
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, 0)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        self.assertRaises(ValueError, handler.query, source.CUSTOM_STR, 2)
        self.assertRaises(ValueError, handler.query, source.CUSTOM_DICT, 2)
        self.assertRaises(ValueError, handler.query, "", 2)
        self.assertRaises(ValueError, handler.query, None, 2)
        self.assertRaises(ValueError, handler.query, [], 2)
        self.assertRaises(ValueError, handler.query, numpy.empty(0), 2)

    def _query_second_batch_of_data_points(self, handler):
        query_vector = numpy.array([7.5, 7.5], dtype=settings.VECTOR_ENTRY_DATA_TYPE)

        expected_data_points = source.FOLDER_MATRIX
        expected_metadata = source.FOLDER_METADATA
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, (numpy.sqrt(2) / 2))
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        expected_data_points = []
        expected_metadata = []
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, (numpy.sqrt(2) / 2) - 0.001)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

        query_vector = source.SUBFOLDER_SECOND_MATRIX[0]

        expected_data_points = source.SUBFOLDER_SECOND_MATRIX
        expected_metadata = source.SUBFOLDER_SECOND_METADATA
        expected_result = DataSet(expected_data_points, expected_metadata)
        retrieved_result = handler.query(query_vector, 0)
        self.assertTrue(comparison.is_equal(expected_result, retrieved_result))

    def test_linear_retrieval_in_memory(self):
        # Create in-memory handler.
        created_handler = Linear()

        self.assertEqual(created_handler.dimension, 0)

        self._add_first_batch_of_data_points(created_handler)

        self.assertGreater(created_handler.dimension, 0)

        self.assertEqual(len(created_handler.data_point_storage), source.NUM_DATA_POINTS_FIRST_BATCH)
        self.assertIsNone(created_handler.file_path_storage)
        self.assertListEqual(list(sorted(created_handler.file_name_cache)),
                             list(sorted(source.FILE_NAMES_FIRST_BATCH)))

        self._query_first_batch_of_data_points(created_handler)

        self._add_second_batch_of_data_points(created_handler)

        self.assertEqual(len(created_handler.data_point_storage),
                         source.NUM_DATA_POINTS_FIRST_BATCH + source.NUM_DATA_POINTS_SECOND_BATCH)
        self.assertIsNone(created_handler.file_path_storage)
        self.assertListEqual(list(sorted(created_handler.file_name_cache)),
                             list(sorted(source.FILE_NAMES_FIRST_BATCH + source.FILE_NAMES_SECOND_BATCH)))

        self._query_second_batch_of_data_points(created_handler)

    def test_linear_retrieval_out_of_core(self):
        # Create out-of-core handler.
        created_handler = Linear(source.RETRIEVAL_TESTING_BASE_PATH)

        self.assertEqual(created_handler.dimension, 0)

        self._add_first_batch_of_data_points(created_handler)

        self.assertGreater(created_handler.dimension, 0)

        self.assertIsNone(created_handler.data_point_storage)
        out_of_core_file_path = os.path.join(source.RETRIEVAL_TESTING_BASE_PATH, LINEAR_STRUCTURE_FOLDER_NAME,
                                             "{}_{}".format(LINEAR_FILE_NAME_PATTERN, 0))
        out_of_core_file_name = os.path.basename(out_of_core_file_path)
        self.assertListEqual(list(sorted(created_handler.file_path_storage)),
                             list(sorted(source.FILE_PATHS_FIRST_BATCH + [out_of_core_file_path])))
        self.assertListEqual(list(sorted(created_handler.file_name_cache)),
                             list(sorted(source.FILE_NAMES_FIRST_BATCH + [out_of_core_file_name])))

        self._query_first_batch_of_data_points(created_handler)

        # Restore existing handler.
        handler_from_existing_folder = Linear(source.RETRIEVAL_TESTING_BASE_PATH)

        self.assertGreater(handler_from_existing_folder.dimension, 0)

        self._add_second_batch_of_data_points(handler_from_existing_folder)

        self.assertIsNone(handler_from_existing_folder.data_point_storage)
        self.assertListEqual(list(sorted(handler_from_existing_folder.file_path_storage)),
                             list(sorted(source.FILE_PATHS_FIRST_BATCH +
                                         [out_of_core_file_path] +
                                         source.FILE_PATHS_SECOND_BATCH)))
        self.assertListEqual(list(sorted(handler_from_existing_folder.file_name_cache)),
                             list(sorted(source.FILE_NAMES_FIRST_BATCH +
                                         [out_of_core_file_name] +
                                         source.FILE_NAMES_SECOND_BATCH)))

        self._query_first_batch_of_data_points(handler_from_existing_folder)
        self._query_second_batch_of_data_points(handler_from_existing_folder)

        shutil.rmtree(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH, LINEAR_STRUCTURE_FOLDER_NAME))
        os.remove(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH,
                               "{}_{}{}".format(LINEAR_STRUCTURE_FOLDER_NAME, PARAMETER_FILE_NAME_PATTERN,
                                                METADATA_FILE_NAME_EXTENSION)))

        # Restore non-existing handler.
        handler_from_non_existing_folder = Linear(source.RETRIEVAL_TESTING_BASE_PATH)

        self.assertEqual(handler_from_non_existing_folder.dimension, 0)

        self._add_second_batch_of_data_points(handler_from_non_existing_folder)

        self.assertGreater(handler_from_non_existing_folder.dimension, 0)

        self.assertIsNone(handler_from_non_existing_folder.data_point_storage)
        self.assertListEqual(list(sorted(handler_from_non_existing_folder.file_path_storage)),
                             list(sorted(source.FILE_PATHS_SECOND_BATCH)))
        self.assertListEqual(list(sorted(handler_from_non_existing_folder.file_name_cache)),
                             list(sorted(source.FILE_NAMES_SECOND_BATCH)))

        self._query_second_batch_of_data_points(handler_from_non_existing_folder)

        shutil.rmtree(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH, LINEAR_STRUCTURE_FOLDER_NAME))
        os.remove(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH,
                               "{}_{}{}".format(LINEAR_STRUCTURE_FOLDER_NAME, PARAMETER_FILE_NAME_PATTERN,
                                                METADATA_FILE_NAME_EXTENSION)))

        # Restore handler in data folder.
        handler_in_data_folder = Linear(source.RETRIEVAL_DATA_BASE_PATH)

        self.assertGreater(handler_in_data_folder.dimension, 0)

        self.assertIsNone(handler_from_existing_folder.data_point_storage)
        self.assertListEqual(list(sorted(handler_in_data_folder.file_path_storage)),
                             list(sorted(source.FILE_PATHS_FIRST_BATCH +
                                         source.FILE_PATHS_SECOND_BATCH)))
        self.assertListEqual(list(sorted(handler_in_data_folder.file_name_cache)),
                             list(sorted(source.FILE_NAMES_FIRST_BATCH +
                                         source.FILE_NAMES_SECOND_BATCH)))

        self._query_second_batch_of_data_points(handler_from_existing_folder)

        shutil.rmtree(os.path.join(source.RETRIEVAL_DATA_BASE_PATH, LINEAR_STRUCTURE_FOLDER_NAME))
        os.remove(os.path.join(source.RETRIEVAL_DATA_BASE_PATH,
                               "{}_{}{}".format(LINEAR_STRUCTURE_FOLDER_NAME, PARAMETER_FILE_NAME_PATTERN,
                                                METADATA_FILE_NAME_EXTENSION)))

    def _test_number_of_hash_table_entries_in_memory(self, handler, num_points):
        for hash_table in handler.hash_tables:
            counter = 0
            for bucket in hash_table.values():
                counter += len(bucket)
            self.assertEqual(counter, num_points)

    def _test_lsh_retrieval_in_memory(self, use_pca):
        # Create in-memory handler.
        handler = Lsh(num_hash_functions=2,
                      num_hash_tables=10,
                      bucket_width=5,
                      use_pca=use_pca)

        self.assertEqual(handler.dimension, 0)

        self._add_first_batch_of_data_points(handler)

        self.assertGreater(handler.dimension, 0)

        self.assertEqual(len(handler.projection_vectors), 10)
        self.assertEqual(len(handler.hash_tables), 10)
        if use_pca:
            self.assertEqual(len(handler.projection_vectors[0]), 2)
        else:
            self.assertEqual(len(handler.projection_vectors[0]), 20)

        self._test_number_of_hash_table_entries_in_memory(handler, source.NUM_DATA_POINTS_FIRST_BATCH)

        self._query_first_batch_of_data_points(handler)

    def test_lsh_retrieval_in_memory(self):
        self._test_lsh_retrieval_in_memory(use_pca=False)
        self._test_lsh_retrieval_in_memory(use_pca=True)

    def _test_lsh_retrieval_out_of_core(self, use_pca):
        # Create out-of-core handler.
        handler = Lsh(source.RETRIEVAL_TESTING_BASE_PATH,
                      num_hash_functions=2,
                      num_hash_tables=10,
                      bucket_width=2,
                      use_pca=use_pca)

        self.assertEqual(handler.dimension, 0)

        self._add_first_batch_of_data_points(handler)

        self.assertGreater(handler.dimension, 0)

        self.assertEqual(len(handler.projection_vectors), 10)
        self.assertEqual(len(handler.projection_vectors[0]), 2)
        self.assertIsNone(handler.hash_tables)

        self._query_first_batch_of_data_points(handler)

        # Restore existing handler.
        handler_from_existing_folder = Lsh(source.RETRIEVAL_TESTING_BASE_PATH)

        self.assertGreater(handler_from_existing_folder.dimension, 0)

        self.assertEqual(len(handler.projection_vectors), 10)
        self.assertEqual(len(handler.projection_vectors[0]), 2)
        self.assertIsNone(handler.hash_tables)

        self._query_first_batch_of_data_points(handler_from_existing_folder)

        self._add_second_batch_of_data_points(handler_from_existing_folder)

        self._query_second_batch_of_data_points(handler_from_existing_folder)

        shutil.rmtree(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH, LSH_STRUCTURE_FOLDER_NAME))
        os.remove(os.path.join(source.RETRIEVAL_TESTING_BASE_PATH,
                               "{}_{}{}".format(LSH_STRUCTURE_FOLDER_NAME, PARAMETER_FILE_NAME_PATTERN,
                                                METADATA_FILE_NAME_EXTENSION)))

        # Restore handler in data folder,
        handler_in_data_folder = Lsh(source.RETRIEVAL_DATA_BASE_PATH,
                                     num_hash_functions=2,
                                     num_hash_tables=10,
                                     bucket_width=5,
                                     use_pca=use_pca)

        self.assertGreater(handler_in_data_folder.dimension, 0)

        self._query_second_batch_of_data_points(handler_in_data_folder)

        shutil.rmtree(os.path.join(source.RETRIEVAL_DATA_BASE_PATH, LSH_STRUCTURE_FOLDER_NAME))
        os.remove(os.path.join(source.RETRIEVAL_DATA_BASE_PATH,
                               "{}_{}{}".format(LSH_STRUCTURE_FOLDER_NAME, PARAMETER_FILE_NAME_PATTERN,
                                                METADATA_FILE_NAME_EXTENSION)))

    def test_lsh_retrieval_out_of_core(self):
        self._test_lsh_retrieval_out_of_core(use_pca=False)
        self._test_lsh_retrieval_out_of_core(use_pca=True)

    @classmethod
    def tearDownClass(cls):
        # Delete data folders.
        shutil.rmtree(source.RETRIEVAL_TESTING_BASE_PATH)
        shutil.rmtree(source.RETRIEVAL_DATA_BASE_PATH)
