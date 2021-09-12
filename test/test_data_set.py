# coding=utf-8
#
# Unit test for DataSet class.
#
# Copyright: C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import copy
import unittest

import numpy

from face_amnesia.base.data_set import DataSet
from test import source


class TestDataSet(unittest.TestCase):

    def _test_init_valid(self, new_data_points, new_metadata):
        # Test with metadata.
        data_set = DataSet(new_data_points, new_metadata)
        self.assertIsInstance(data_set.data_points, list)
        self.assertIsInstance(data_set.metadata, list)
        numpy.testing.assert_allclose(data_set.data_points, new_data_points)

        # Test without metadata.
        data_set = DataSet(new_data_points, [])
        self.assertIsInstance(data_set.data_points, list)
        self.assertIsInstance(data_set.metadata, list)
        self.assertListEqual(data_set.metadata, [set() for _i in range(len(new_data_points))])

    def test_init(self):
        # Test valid matrices (with and without metadata).
        self._test_init_valid(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA)

        self._test_init_valid(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                              source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL)

        self._test_init_valid(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS,
                              source.MATRIX_ONE_ENTRY_METADATA)

        self._test_init_valid(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS,
                              source.MATRIX_ONE_ENTRY_METADATA)

        self._test_init_valid([], [])

        # Test invalid matrices and metadata.
        self.assertRaises(ValueError, DataSet, source.MATRIX_NUMPY, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_LIST, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_TUPLE, source.MATRIX_METADATA)

        self.assertRaises(ValueError, DataSet, source.MATRIX_WITH_DUPLICATES_NUMPY,
                          source.MATRIX_WITH_DUPLICATES_METADATA_EQUAL)

        self.assertRaises(ValueError, DataSet, source.MATRIX_ONE_ENTRY_NUMPY, source.MATRIX_ONE_ENTRY_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_ONE_ENTRY_LIST, source.MATRIX_ONE_ENTRY_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_ONE_ENTRY_TUPLE, source.MATRIX_ONE_ENTRY_METADATA)

        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_NUMPY, source.MATRIX_METADATA)

        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_ROW_SIZE_LIST_OF_NUMPYS, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_ROW_SIZE_NUMPY, source.MATRIX_METADATA)

        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_SHAPE_LIST_OF_NUMPYS, source.MATRIX_METADATA)

        self.assertRaises(ValueError, DataSet, source.INVALID_MATRIX_DTYPE_LIST_OF_NUMPYS, source.MATRIX_METADATA)

        self.assertRaises(ValueError, DataSet, numpy.empty(0), [])
        self.assertRaises(ValueError, DataSet, (), [])

        # Test non-matching matrices and metadata.
        self.assertRaises(ValueError, DataSet, [], source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA)

        # Test 1D vectors.
        self.assertRaises(ValueError, DataSet, source.FIRST_VECTOR_2D_NUMPY, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, DataSet, source.FIRST_VECTOR_2D_LIST, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, DataSet, source.FIRST_VECTOR_2D_TUPLE, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, DataSet, source.INVALID_VECTOR_NUMPY, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, DataSet, source.INVALID_VECTOR_LIST, source.FIRST_VECTOR_2D_METADATA)

        # Test 3D matrices.
        self.assertRaises(ValueError, DataSet, source.CUSTOM_IMG_BGR_NUMPY, [])
        self.assertRaises(ValueError, DataSet, source.CUSTOM_IMG_BGR_LIST, [])
        self.assertRaises(ValueError, DataSet, source.CUSTOM_IMG_BGR_TUPLE, [])

        # Test other data types and None values.
        self.assertRaises(ValueError, DataSet, source.CUSTOM_DICT, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.CUSTOM_STR, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, None, source.MATRIX_METADATA)
        self.assertRaises(ValueError, DataSet, source.MATRIX_LIST_OF_NUMPYS, source.CUSTOM_DICT)
        self.assertRaises(ValueError, DataSet, source.MATRIX_LIST_OF_NUMPYS, source.CUSTOM_STR)
        self.assertRaises(ValueError, DataSet, source.MATRIX_LIST_OF_NUMPYS, None)

    def _test_add_data_point_invalid(self, data_set: DataSet):
        # Valid 2D matrices.
        self.assertRaises(ValueError, data_set.add_data_point, source.MATRIX_LIST_OF_NUMPYS,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.MATRIX_NUMPY,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.MATRIX_LIST,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.MATRIX_TUPLE,
                          source.MATRIX_METADATA)

        # Valid 3D matrices.
        self.assertRaises(ValueError, data_set.add_data_point, source.CUSTOM_IMG_BGR_NUMPY, [])
        self.assertRaises(ValueError, data_set.add_data_point, source.CUSTOM_IMG_BGR_LIST, [])
        self.assertRaises(ValueError, data_set.add_data_point, source.CUSTOM_IMG_BGR_TUPLE, [])

        # Empty data.
        self.assertRaises(ValueError, data_set.add_data_point, numpy.empty(0), set())
        self.assertRaises(ValueError, data_set.add_data_point, [], set())

        # Invalid matrices.
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_MATRIX_LIST_OF_NUMPYS,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_MATRIX_NUMPY,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_MATRIX_SHAPE_LIST_OF_NUMPYS,
                          source.MATRIX_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_MATRIX_DTYPE_LIST_OF_NUMPYS,
                          source.MATRIX_METADATA)

        # Invalid vectors.
        self.assertRaises(ValueError, data_set.add_data_point, source.FIRST_VECTOR_2D_LIST,
                          source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.FIRST_VECTOR_2D_TUPLE,
                          source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_VECTOR_NUMPY,
                          source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.INVALID_VECTOR_LIST,
                          source.FIRST_VECTOR_2D_METADATA)

        # Invalid data types.
        self.assertRaises(ValueError, data_set.add_data_point, source.CUSTOM_DICT, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.CUSTOM_STR, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, None, source.FIRST_VECTOR_2D_METADATA)
        self.assertRaises(ValueError, data_set.add_data_point, source.FIRST_VECTOR_2D_NUMPY, source.CUSTOM_DICT)
        self.assertRaises(ValueError, data_set.add_data_point, source.FIRST_VECTOR_2D_NUMPY, source.CUSTOM_STR)
        self.assertRaises(ValueError, data_set.add_data_point, source.FIRST_VECTOR_2D_NUMPY, None)

    def test_add_data_point(self):
        # Create empty data set.
        data_set_2d = DataSet([], [])

        # Test invalid data on empty data set.
        self._test_add_data_point_invalid(data_set_2d)

        # Add 2D data points to initially empty DataSet.
        data_set_2d.add_data_point(source.FIRST_VECTOR_2D_NUMPY, source.FIRST_VECTOR_2D_METADATA)
        data_set_2d.add_data_point(source.FIRST_VECTOR_2D_REVERSED_NUMPY, source.FIRST_VECTOR_2D_REVERSED_METADATA)
        data_set_2d.add_data_point(source.SECOND_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_METADATA)
        data_set_2d.add_data_point(source.SECOND_VECTOR_2D_REVERSED_NUMPY, source.SECOND_VECTOR_2D_REVERSED_METADATA)
        self.assertIsInstance(data_set_2d.data_points, list)
        self.assertIsInstance(data_set_2d.metadata, list)
        numpy.testing.assert_allclose(data_set_2d.data_points, source.MATRIX_LIST_OF_NUMPYS)
        self.assertListEqual(data_set_2d.metadata, source.MATRIX_METADATA)

        # Test invalid data on non-empty data set.
        self._test_add_data_point_invalid(data_set_2d)

        # Add data point of wrong shape.
        self.assertRaises(ValueError, data_set_2d.add_data_point, source.VECTOR_3D_NUMPY, source.VECTOR_3D_METADATA)

        # Add the same 3D data point multiple
        # times to initially empty data set.
        data_set_3d = DataSet([], [])
        data_set_3d.add_data_point(source.VECTOR_3D_NUMPY, source.VECTOR_3D_METADATA)
        data_set_3d.add_data_point(source.VECTOR_3D_NUMPY, source.VECTOR_3D_METADATA)
        data_set_3d.add_data_point(source.VECTOR_3D_NUMPY, source.VECTOR_3D_METADATA)
        self.assertIsInstance(data_set_3d.data_points, list)
        self.assertIsInstance(data_set_3d.metadata, list)
        result_matrix = [source.VECTOR_3D_NUMPY, source.VECTOR_3D_NUMPY, source.VECTOR_3D_NUMPY]
        result_metadata = [source.VECTOR_3D_METADATA, source.VECTOR_3D_METADATA, source.VECTOR_3D_METADATA]
        numpy.testing.assert_allclose(data_set_3d.data_points, result_matrix)
        self.assertListEqual(data_set_3d.metadata, result_metadata)

    def _test_add_data_set_invalid(self, data_set: DataSet):
        # Store old data set.
        old_data_points = numpy.copy(data_set.data_points)
        old_metadata = copy.deepcopy(data_set.metadata)

        # Test empty matrices.
        data_set.add_data_set(DataSet([], []))
        self.assertIsInstance(data_set.data_points, list)
        self.assertIsInstance(data_set.metadata, list)
        numpy.testing.assert_allclose(data_set.data_points, old_data_points)
        self.assertListEqual(data_set.metadata, old_metadata)

        # Test other data types.
        self.assertRaises(ValueError, data_set.add_data_set, source.CUSTOM_DICT)
        self.assertRaises(ValueError, data_set.add_data_set, source.CUSTOM_STR)
        self.assertRaises(ValueError, data_set.add_data_set, None)

    def test_add_data_set(self):
        # Create empty data set.
        data_set = DataSet([], [])

        # Test invalid data on empty data set.
        self._test_add_data_set_invalid(data_set)

        # Add data sets.
        data_set.add_data_set(DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        data_set.add_data_set(
            DataSet(source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS,
                    source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL))
        self.assertIsInstance(data_set.data_points, list)
        self.assertIsInstance(data_set.metadata, list)
        result_matrix = source.MATRIX_LIST_OF_NUMPYS + source.MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS
        result_metadata = source.MATRIX_METADATA + source.MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL
        numpy.testing.assert_allclose(data_set.data_points, result_matrix)
        self.assertListEqual(data_set.metadata, result_metadata)

        # Test invalid data on non-empty data set.
        self._test_add_data_set_invalid(data_set)

        # Add data set of wrong shape.
        new_data_set = DataSet([source.VECTOR_3D_NUMPY], [source.VECTOR_3D_METADATA])
        self.assertRaises(ValueError, data_set.add_data_set, new_data_set)
