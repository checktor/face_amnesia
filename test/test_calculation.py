# coding=utf-8
#
# Unit test for mathematical module.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import unittest

import numpy

from face_amnesia.utils import calculation
from test import source


class TestCalculation(unittest.TestCase):

    def test_get_norm(self):
        # Test numpy arrays.
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_NUMPY, order=1),
                               source.FIRST_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_NUMPY, order=1),
                               source.SECOND_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_NUMPY, order=1), source.VECTOR_3D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_NUMPY, order=2),
                               source.FIRST_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_NUMPY, order=2),
                               source.SECOND_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_NUMPY, order=2), source.VECTOR_3D_2_NORM)

        # Test lists.
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_LIST, order=1),
                               source.FIRST_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_LIST, order=1),
                               source.SECOND_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_LIST, order=1), source.VECTOR_3D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_LIST, order=2),
                               source.FIRST_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_LIST, order=2),
                               source.SECOND_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_LIST, order=2), source.VECTOR_3D_2_NORM)

        # Test tuples.
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_TUPLE, order=1),
                               source.FIRST_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_TUPLE, order=1),
                               source.SECOND_VECTOR_2D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_TUPLE, order=1), source.VECTOR_3D_1_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.FIRST_VECTOR_2D_TUPLE, order=2),
                               source.FIRST_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.SECOND_VECTOR_2D_TUPLE, order=2),
                               source.SECOND_VECTOR_2D_2_NORM)
        self.assertAlmostEqual(calculation.get_norm(source.VECTOR_3D_TUPLE, order=2), source.VECTOR_3D_2_NORM)

        # Test matrices.
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_NUMPY, order=1), source.MATRIX_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_LIST, order=1), source.MATRIX_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_TUPLE, order=1), source.MATRIX_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_LIST_OF_NUMPYS, order=1), source.MATRIX_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_NUMPY, order=2), source.MATRIX_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_LIST, order=2), source.MATRIX_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_TUPLE, order=2), source.MATRIX_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_LIST_OF_NUMPYS, order=2), source.MATRIX_2_NORM)

        # Test matrices with only one row vector.
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_NUMPY, order=1),
                                      source.MATRIX_ONE_ENTRY_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_LIST, order=1),
                                      source.MATRIX_ONE_ENTRY_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_TUPLE, order=1),
                                      source.MATRIX_ONE_ENTRY_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, order=1),
                                      source.MATRIX_ONE_ENTRY_1_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_NUMPY, order=2),
                                      source.MATRIX_ONE_ENTRY_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_LIST, order=2),
                                      source.MATRIX_ONE_ENTRY_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_TUPLE, order=2),
                                      source.MATRIX_ONE_ENTRY_2_NORM)
        numpy.testing.assert_allclose(calculation.get_norm(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, order=2),
                                      source.MATRIX_ONE_ENTRY_2_NORM)

        # Test invalid norms.
        self.assertRaises(ValueError, calculation.get_norm, source.FIRST_VECTOR_2D_NUMPY, 0)
        self.assertRaises(ValueError, calculation.get_norm, source.FIRST_VECTOR_2D_NUMPY, -1)

        # Test invalid vectors and matrices.
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_VECTOR_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_VECTOR_LIST, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_VECTOR_TUPLE, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_MATRIX_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_MATRIX_LIST, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.INVALID_MATRIX_TUPLE, 2)

        # Test empty data.
        self.assertRaises(ValueError, calculation.get_norm, numpy.empty(0), 2)
        self.assertRaises(ValueError, calculation.get_norm, [], 2)
        self.assertRaises(ValueError, calculation.get_norm, (), 2)

        # Test other data types and None.
        self.assertRaises(TypeError, calculation.get_norm, source.CUSTOM_DICT, 2)
        self.assertRaises(ValueError, calculation.get_norm, source.CUSTOM_STR, 2)
        self.assertRaises(ValueError, calculation.get_norm, None, 2)

    def test_get_distance(self):
        # Test numpy arrays.
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_NUMPY, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_NUMPY, source.FIRST_VECTOR_2D_NUMPY, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_NUMPY, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_NUMPY, source.FIRST_VECTOR_2D_NUMPY, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)

        # Test lists.
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_LIST, source.FIRST_VECTOR_2D_LIST, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_LIST, source.FIRST_VECTOR_2D_LIST, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_LIST, source.FIRST_VECTOR_2D_LIST, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_LIST, source.FIRST_VECTOR_2D_LIST, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)

        # Test tuples.
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_TUPLE, source.FIRST_VECTOR_2D_TUPLE, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_TUPLE, source.FIRST_VECTOR_2D_TUPLE, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_TUPLE, source.FIRST_VECTOR_2D_TUPLE, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.SECOND_VECTOR_2D_TUPLE, source.FIRST_VECTOR_2D_TUPLE, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)

        # Mix numpy arrays, lists and tuples.
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_LIST, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_TUPLE, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_LIST, source.SECOND_VECTOR_2D_NUMPY, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_LIST, source.SECOND_VECTOR_2D_TUPLE, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_TUPLE, source.SECOND_VECTOR_2D_NUMPY, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_TUPLE, source.SECOND_VECTOR_2D_LIST, 1),
                               source.DISTANCE_1_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_LIST, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_NUMPY, source.SECOND_VECTOR_2D_TUPLE, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_LIST, source.SECOND_VECTOR_2D_NUMPY, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_LIST, source.SECOND_VECTOR_2D_TUPLE, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_TUPLE, source.SECOND_VECTOR_2D_NUMPY, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)
        self.assertAlmostEqual(calculation.get_distance(source.FIRST_VECTOR_2D_TUPLE, source.SECOND_VECTOR_2D_LIST, 2),
                               source.DISTANCE_2_NORM_FIRST_SECOND)

        # Test invalid norms.
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.SECOND_VECTOR_2D_NUMPY, 0)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.SECOND_VECTOR_2D_NUMPY, -1)

        # Test vectors of different size.
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.VECTOR_3D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.VECTOR_3D_NUMPY, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_LIST, source.VECTOR_3D_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.VECTOR_3D_LIST, source.FIRST_VECTOR_2D_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_TUPLE, source.VECTOR_3D_TUPLE, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.VECTOR_3D_TUPLE, source.FIRST_VECTOR_2D_TUPLE, 2)

        # Test invalid matrices and vectors.
        self.assertRaises(TypeError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_VECTOR_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_VECTOR_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_VECTOR_TUPLE, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_MATRIX_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_MATRIX_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.INVALID_MATRIX_TUPLE, 2)
        self.assertRaises(TypeError, calculation.get_distance, source.INVALID_VECTOR_NUMPY,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.INVALID_VECTOR_LIST,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.INVALID_VECTOR_TUPLE,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.INVALID_MATRIX_NUMPY,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.INVALID_MATRIX_LIST,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.INVALID_MATRIX_TUPLE,
                          source.FIRST_VECTOR_2D_NUMPY, 2)

        # Test empty data.
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, numpy.empty(0), 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, [], 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, (), 2)
        self.assertRaises(ValueError, calculation.get_distance, numpy.empty(0), source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, [], source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, (), source.FIRST_VECTOR_2D_NUMPY, 2)

        # Test matrices.
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_NUMPY, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_LIST, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_TUPLE, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_LIST_OF_NUMPYS,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.MATRIX_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.MATRIX_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.MATRIX_TUPLE, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.MATRIX_LIST_OF_NUMPYS, 2)

        # Test matrices with only one row vector.
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_ONE_ENTRY_NUMPY,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_ONE_ENTRY_LIST,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_ONE_ENTRY_TUPLE,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS,
                          source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.MATRIX_ONE_ENTRY_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.MATRIX_ONE_ENTRY_LIST, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.MATRIX_ONE_ENTRY_TUPLE, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY,
                          source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, 2)

        # Test other data types and None.
        self.assertRaises(TypeError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.CUSTOM_DICT, 2)
        self.assertRaises(TypeError, calculation.get_distance, source.CUSTOM_DICT, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, source.CUSTOM_STR, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.CUSTOM_STR, source.FIRST_VECTOR_2D_NUMPY, 2)
        self.assertRaises(ValueError, calculation.get_distance, source.FIRST_VECTOR_2D_NUMPY, None, 2)
        self.assertRaises(ValueError, calculation.get_distance, None, source.FIRST_VECTOR_2D_NUMPY, 2)

    def test_get_mean(self):
        # Test matrices.
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_NUMPY), source.MATRIX_MEAN_VECTOR)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_LIST), source.MATRIX_MEAN_VECTOR)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_TUPLE), source.MATRIX_MEAN_VECTOR)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_LIST_OF_NUMPYS), source.MATRIX_MEAN_VECTOR)

        # Test matrices with only one row vector.
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_ONE_ENTRY_NUMPY), source.FIRST_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_ONE_ENTRY_LIST), source.FIRST_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_ONE_ENTRY_TUPLE), source.FIRST_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(calculation.get_mean(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS),
                                      source.FIRST_VECTOR_2D_NUMPY)

        # Test NumPy arrays.
        res = calculation.get_mean(source.FIRST_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(res, source.FIRST_VECTOR_2D_NUMPY)
        res = calculation.get_mean(source.SECOND_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(res, source.SECOND_VECTOR_2D_NUMPY)
        res = calculation.get_mean(source.VECTOR_3D_NUMPY)
        numpy.testing.assert_allclose(res, source.VECTOR_3D_NUMPY)

        # Test lists.
        res = calculation.get_mean(source.FIRST_VECTOR_2D_LIST)
        numpy.testing.assert_allclose(res, source.FIRST_VECTOR_2D_NUMPY)
        res = calculation.get_mean(source.SECOND_VECTOR_2D_LIST)
        numpy.testing.assert_allclose(res, source.SECOND_VECTOR_2D_NUMPY)
        res = calculation.get_mean(source.VECTOR_3D_LIST)
        numpy.testing.assert_allclose(res, source.VECTOR_3D_NUMPY)

        # Test tuples.
        numpy.testing.assert_allclose(calculation.get_mean(source.FIRST_VECTOR_2D_TUPLE), source.FIRST_VECTOR_2D_NUMPY)
        numpy.testing.assert_allclose(calculation.get_mean(source.SECOND_VECTOR_2D_TUPLE),
                                      source.SECOND_VECTOR_2D_NUMPY)

        # Test empty data.
        self.assertRaises(ValueError, calculation.get_mean, numpy.empty(0))
        self.assertRaises(ValueError, calculation.get_mean, [])
        self.assertRaises(ValueError, calculation.get_mean, ())

        # Test invalid vectors and matrices.
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_VECTOR_NUMPY)
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_VECTOR_LIST)
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_VECTOR_TUPLE)
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_MATRIX_NUMPY)
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_MATRIX_LIST)
        self.assertRaises(ValueError, calculation.get_mean, source.INVALID_MATRIX_TUPLE)

        # Test other data types and None.
        self.assertRaises(TypeError, calculation.get_mean, source.CUSTOM_DICT)
        self.assertRaises(ValueError, calculation.get_mean, source.CUSTOM_STR)
        self.assertRaises(ValueError, calculation.get_mean, None)

    def test_get_random_vectors(self):
        num_vectors_list = [-1, 0, 1, 5, 128, 500]
        dimension_list = [-1, 0, 1, 5, 128, 500]
        for i in num_vectors_list:
            for j in dimension_list:
                if i <= 0 or j <= 0:
                    self.assertRaises(ValueError, calculation.get_random_vectors, i, j, False)
                else:
                    res = calculation.get_random_vectors(i, j, True)
                    self.assertIsInstance(res, numpy.ndarray)
                    self.assertTrue(res.size > 0)
                    self.assertTupleEqual((i, j), res.shape)
                    res = calculation.get_random_vectors(i, j, False)
                    self.assertIsInstance(res, numpy.ndarray)
                    self.assertTrue(res.size > 0)
                    self.assertTupleEqual((i, j), res.shape)
