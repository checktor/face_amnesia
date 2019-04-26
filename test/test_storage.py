# coding=utf-8
#
# Unit test for storage module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.


import os
import unittest

import numpy

from face_amnesia.base import storage
from face_amnesia.base.data_set import DataSet
from test import helper, source


class TestStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create non-readable file.
        helper.create_non_readable_file(os.path.join(source.BASE_PATH, source.TXT_TEXT),
                                        os.path.join(source.BASE_PATH, source.NON_READABLE_FILE))
        # Create non-writable file.
        helper.create_non_writable_file(os.path.join(source.BASE_PATH, source.TXT_TEXT),
                                        os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + ".dat"))

    def test_read_and_write_data_structure(self):
        # Define test file path.
        test_file_path = os.path.join(
            source.BASE_PATH, "data_structure_test.txt")

        # Write None value and read it again.
        res = storage.write_data_structure_to_file(test_file_path, None)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        self.assertIsNone(dat)

        # Write integer and read it again.
        res = storage.write_data_structure_to_file(test_file_path, 5)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        self.assertEqual(dat, 5)

        # Write string and read it again.
        res = storage.write_data_structure_to_file(
            test_file_path, source.CUSTOM_STR)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        self.assertEqual(dat, source.CUSTOM_STR)

        # Write dictionary and read it again.
        res = storage.write_data_structure_to_file(
            test_file_path, source.CUSTOM_DICT)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        self.assertEqual(dat, source.CUSTOM_DICT)

        # Write numpy matrix and read it again.
        res = storage.write_data_structure_to_file(
            test_file_path, source.MATRIX_LIST_OF_NUMPYS)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        numpy.testing.assert_allclose(dat, source.MATRIX_LIST_OF_NUMPYS)

        # Write list of multiple data structures and read it again.
        data_list = [None, source.CUSTOM_STR, source.CUSTOM_DICT]
        res = storage.write_data_structure_to_file(test_file_path, data_list)
        self.assertTrue(res)
        dat = storage.read_data_structure_from_file(test_file_path)
        self.assertListEqual(dat, data_list)

        # Read non-existing file.
        res = storage.read_data_structure_from_file(
            os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE))
        self.assertIsNone(res)

        # Read non-readable file.
        res = storage.read_data_structure_from_file(
            os.path.join(source.BASE_PATH, source.NON_READABLE_FILE))
        self.assertIsNone(res)

        # Read directory.
        res = storage.read_data_structure_from_file(source.BASE_PATH)
        self.assertIsNone(res)

        # Write non-writable file.
        if os.geteuid() == 0:
            # Root is always able to read and write files.
            res = storage.write_data_structure_to_file(
                os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + ".dat"),
                source.CUSTOM_DICT)
            self.assertTrue(res)
            dat = storage.read_data_structure_from_file(
                os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + ".dat"))
            self.assertEqual(dat, source.CUSTOM_DICT)
        else:
            res = storage.write_data_structure_to_file(
                os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + ".dat"),
                source.CUSTOM_DICT)
            self.assertFalse(res)

        # Write directory.
        res = storage.write_data_structure_to_file(source.BASE_PATH,
                                                   source.CUSTOM_DICT)
        self.assertFalse(res)

        # Read invalid file paths.
        res = storage.read_data_structure_from_file(source.CUSTOM_DICT)
        self.assertIsNone(res)
        res = storage.read_data_structure_from_file(source.CUSTOM_STR)
        self.assertIsNone(res)
        res = storage.read_data_structure_from_file("")
        self.assertIsNone(res)
        res = storage.read_data_structure_from_file(None)
        self.assertIsNone(res)

        # Write invalid file paths.
        res = storage.write_data_structure_to_file(
            source.CUSTOM_DICT, data_list)
        self.assertFalse(res)
        res = storage.write_data_structure_to_file("", data_list)
        self.assertFalse(res)
        res = storage.write_data_structure_to_file(None, data_list)
        self.assertFalse(res)

        # Delete test file.
        os.remove(test_file_path)

    def test_read_and_write_data_set(self):
        # Define test file name.
        test_file_path = os.path.join(source.BASE_PATH, "data_set_test.txt")

        # Write DataSet instance and read it again.
        res = storage.write_data_set(test_file_path, DataSet(
            source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        self.assertTrue(res)
        data_set = storage.read_data_set(test_file_path)
        numpy.testing.assert_allclose(data_set.data_points, source.MATRIX_LIST_OF_NUMPYS)
        self.assertListEqual(data_set.metadata, source.MATRIX_METADATA)

        # Write DataSet instance with only one entry and read it again.
        res = storage.write_data_set(test_file_path,
                                     DataSet(source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS, source.MATRIX_ONE_ENTRY_METADATA))
        self.assertTrue(res)
        data_set = storage.read_data_set(test_file_path)
        numpy.testing.assert_allclose(
            data_set.data_points, source.MATRIX_ONE_ENTRY_LIST_OF_NUMPYS)
        self.assertListEqual(
            data_set.metadata, source.MATRIX_ONE_ENTRY_METADATA)

        # Write empty DataSet and read it again.
        res = storage.write_data_set(test_file_path, DataSet([], []))
        self.assertTrue(res)
        data_set = storage.read_data_set(test_file_path)
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])

        # Read non-existing file.
        data_set = storage.read_data_set(os.path.join(
            source.BASE_PATH, source.NON_EXISTING_FILE))
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])

        # Read non-readable file.
        data_set = storage.read_data_set(os.path.join(
            source.BASE_PATH, source.NON_READABLE_FILE))
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])

        # Read directory.
        data_set = storage.read_data_set(source.BASE_PATH)
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])

        # Write non-writable file.
        if os.geteuid() == 0:
            # Root is always able to read and write data sets.
            res = storage.write_data_set(os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE),
                                         DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
            self.assertTrue(res)
            data_set = storage.read_data_set(os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE))
            numpy.testing.assert_allclose(data_set.data_points, source.MATRIX_LIST_OF_NUMPYS)
            self.assertListEqual(data_set.metadata, source.MATRIX_METADATA)
        else:
            res = storage.write_data_set(os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE),
                                         DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
            self.assertFalse(res)

        # Write directory.
        res = storage.write_data_set(source.BASE_PATH,
                                     DataSet(source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        self.assertFalse(res)

        # Write None value.
        res = storage.write_data_set(test_file_path, None)
        self.assertFalse(res)

        # Read invalid file paths.
        data_set = storage.read_data_set(source.CUSTOM_DICT)
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])
        data_set = storage.read_data_set(source.CUSTOM_STR)
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])
        data_set = storage.read_data_set("")
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])
        data_set = storage.read_data_set(None)
        numpy.testing.assert_allclose(data_set.data_points, numpy.empty(0))
        self.assertListEqual(data_set.metadata, [])

        # Write invalid file paths.
        res = storage.write_data_set(source.CUSTOM_DICT, DataSet(
            source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        self.assertFalse(res)
        res = storage.write_data_set("", DataSet(
            source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        self.assertFalse(res)
        res = storage.write_data_set(None, DataSet(
            source.MATRIX_LIST_OF_NUMPYS, source.MATRIX_METADATA))
        self.assertFalse(res)

        # Delete test file.
        os.remove(test_file_path + storage.DATA_POINT_FILE_NAME_EXTENSION)
        os.remove(test_file_path + storage.METADATA_FILE_NAME_EXTENSION)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(source.BASE_PATH, source.NON_READABLE_FILE))
        os.remove(os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + ".dat"))
        os.remove(os.path.join(source.BASE_PATH, source.NON_WRITABLE_FILE + storage.DATA_POINT_FILE_NAME_EXTENSION))
