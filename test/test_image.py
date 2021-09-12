# coding=utf-8
#
# Unit test for image module.
#
# Copyright: C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
import unittest

import numpy

from face_amnesia.media import image
from test import helper, source


class TestImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_non_readable_file(os.path.join(source.IMG_BASE_PATH, source.JPG_IMG),
                                        os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG))

    def test_is_valid_image_file(self):
        # Test image data.
        self.assertTrue(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.BMP_IMG)))
        self.assertTrue(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.JPG_IMG)))
        self.assertTrue(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.PNG_IMG)))
        self.assertTrue(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG)))

        # Test video data.
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO)))

        # Test text data.
        self.assertFalse(image.is_valid_image_file(os.path.join(source.BASE_PATH, source.TXT_TEXT)))
        self.assertFalse(image.is_valid_image_file(os.path.join(source.BASE_PATH, source.MD_TEXT)))

        # Test non-existing file.
        self.assertFalse(image.is_valid_image_file(os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE)))

        # Test non-readable file.
        if os.geteuid() == 0:
            # Root is always able to read image data.
            self.assertTrue(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG)))
        else:
            self.assertFalse(image.is_valid_image_file(os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG)))

        # Test directory.
        self.assertFalse(image.is_valid_image_file(source.BASE_PATH))

        # Test other data types and None values.
        self.assertFalse(image.is_valid_image_file(source.CUSTOM_DICT))
        self.assertFalse(image.is_valid_image_file(source.CUSTOM_STR))
        self.assertFalse(image.is_valid_image_file(None))

    def test_is_valid_pixel_matrix(self):
        # Test valid pixel data.
        self.assertTrue(image.is_valid_pixel_matrix(source.CUSTOM_IMG_BGR_NUMPY))
        self.assertTrue(image.is_valid_pixel_matrix(source.CUSTOM_IMG_RGB_NUMPY))

        # Test invalid pixel data.
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_BGR_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_RGB_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_BGR_TUPLE))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_RGB_TUPLE))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_1D_STRUCTURE_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_1D_STRUCTURE_NUMPY))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_2D_STRUCTURE_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_2D_STRUCTURE_NUMPY))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_TWO_COLOR_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_TWO_COLOR_NUMPY))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_FOUR_COLOR_LIST))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_FOUR_COLOR_NUMPY))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_IMG_WRONG_DATA_TYPE_NUMPY))

        # Test empty data.
        self.assertFalse(image.is_valid_pixel_matrix(numpy.empty(0)))
        self.assertFalse(image.is_valid_pixel_matrix([]))
        self.assertFalse(image.is_valid_pixel_matrix(()))

        # Test other data types and None.
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_DICT))
        self.assertFalse(image.is_valid_pixel_matrix(source.CUSTOM_STR))
        self.assertFalse(image.is_valid_pixel_matrix(None))

    def test_read_image_from_file(self):
        # Test image data.
        res = image.read_image_from_file(os.path.join(source.IMG_BASE_PATH, source.BMP_IMG))
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual(res.shape[0], source.IMG_HEIGHT)
        self.assertEqual(res.shape[1], source.IMG_WIDTH)
        self.assertListEqual(list(res[0][0]), source.TOP_LEFT_BMP)
        res = image.read_image_from_file(os.path.join(source.IMG_BASE_PATH, source.JPG_IMG))
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual(res.shape[0], source.IMG_HEIGHT)
        self.assertEqual(res.shape[1], source.IMG_WIDTH)
        self.assertListEqual(list(res[0][0]), source.TOP_LEFT_JPG)
        res = image.read_image_from_file(os.path.join(source.IMG_BASE_PATH, source.PNG_IMG))
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual(res.shape[0], source.IMG_HEIGHT)
        self.assertEqual(res.shape[1], source.IMG_WIDTH)
        self.assertListEqual(list(res[0][0]), source.TOP_LEFT_PNG)
        res = image.read_image_from_file(os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG))
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual(res.shape[0], source.IMG_HEIGHT)
        self.assertEqual(res.shape[1], source.IMG_WIDTH)
        self.assertListEqual(list(res[0][0]), source.TOP_LEFT_TIFF)

        # Test video data.
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO))
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO))
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO))
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO))
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO))
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO))

        # Test text data.
        self.assertRaises(ValueError, image.read_image_from_file, os.path.join(source.BASE_PATH, source.TXT_TEXT))
        self.assertRaises(ValueError, image.read_image_from_file, os.path.join(source.BASE_PATH, source.MD_TEXT))

        # Test non-existing file.
        self.assertRaises(ValueError, image.read_image_from_file,
                          os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE))

        # Test non-readable file.
        if os.geteuid() == 0:
            # Root is always able to read image data.
            res = image.read_image_from_file(os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG))
            self.assertIsInstance(res, numpy.ndarray)
            self.assertEqual(res.shape[0], source.IMG_HEIGHT)
            self.assertEqual(res.shape[1], source.IMG_WIDTH)
            self.assertListEqual(list(res[0][0]), source.TOP_LEFT_JPG)
        else:
            self.assertRaises(ValueError, image.read_image_from_file,
                              os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG))

        # Test directory.
        self.assertRaises(ValueError, image.read_image_from_file, source.BASE_PATH)

        # Test empty stringsm other data types and None values.
        self.assertRaises(ValueError, image.read_image_from_file, "")
        self.assertRaises(ValueError, image.read_image_from_file, source.CUSTOM_DICT)
        self.assertRaises(ValueError, image.read_image_from_file, source.CUSTOM_STR)
        self.assertRaises(ValueError, image.read_image_from_file, None)

    def test_swap_color_encoding(self):
        # Test valid pixel data.
        try:
            # BGR => RGB
            res = image.swap_color_encoding(source.CUSTOM_IMG_BGR_NUMPY)
            self.assertIsInstance(res, numpy.ndarray)
            numpy.testing.assert_equal(res, source.CUSTOM_IMG_RGB_NUMPY)
            # RGB => BGR
            res = image.swap_color_encoding(source.CUSTOM_IMG_RGB_NUMPY)
            self.assertIsInstance(res, numpy.ndarray)
            numpy.testing.assert_equal(res, source.CUSTOM_IMG_BGR_NUMPY)
            # RGB => BGR => RGB
            res = image.swap_color_encoding(image.swap_color_encoding(source.CUSTOM_IMG_RGB_NUMPY))
            self.assertIsInstance(res, numpy.ndarray)
            numpy.testing.assert_equal(res, source.CUSTOM_IMG_RGB_NUMPY)
        except AssertionError:
            self.fail()

        # Test valid pixel data not stored as numpy array.
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_BGR_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_RGB_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_BGR_TUPLE)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_RGB_TUPLE)

        # Test incorrectly structured pixel data.
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_1D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_1D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_2D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_2D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_TWO_COLOR_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_TWO_COLOR_NUMPY)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_FOUR_COLOR_LIST)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_FOUR_COLOR_NUMPY)

        # Test pixel data with incorrect data type.
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_IMG_WRONG_DATA_TYPE_NUMPY)

        # Test empty data.
        self.assertRaises(ValueError, image.swap_color_encoding, numpy.empty(0))
        self.assertRaises(ValueError, image.swap_color_encoding, [])
        self.assertRaises(ValueError, image.swap_color_encoding, ())

        # Test other data types and None.
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_DICT)
        self.assertRaises(ValueError, image.swap_color_encoding, source.CUSTOM_STR)
        self.assertRaises(ValueError, image.swap_color_encoding, None)

    def test_get_image_height(self):
        # Test valid pixel data.
        self.assertEqual(image.get_image_height(source.CUSTOM_IMG_BGR_NUMPY), source.CUSTOM_IMG_HEIGHT)
        self.assertEqual(image.get_image_height(source.CUSTOM_IMG_RGB_NUMPY), source.CUSTOM_IMG_HEIGHT)

        # Test invalid pixel data.
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_BGR_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_RGB_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_BGR_TUPLE)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_RGB_TUPLE)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_1D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_1D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_2D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_2D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_TWO_COLOR_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_TWO_COLOR_NUMPY)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_FOUR_COLOR_LIST)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_FOUR_COLOR_NUMPY)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_IMG_WRONG_DATA_TYPE_NUMPY)

        # Test empty data.
        self.assertRaises(ValueError, image.get_image_height, numpy.empty(0))
        self.assertRaises(ValueError, image.get_image_height, [])
        self.assertRaises(ValueError, image.get_image_height, ())

        # Test other data types and None values.
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_DICT)
        self.assertRaises(ValueError, image.get_image_height, source.CUSTOM_STR)
        self.assertRaises(ValueError, image.get_image_height, None)

    def test_get_image_width(self):
        # Test valid pixel data.
        self.assertEqual(image.get_image_width(source.CUSTOM_IMG_BGR_NUMPY), source.CUSTOM_IMG_WIDTH)
        self.assertEqual(image.get_image_width(source.CUSTOM_IMG_RGB_NUMPY), source.CUSTOM_IMG_WIDTH)

        # Test invalid pixel data.
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_BGR_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_RGB_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_BGR_TUPLE)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_RGB_TUPLE)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_1D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_1D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_2D_STRUCTURE_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_2D_STRUCTURE_NUMPY)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_TWO_COLOR_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_TWO_COLOR_NUMPY)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_FOUR_COLOR_LIST)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_FOUR_COLOR_NUMPY)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_IMG_WRONG_DATA_TYPE_NUMPY)

        # Test empty data.
        self.assertRaises(ValueError, image.get_image_width, numpy.empty(0))
        self.assertRaises(ValueError, image.get_image_width, [])
        self.assertRaises(ValueError, image.get_image_width, ())

        # Test other data types and None values.
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_DICT)
        self.assertRaises(ValueError, image.get_image_width, source.CUSTOM_STR)
        self.assertRaises(ValueError, image.get_image_width, None)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(source.IMG_BASE_PATH, source.NON_READABLE_IMG))
