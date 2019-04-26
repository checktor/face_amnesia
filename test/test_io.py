# coding=utf-8
#
# Unit test for IO module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import os
import unittest

from face_amnesia.utils import io
from test import helper, source


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create non-readable text file.
        helper.create_non_readable_file(os.path.join(source.BASE_PATH, source.TXT_TEXT),
                                        os.path.join(source.BASE_PATH, source.NON_READABLE_FILE))

    def test_is_readable_file(self):
        # Test image data.
        self.assertTrue(io.is_readable_file(os.path.join(source.IMG_BASE_PATH, source.BMP_IMG)))
        self.assertTrue(io.is_readable_file(os.path.join(source.IMG_BASE_PATH, source.JPG_IMG)))
        self.assertTrue(io.is_readable_file(os.path.join(source.IMG_BASE_PATH, source.PNG_IMG)))
        self.assertTrue(io.is_readable_file(os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG)))

        # Test video data.
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO)))
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO)))
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO)))
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO)))
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO)))
        self.assertTrue(io.is_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO)))

        # Test text data.
        self.assertTrue(io.is_readable_file(os.path.join(source.BASE_PATH, source.TXT_TEXT)))
        self.assertTrue(io.is_readable_file(os.path.join(source.BASE_PATH, source.MD_TEXT)))

        # Test non-existing file.
        self.assertFalse(io.is_readable_file(os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE)))

        # Test non-readable data.
        if os.geteuid() == 0:
            # Root is always able to read data.
            self.assertTrue(io.is_readable_file(os.path.join(source.BASE_PATH, source.NON_READABLE_FILE)))
        else:
            self.assertFalse(io.is_readable_file(os.path.join(source.BASE_PATH, source.NON_READABLE_FILE)))

        # Test directory.
        self.assertFalse(io.is_readable_file(source.BASE_PATH))

        # Test other data types and None values.
        self.assertFalse(io.is_readable_file(source.CUSTOM_DICT))
        self.assertFalse(io.is_readable_file(source.CUSTOM_STR))
        self.assertFalse(io.is_readable_file(None))

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(source.BASE_PATH, source.NON_READABLE_FILE))
