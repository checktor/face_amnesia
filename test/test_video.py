# coding=utf-8
#
# Unit test for video module.
#
# Copyright: C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
import unittest

import numpy

from face_amnesia.media import video
from test import helper, source


class TestVideo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helper.create_non_readable_file(os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO),
                                        os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO))

    def test_is_valid_video_file(self):
        # Test image data.
        self.assertFalse(video.is_valid_video_file(os.path.join(source.IMG_BASE_PATH, source.BMP_IMG)))
        self.assertFalse(video.is_valid_video_file(os.path.join(source.IMG_BASE_PATH, source.JPG_IMG)))
        self.assertFalse(video.is_valid_video_file(os.path.join(source.IMG_BASE_PATH, source.PNG_IMG)))
        self.assertFalse(video.is_valid_video_file(os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG)))

        # Test video data.
        self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO)))
        self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO)))
        self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO)))
        self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO)))
        self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO)))

        # Test text data.
        self.assertFalse(video.is_valid_video_file(os.path.join(source.BASE_PATH, source.TXT_TEXT)))
        self.assertFalse(video.is_valid_video_file(os.path.join(source.BASE_PATH, source.MD_TEXT)))

        # Test non-existing file.
        self.assertFalse(video.is_valid_video_file(os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE)))

        # Test non-readable video.
        if os.geteuid() == 0:
            # Root is always able to read video data.
            self.assertTrue(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO)))
        else:
            self.assertFalse(video.is_valid_video_file(os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO)))

        # Test directory.
        self.assertFalse(video.is_valid_video_file(source.BASE_PATH))

        # Test other data types and None values.
        self.assertFalse(video.is_valid_video_file(source.CUSTOM_DICT))
        self.assertFalse(video.is_valid_video_file(source.CUSTOM_STR))
        self.assertFalse(video.is_valid_video_file(None))

    def test_get_frame_rate(self):
        # Test image data.
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.IMG_BASE_PATH, source.BMP_IMG))
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.IMG_BASE_PATH, source.JPG_IMG))
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.IMG_BASE_PATH, source.PNG_IMG))
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG))

        # Test video data.
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO)),
                         source.VIDEO_FRAME_RATE)
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO)),
                         source.VIDEO_FRAME_RATE)
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO)),
                         source.VIDEO_FRAME_RATE)
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO)),
                         source.VIDEO_FRAME_RATE)
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO)),
                         1000)
        self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO)),
                         source.VIDEO_FRAME_RATE)

        # Test text data.
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.BASE_PATH, source.TXT_TEXT))
        self.assertRaises(ValueError, video.get_frame_rate, os.path.join(source.BASE_PATH, source.MD_TEXT))

        # Test non-existing file.
        self.assertRaises(ValueError, video.get_frame_rate,
                          os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE))

        # Test non-readable video.
        if os.geteuid() == 0:
            # Root is always able to read video data.
            self.assertEqual(video.get_frame_rate(os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO)),
                             source.VIDEO_FRAME_RATE)
        else:
            self.assertRaises(ValueError, video.get_frame_rate,
                              os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO))

        # Test directory.
        self.assertRaises(ValueError, video.get_frame_rate, source.BASE_PATH)

        # Test other data types and None values.
        self.assertRaises(ValueError, video.get_frame_rate, source.CUSTOM_DICT)
        self.assertRaises(ValueError, video.get_frame_rate, source.CUSTOM_STR)
        self.assertRaises(ValueError, video.get_frame_rate, None)

    def _test_read_frame_from_valid_video_file(self,
                                               frame_indices: list,
                                               video_file_path: str,
                                               rgb_first_frame: list,
                                               rgb_given_frame: list):
        for i in frame_indices:
            res = video.read_frame_from_file(video_file_path, i)
            if i == 0:
                self.assertIsInstance(res, numpy.ndarray)
                self.assertEqual(res.shape[0], source.VIDEO_HEIGHT)
                self.assertEqual(res.shape[1], source.VIDEO_WIDTH)
                self.assertListEqual(list(res[source.PIXEL_Y_COORDINATE][source.PIXEL_X_COORDINATE]), rgb_first_frame)
            elif i == source.VIDEO_GIVEN_FRAME_INDEX:
                self.assertIsInstance(res, numpy.ndarray)
                self.assertEqual(res.shape[0], source.VIDEO_HEIGHT)
                self.assertEqual(res.shape[1], source.VIDEO_WIDTH)
                self.assertListEqual(list(res[source.PIXEL_Y_COORDINATE][source.PIXEL_X_COORDINATE]), rgb_given_frame)
            elif 0 < i <= source.VIDEO_LAST_FRAME_INDEX:
                self.assertIsInstance(res, numpy.ndarray)
                self.assertEqual(res.shape[0], source.VIDEO_HEIGHT)
                self.assertEqual(res.shape[1], source.VIDEO_WIDTH)
            else:
                self.assertIsInstance(res, numpy.ndarray)
                self.assertEqual(res.size, 0)

    def _test_read_frame_from_invalid_video_file(self, frame_indices: list, video_file_path: str):
        for i in frame_indices:
            self.assertRaises(ValueError, video.read_frame_from_file, video_file_path, i)

    def test_read_frame_from_file(self):
        # Define frame indices to be tested.
        frame_indices = [-1, 0, 1,
                         source.VIDEO_GIVEN_FRAME_INDEX,
                         source.VIDEO_LAST_FRAME_INDEX + 1]

        # Test video data.
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO),
                                                    source.PIXEL_AVI_MPEG2_FIRST_FRAME,
                                                    source.PIXEL_AVI_MPEG2_GIVEN_FRAME)
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO),
                                                    source.PIXEL_MOV_MPEG4_FIRST_FRAME,
                                                    source.PIXEL_MOV_MPEG4_GIVEN_FRAME)
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO),
                                                    source.PIXEL_MP4_XVID_FIRST_FRAME,
                                                    source.PIXEL_MP4_XVID_GIVEN_FRAME)
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO),
                                                    source.PIXEL_OGG_THEORA_FIRST_FRAME,
                                                    source.PIXEL_OGG_THEORA_GIVEN_FRAME)
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO),
                                                    source.PIXEL_WEBM_VP8_FIRST_FRAME,
                                                    source.PIXEL_WEBM_VP8_GIVEN_FRAME)
        self._test_read_frame_from_valid_video_file(frame_indices,
                                                    os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO),
                                                    source.PIXEL_FLV_H264_FIRST_FRAME,
                                                    source.PIXEL_FLV_H264_GIVEN_FRAME)

        # Test image data.
        self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.IMG_BASE_PATH, source.BMP_IMG))
        self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.IMG_BASE_PATH, source.JPG_IMG))
        self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.IMG_BASE_PATH, source.PNG_IMG))
        self._test_read_frame_from_invalid_video_file(frame_indices,
                                                      os.path.join(source.IMG_BASE_PATH, source.TIFF_IMG))

        # Test text data.
        self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.BASE_PATH, source.TXT_TEXT))
        self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.BASE_PATH, source.MD_TEXT))

        # Test non-existing file.
        self._test_read_frame_from_invalid_video_file(frame_indices,
                                                      os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE))

        # Test non-readable video.
        if os.geteuid() == 0:
            # Root is always able to read video data.
            self._test_read_frame_from_valid_video_file(frame_indices, os.path.join(source.VIDEO_BASE_PATH,
                                                                                    source.NON_READABLE_VIDEO),
                                                        source.PIXEL_FLV_H264_FIRST_FRAME,
                                                        source.PIXEL_FLV_H264_GIVEN_FRAME)
        else:
            self._test_read_frame_from_invalid_video_file(frame_indices, os.path.join(source.VIDEO_BASE_PATH,
                                                                                      source.NON_READABLE_VIDEO))

        # Test directory.
        self._test_read_frame_from_invalid_video_file(frame_indices, source.BASE_PATH)

        # Test other data tyoes and None values.
        self._test_read_frame_from_invalid_video_file(frame_indices, source.CUSTOM_DICT)
        self._test_read_frame_from_invalid_video_file(frame_indices, source.CUSTOM_STR)
        self._test_read_frame_from_invalid_video_file(frame_indices, None)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(source.VIDEO_BASE_PATH, source.NON_READABLE_VIDEO))
