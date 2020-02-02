# coding=utf-8
#
# Unit test for data point creation module.
#
# Copyright: 2020 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import logging
import os
import unittest

from face_amnesia.base.creation import Creation
from face_amnesia.base.data_set import DataSet
from test import source


class TestCreation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Disable logging while testing.
        logging.disable(logging.WARNING)

    def _test_pipeline_with_image(self, handler, use_cnn, res):
        # Test valid image file.
        img_file_path = os.path.join(source.IMG_BASE_PATH, source.JPG_IMG)
        data = handler.process_file(img_file_path, 10)
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 1)
        # Check vectors.
        self.assertEqual(data.get_vector_at_index(0).shape[0], 128)
        # Check metadata.
        (metadata,) = data.get_metadata_at_index(0)
        self.assertEqual(len(metadata), 3)
        self.assertEqual(metadata[0], os.path.basename(img_file_path))
        if use_cnn and res == 0:
            self.assertTupleEqual(metadata[1], source.IMG_CNN_RES_0_BOUNDING_BOX_TOP_LEFT)
            self.assertTupleEqual(metadata[2], source.IMG_CNN_RES_0_BOUNDING_BOX_BOTTOM_RIGHT)
        elif use_cnn and res == 1:
            self.assertTupleEqual(metadata[1], source.IMG_CNN_RES_1_BOUNDING_BOX_TOP_LEFT)
            self.assertTupleEqual(metadata[2], source.IMG_CNN_RES_1_BOUNDING_BOX_BOTTOM_RIGHT)
        elif not use_cnn and res == 0:
            self.assertTupleEqual(metadata[1], source.IMG_HOG_RES_0_BOUNDING_BOX_TOP_LEFT)
            self.assertTupleEqual(metadata[2], source.IMG_HOG_RES_0_BOUNDING_BOX_BOTTOM_RIGHT)
        elif not use_cnn and res == 1:
            self.assertTupleEqual(metadata[1], source.IMG_HOG_RES_1_BOUNDING_BOX_TOP_LEFT)
            self.assertTupleEqual(metadata[2], source.IMG_HOG_RES_1_BOUNDING_BOX_BOTTOM_RIGHT)

        # Test invalid files.
        data = handler.process_file(os.path.join(source.BASE_PATH, source.TXT_TEXT))
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 0)
        data = handler.process_file(os.path.join(source.BASE_PATH, source.NON_EXISTING_FILE))
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 0)
        data = handler.process_file(source.BASE_PATH)
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 0)
        data = handler.process_file(source.CUSTOM_STR)
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 0)

    def test_process_image_file(self):
        handler = Creation(use_cnn=False,
                           use_68_points=True,
                           upsample=0)
        self._test_pipeline_with_image(handler, use_cnn=False, res=0)

        handler = Creation(use_cnn=False,
                           use_68_points=True,
                           upsample=1)
        self._test_pipeline_with_image(handler, use_cnn=False, res=1)

        handler = Creation(use_cnn=False,
                           use_68_points=False,
                           upsample=0)
        self._test_pipeline_with_image(handler, use_cnn=False, res=0)

        handler = Creation(use_cnn=False,
                           use_68_points=False,
                           upsample=1)
        self._test_pipeline_with_image(handler, use_cnn=False, res=1)

        handler = Creation(use_cnn=True,
                           use_68_points=True,
                           upsample=0)
        self._test_pipeline_with_image(handler, use_cnn=True, res=0)

        handler = Creation(use_cnn=True,
                           use_68_points=True,
                           upsample=1)
        self._test_pipeline_with_image(handler, use_cnn=True, res=1)

        handler = Creation(use_cnn=True,
                           use_68_points=False,
                           upsample=0)
        self._test_pipeline_with_image(handler, use_cnn=True, res=0)

        handler = Creation(use_cnn=True,
                           use_68_points=False,
                           upsample=1)
        self._test_pipeline_with_image(handler, use_cnn=True, res=1)

    def _test_pipeline_with_video(self, video_file_path, top_left, bottom_right):
        handler = Creation(use_cnn=False,
                           use_68_points=True,
                           upsample=0)
        data = handler.process_file(video_file_path, source.VIDEO_GIVEN_FRAME_INDEX - 1)
        self.assertIsInstance(data, DataSet)
        self.assertEqual(len(data), 1)
        # Check vector.
        self.assertEqual(data.get_vector_at_index(0).shape[0], 128)
        # Check metadata.
        (metadata,) = data.get_metadata_at_index(0)
        self.assertEqual(len(metadata), 4)
        self.assertEqual(metadata[0], os.path.basename(video_file_path))
        self.assertEqual(metadata[1], 0)
        self.assertTupleEqual(metadata[2], top_left)
        self.assertTupleEqual(metadata[3], bottom_right)

    def test_process_video_file(self):
        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.AVI_MPEG2_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_AVI_MPEG2_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_AVI_MPEG2_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.MOV_MPEG4_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_MOV_MPEG4_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_MOV_MPEG4_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.MP4_XVID_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_MP4_XVID_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_MP4_XVID_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.WEBM_VP8_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_WEBM_VP8_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_WEBM_VP8_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.OGG_THEORA_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_OGG_THEORA_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_OGG_THEORA_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

        video_file_path = os.path.join(source.VIDEO_BASE_PATH, source.FLV_H264_VIDEO)
        self._test_pipeline_with_video(video_file_path,
                                       source.VIDEO_FLV_H264_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT,
                                       source.VIDEO_FLV_H264_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT)

    @classmethod
    def tearDownClass(cls):
        # Enable logging again.
        logging.disable(logging.NOTSET)
