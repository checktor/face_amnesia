# coding=utf-8
#
# Data for unit test.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import os

import numpy

from face_amnesia import settings

# Image and video files.
# ======================

# Base path.
BASE_PATH = "test/files"

# Images.
IMG_BASE_PATH = os.path.join(BASE_PATH, "img")
BMP_IMG = "image.bmp"
JPG_IMG = "image.jpg"
PNG_IMG = "image.png"
TIFF_IMG = "image.tiff"
NON_READABLE_IMG = "non_readable_img.jpg"
IMG_WIDTH = 300
IMG_HEIGHT = 358

# Videos.
VIDEO_BASE_PATH = os.path.join(BASE_PATH, "video")
AVI_MPEG2_VIDEO = "video_mpeg2.avi"
MOV_MPEG4_VIDEO = "video_mpeg4.mov"
MP4_XVID_VIDEO = "video_xvid.mp4"
OGG_THEORA_VIDEO = "video_theora.ogg"
WEBM_VP8_VIDEO = "video_vp8.webm"
FLV_H264_VIDEO = "video_h264.flv"
NON_READABLE_VIDEO = "non_readable_video.webm"
VIDEO_WIDTH = 384
VIDEO_HEIGHT = 288
VIDEO_FRAME_RATE = 25
VIDEO_LAST_FRAME_INDEX = 232

# Texts.
TXT_TEXT = "text.txt"
MD_TEXT = "text.md"

# Other data.
NON_EXISTING_FILE = "non_existing_file.txt"
NON_READABLE_FILE = "non_readable_file.txt"
NON_WRITABLE_FILE = "non_writable_file.txt"

# BGR data of top left pixel
# concerning each image file.
TOP_LEFT_BMP = [255, 23, 20]
TOP_LEFT_JPG = [22, 19, 14]
TOP_LEFT_PNG = [23, 20, 15]
TOP_LEFT_TIFF = [23, 20, 15]

# BGR data of a single pixel
# in first and specified
# frame of each video file.
PIXEL_X_COORDINATE = 0
PIXEL_Y_COORDINATE = 40
VIDEO_GIVEN_FRAME_INDEX = 200
PIXEL_AVI_MPEG2_FIRST_FRAME = [85, 76, 80]
PIXEL_AVI_MPEG2_GIVEN_FRAME = [127, 114, 118]
PIXEL_MOV_MPEG4_FIRST_FRAME = [90, 77, 81]
PIXEL_MOV_MPEG4_GIVEN_FRAME = [131, 116, 119]
PIXEL_MP4_XVID_FIRST_FRAME = [87, 78, 82]
PIXEL_MP4_XVID_GIVEN_FRAME = [138, 114, 116]
PIXEL_OGG_THEORA_FIRST_FRAME = [88, 78, 80]
PIXEL_OGG_THEORA_GIVEN_FRAME = [132, 112, 114]
PIXEL_WEBM_VP8_FIRST_FRAME = [86, 75, 79]
PIXEL_WEBM_VP8_GIVEN_FRAME = [133, 115, 120]
PIXEL_FLV_H264_FIRST_FRAME = [91, 83, 85]
PIXEL_FLV_H264_GIVEN_FRAME = [134, 114, 118]

# Custom pixel data.
# ==================

CUSTOM_IMG_BGR_LIST = [[[255, 1, 2], [1, 255, 2]],
                       [[1, 2, 255], [0, 1, 2]],
                       [[20, 21, 22], [50, 51, 52]]]
CUSTOM_IMG_BGR_NUMPY = numpy.array(CUSTOM_IMG_BGR_LIST, dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)
CUSTOM_IMG_BGR_TUPLE = tuple(CUSTOM_IMG_BGR_LIST)

CUSTOM_IMG_RGB_LIST = [[[2, 1, 255], [2, 255, 1]],
                       [[255, 2, 1], [2, 1, 0]],
                       [[22, 21, 20], [52, 51, 50]]]
CUSTOM_IMG_RGB_NUMPY = numpy.array(
    CUSTOM_IMG_RGB_LIST, dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)
CUSTOM_IMG_RGB_TUPLE = tuple(CUSTOM_IMG_RGB_LIST)

CUSTOM_IMG_1D_STRUCTURE_LIST = [2, 1, 255, 2, 255,
                                1, 255, 2, 1, 2, 1, 0, 22, 21, 20, 52, 51, 50]
CUSTOM_IMG_1D_STRUCTURE_NUMPY = numpy.array(CUSTOM_IMG_1D_STRUCTURE_LIST, dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)

CUSTOM_IMG_2D_STRUCTURE_LIST = [[2, 1, 255],
                                [2, 255, 1],
                                [255, 2, 1],
                                [2, 1, 0],
                                [22, 21, 20],
                                [52, 51, 50]]
CUSTOM_IMG_2D_STRUCTURE_NUMPY = numpy.array(CUSTOM_IMG_2D_STRUCTURE_LIST,
                                            dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)

CUSTOM_IMG_TWO_COLOR_LIST = [[[2, 1], [2, 255]],
                             [[255, 2], [2, 1]],
                             [[22, 21], [52, 51]]]
CUSTOM_IMG_TWO_COLOR_NUMPY = numpy.array(CUSTOM_IMG_TWO_COLOR_LIST,
                                         dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)

CUSTOM_IMG_FOUR_COLOR_LIST = [[[2, 1, 255, 0], [2, 255, 1, 0]],
                              [[255, 2, 1, 0], [2, 1, 0, 0]],
                              [[22, 21, 20, 0], [52, 51, 50, 0]]]
CUSTOM_IMG_FOUR_COLOR_NUMPY = numpy.array(CUSTOM_IMG_FOUR_COLOR_LIST,
                                          dtype=settings.RGB_CHANNEL_VALUE_DATA_TYPE)

CUSTOM_IMG_WRONG_DATA_TYPE_NUMPY = numpy.array(
    CUSTOM_IMG_BGR_LIST, dtype=numpy.int8)

CUSTOM_IMG_WIDTH = 2
CUSTOM_IMG_HEIGHT = 3

# Vectors.
# ========

FIRST_VECTOR_2D_LIST = [1, 2]
FIRST_VECTOR_2D_1_NORM = 3
FIRST_VECTOR_2D_2_NORM = numpy.sqrt(5)
FIRST_VECTOR_2D_METADATA = set(list(("meta 1a", "meta 1b")))
FIRST_VECTOR_2D_NUMPY = numpy.array(
    FIRST_VECTOR_2D_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
FIRST_VECTOR_2D_TUPLE = tuple(FIRST_VECTOR_2D_LIST)
FIRST_VECTOR_2D_NUMPY_WRONG_DTYPE = numpy.array(FIRST_VECTOR_2D_NUMPY, dtype=numpy.int64)

FIRST_VECTOR_2D_REVERSED_LIST = FIRST_VECTOR_2D_LIST[::-1]
FIRST_VECTOR_2D_REVERSED_METADATA = set(list(("meta 1a-rev", "meta 1b-rev")))
FIRST_VECTOR_2D_REVERSED_NUMPY = numpy.array(
    FIRST_VECTOR_2D_REVERSED_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
FIRST_VECTOR_2D_REVERSED_TUPLE = tuple(FIRST_VECTOR_2D_REVERSED_LIST)

SECOND_VECTOR_2D_LIST = [3, 4]
SECOND_VECTOR_2D_1_NORM = 7
SECOND_VECTOR_2D_2_NORM = 5
SECOND_VECTOR_2D_METADATA = set(list(("meta 2a", "meta 2b")))
SECOND_VECTOR_2D_NUMPY = numpy.array(
    SECOND_VECTOR_2D_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
SECOND_VECTOR_2D_TUPLE = tuple(SECOND_VECTOR_2D_LIST)

SECOND_VECTOR_2D_REVERSED_LIST = SECOND_VECTOR_2D_LIST[::-1]
SECOND_VECTOR_2D_REVERSED_METADATA = set(list(("meta 2a-rev", "meta 2b-rev")))
SECOND_VECTOR_2D_REVERSED_NUMPY = numpy.array(
    SECOND_VECTOR_2D_REVERSED_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
SECOND_VECTOR_2D_REVERSED_TUPLE = tuple(SECOND_VECTOR_2D_REVERSED_LIST)

DISTANCE_1_NORM_FIRST_SECOND = 4
DISTANCE_2_NORM_FIRST_SECOND = numpy.sqrt(8)

VECTOR_3D_LIST = [1, 2, 3]
VECTOR_3D_METADATA = set(list(("meta 3a", "meta 3b", "meta 3c", "meta 3d")))
VECTOR_3D_1_NORM = 6
VECTOR_3D_2_NORM = numpy.sqrt(14)
VECTOR_3D_NUMPY = numpy.array(
    VECTOR_3D_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
VECTOR_3D_TUPLE = tuple(VECTOR_3D_LIST)

INVALID_VECTOR_LIST = ["a", "b"]
INVALID_VECTOR_NUMPY = numpy.array(INVALID_VECTOR_LIST)
INVALID_VECTOR_TUPLE = tuple(INVALID_VECTOR_LIST)

# Matrices.
# =========

MATRIX_LIST = [FIRST_VECTOR_2D_LIST,
               FIRST_VECTOR_2D_REVERSED_LIST,
               SECOND_VECTOR_2D_LIST,
               SECOND_VECTOR_2D_REVERSED_LIST]
MATRIX_1_NORM = numpy.array(
    [FIRST_VECTOR_2D_1_NORM,
     FIRST_VECTOR_2D_1_NORM,
     SECOND_VECTOR_2D_1_NORM,
     SECOND_VECTOR_2D_1_NORM],
    dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_2_NORM = numpy.array(
    [FIRST_VECTOR_2D_2_NORM,
     FIRST_VECTOR_2D_2_NORM,
     SECOND_VECTOR_2D_2_NORM,
     SECOND_VECTOR_2D_2_NORM],
    dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_METADATA = [FIRST_VECTOR_2D_METADATA,
                   FIRST_VECTOR_2D_REVERSED_METADATA,
                   SECOND_VECTOR_2D_METADATA,
                   SECOND_VECTOR_2D_REVERSED_METADATA]
MATRIX_NUMPY = numpy.array(MATRIX_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_TUPLE = tuple(MATRIX_LIST)
MATRIX_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY,
                         FIRST_VECTOR_2D_REVERSED_NUMPY,
                         SECOND_VECTOR_2D_NUMPY,
                         SECOND_VECTOR_2D_REVERSED_NUMPY]

MATRIX_MEAN_VECTOR = numpy.array(
    [2.5, 2.5], dtype=settings.VECTOR_ENTRY_DATA_TYPE)

INVALID_MATRIX_LIST = [FIRST_VECTOR_2D_LIST,
                       INVALID_VECTOR_LIST,
                       SECOND_VECTOR_2D_LIST,
                       SECOND_VECTOR_2D_REVERSED_LIST]
INVALID_MATRIX_NUMPY = numpy.array(INVALID_MATRIX_LIST)
INVALID_MATRIX_TUPLE = tuple(INVALID_MATRIX_LIST)
INVALID_MATRIX_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY,
                                 INVALID_VECTOR_NUMPY,
                                 SECOND_VECTOR_2D_NUMPY,
                                 SECOND_VECTOR_2D_REVERSED_NUMPY]

INVALID_MATRIX_ROW_SIZE_LIST_OF_NUMPYS = [numpy.empty(0),
                                          numpy.empty(0),
                                          numpy.empty(0),
                                          numpy.empty(0)]
INVALID_MATRIX_ROW_SIZE_NUMPY = numpy.array(INVALID_MATRIX_ROW_SIZE_LIST_OF_NUMPYS)

INVALID_MATRIX_SHAPE_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY,
                                       VECTOR_3D_NUMPY,
                                       SECOND_VECTOR_2D_NUMPY,
                                       SECOND_VECTOR_2D_REVERSED_NUMPY]
INVALID_MATRIX_DTYPE_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY,
                                       FIRST_VECTOR_2D_NUMPY_WRONG_DTYPE,
                                       SECOND_VECTOR_2D_NUMPY,
                                       SECOND_VECTOR_2D_REVERSED_NUMPY]

MATRIX_ONE_ENTRY_LIST = [FIRST_VECTOR_2D_LIST]
MATRIX_ONE_ENTRY_1_NORM = numpy.array(
    [FIRST_VECTOR_2D_1_NORM], dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_ONE_ENTRY_2_NORM = numpy.array(
    [FIRST_VECTOR_2D_2_NORM], dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_ONE_ENTRY_METADATA = [FIRST_VECTOR_2D_METADATA]
MATRIX_ONE_ENTRY_NUMPY = numpy.array(
    MATRIX_ONE_ENTRY_LIST, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_ONE_ENTRY_TUPLE = tuple(MATRIX_ONE_ENTRY_LIST)
MATRIX_ONE_ENTRY_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY]

MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS = [FIRST_VECTOR_2D_NUMPY,
                                         FIRST_VECTOR_2D_REVERSED_NUMPY,
                                         FIRST_VECTOR_2D_REVERSED_NUMPY,
                                         SECOND_VECTOR_2D_NUMPY,
                                         SECOND_VECTOR_2D_REVERSED_NUMPY,
                                         FIRST_VECTOR_2D_REVERSED_NUMPY,
                                         FIRST_VECTOR_2D_REVERSED_NUMPY]
MATRIX_WITH_DUPLICATES_METADATA_EQUAL = [FIRST_VECTOR_2D_METADATA,
                                         FIRST_VECTOR_2D_REVERSED_METADATA,
                                         FIRST_VECTOR_2D_REVERSED_METADATA,
                                         SECOND_VECTOR_2D_METADATA,
                                         SECOND_VECTOR_2D_REVERSED_METADATA,
                                         FIRST_VECTOR_2D_REVERSED_METADATA,
                                         FIRST_VECTOR_2D_REVERSED_METADATA]
MATRIX_WITH_DUPLICATES_METADATA_UNEQUAL = [FIRST_VECTOR_2D_METADATA,
                                           FIRST_VECTOR_2D_REVERSED_METADATA,
                                           set(list(("meta 1a-dup", "meta 1b-dup"))),
                                           SECOND_VECTOR_2D_METADATA,
                                           SECOND_VECTOR_2D_REVERSED_METADATA,
                                           set(list(("meta 1a-dup", "meta 1b-dup"))),
                                           set(list(("meta 1a-dup", "meta 1b-dup")))]
MATRIX_WITH_DUPLICATES_NUMPY = numpy.array(
    MATRIX_WITH_DUPLICATES_LIST_OF_NUMPYS, dtype=settings.VECTOR_ENTRY_DATA_TYPE)

# Other data types.
# =================

CUSTOM_STR = "str"
CUSTOM_DICT = {"key1": "value1", "key2": ["value2a", "value2b"]}

# Clustering data.
# ================

MATRIX_ONE_CLUSTER_NUMPY = numpy.array(
    [[2.5, 2.5]], dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_ONE_CLUSTER_METADATA = [
    set(list(('meta 1a', 'meta 1b', 'meta 1a-rev', 'meta 1b-rev', 'meta 2a', 'meta 2b', 'meta 2a-rev', 'meta 2b-rev')))]

MATRIX_TWO_CLUSTER_NUMPY = numpy.array(
    [[1.5, 1.5], [3.5, 3.5]], dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_TWO_CLUSTER_METADATA = [set(list(('meta 1a', 'meta 1b', 'meta 1a-rev', 'meta 1b-rev'))),
                               set(list(('meta 2a', 'meta 2b', 'meta 2a-rev', 'meta 2b-rev')))]

MATRIX_WITH_DUPLICATES_ONE_CLUSTER_NUMPY = numpy.array(
    [[(16 / 7), (13 / 7)]], dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_WITH_DUPLICATES_ONE_CLUSTER_METADATA_EQUAL = [
    set(list(('meta 1a', 'meta 1b', 'meta 1a-rev', 'meta 1b-rev', 'meta 2a', 'meta 2b', 'meta 2a-rev', 'meta 2b-rev')))]
MATRIX_WITH_DUPLICATES_ONE_CLUSTER_METADATA_UNEQUAL = [set(list((
    ('meta 1a', 'meta 1b', 'meta 1a-dup', 'meta 1b-dup', 'meta 1a-rev', 'meta 1b-rev', 'meta 2a', 'meta 2b',
     'meta 2a-rev',
     'meta 2b-rev'))))]

MATRIX_WITH_DUPLICATES_TWO_CLUSTER_NUMPY = numpy.array([[(9 / 5), (6 / 5)], [3.5, 3.5]],
                                                       dtype=settings.VECTOR_ENTRY_DATA_TYPE)
MATRIX_WITH_DUPLICATES_TWO_CLUSTER_METADATA_EQUAL = [set(list(('meta 1a', 'meta 1b', 'meta 1a-rev', 'meta 1b-rev'))),
                                                     set(list(('meta 2a', 'meta 2b', 'meta 2a-rev', 'meta 2b-rev')))]
MATRIX_WITH_DUPLICATES_TWO_CLUSTER_METADATA_UNEQUAL = [
    set(list(('meta 1a', 'meta 1b', 'meta 1a-dup', 'meta 1b-dup', 'meta 1a-rev', 'meta 1b-rev'))),
    set(list(('meta 2a', 'meta 2b', 'meta 2a-rev', 'meta 2b-rev')))]

MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_EQUAL = [set(list(('meta 1a', 'meta 1b'))),
                                                      set(list(('meta 1a-rev', 'meta 1b-rev'))),
                                                      set(list(('meta 2a', 'meta 2b'))),
                                                      set(list(('meta 2a-rev', 'meta 2b-rev')))]
MATRIX_WITH_DUPLICATES_FOUR_CLUSTER_METADATA_UNEQUAL = [set(list(('meta 1a', 'meta 1b'))),
                                                        set(list(('meta 1a-dup', 'meta 1b-dup', 'meta 1a-rev',
                                                                  'meta 1b-rev'))),
                                                        set(list(('meta 2a', 'meta 2b'))),
                                                        set(list(('meta 2a-rev', 'meta 2b-rev')))]

# Create data points.
# ===================

IMG_HOG_RES_0_BOUNDING_BOX_TOP_LEFT = (100, 100)
IMG_HOG_RES_0_BOUNDING_BOX_BOTTOM_RIGHT = (204, 204)
IMG_HOG_RES_1_BOUNDING_BOX_TOP_LEFT = (91, 92)
IMG_HOG_RES_1_BOUNDING_BOX_BOTTOM_RIGHT = (199, 199)
IMG_CNN_RES_0_BOUNDING_BOX_TOP_LEFT = (86, 92)
IMG_CNN_RES_0_BOUNDING_BOX_BOTTOM_RIGHT = (200, 206)
IMG_CNN_RES_1_BOUNDING_BOX_TOP_LEFT = (91, 101)
IMG_CNN_RES_1_BOUNDING_BOX_BOTTOM_RIGHT = (190, 199)

VIDEO_AVI_MPEG2_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (74, 121)
VIDEO_AVI_MPEG2_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (160, 208)
VIDEO_MOV_MPEG4_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (74, 121)
VIDEO_MOV_MPEG4_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (160, 208)
VIDEO_OGG_THEORA_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (74, 121)
VIDEO_OGG_THEORA_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (160, 208)
VIDEO_WEBM_VP8_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (74, 121)
VIDEO_WEBM_VP8_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (160, 208)
VIDEO_MP4_XVID_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (65, 112)
VIDEO_MP4_XVID_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (169, 215)
VIDEO_FLV_H264_FIRST_FRAME_BOUNDING_BOX_TOP_LEFT = (65, 112)
VIDEO_FLV_H264_FIRST_FRAME_BOUNDING_BOX_BOTTOM_RIGHT = (169, 215)

# Retrieve data points.
# =====================

RETRIEVAL_TESTING_BASE_PATH = os.path.join(BASE_PATH, "retrieval_testing")

RETRIEVAL_DATA_BASE_PATH = os.path.join(BASE_PATH, "retrieval_data")
RETRIEVAL_DATA_FOLDER_PATH = os.path.join(RETRIEVAL_DATA_BASE_PATH, "folder")
RETRIEVAL_DATA_SUBFOLDER_PATH = os.path.join(RETRIEVAL_DATA_BASE_PATH, "folder", "subfolder")

FILE_MATRIX = [numpy.array([5, 6], dtype=settings.VECTOR_ENTRY_DATA_TYPE),
               numpy.array([6, 5], dtype=settings.VECTOR_ENTRY_DATA_TYPE)]
FILE_METADATA = [set(list(("meta 5a", "meta 5b"))), set(list(("meta 5a-rev", "meta 5b-rev")))]
FILE_DATA_NAME = "file_data"

FOLDER_MATRIX = [numpy.array([7, 8], dtype=settings.VECTOR_ENTRY_DATA_TYPE),
                 numpy.array([8, 7], dtype=settings.VECTOR_ENTRY_DATA_TYPE)]
FOLDER_METADATA = [set(list(("meta 7a", "meta 7b"))), set(list(("meta 7a-rev", "meta 7b-rev")))]
FOLDER_DATA_NAME = "folder_data"

SUBFOLDER_FIRST_MATRIX = [numpy.array([9, 10], dtype=settings.VECTOR_ENTRY_DATA_TYPE),
                          numpy.array([10, 9], dtype=settings.VECTOR_ENTRY_DATA_TYPE)]
SUBFOLDER_FIRST_METADATA = [set(list(("meta 9a", "meta 9b"))), set(list(("meta 9a-rev", "meta 9b-rev")))]
SUBFOLDER_FIRST_DATA_NAME = "subfolder_data_1"

SUBFOLDER_SECOND_MATRIX = [numpy.array([11, 12], dtype=settings.VECTOR_ENTRY_DATA_TYPE)]
SUBFOLDER_SECOND_METADATA = [set(list(("meta 11a", "meta 11b")))]
SUBFOLDER_SECOND_DATA_NAME = "subfolder_data_2"

NUM_DATA_POINTS_FIRST_BATCH = 6
NUM_DATA_POINTS_SECOND_BATCH = 5

FILE_NAMES_FIRST_BATCH = [FILE_DATA_NAME]
FILE_NAMES_SECOND_BATCH = [FOLDER_DATA_NAME, SUBFOLDER_FIRST_DATA_NAME, SUBFOLDER_SECOND_DATA_NAME]

FILE_PATHS_FIRST_BATCH = [os.path.abspath(os.path.join(RETRIEVAL_DATA_BASE_PATH, FILE_DATA_NAME))]
FILE_PATHS_SECOND_BATCH = [os.path.abspath(os.path.join(RETRIEVAL_DATA_FOLDER_PATH, FOLDER_DATA_NAME)),
                           os.path.abspath(os.path.join(RETRIEVAL_DATA_SUBFOLDER_PATH, SUBFOLDER_FIRST_DATA_NAME)),
                           os.path.abspath(os.path.join(RETRIEVAL_DATA_SUBFOLDER_PATH, SUBFOLDER_SECOND_DATA_NAME))]
