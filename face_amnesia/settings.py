# coding=utf-8
#
# Global settings.
#
# Copyright: 2020 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import logging
import os
import sys

import numpy

# Data type for description vector entries.
VECTOR_ENTRY_DATA_TYPE = numpy.float64

# Data type for RGB or BGR channel values.
RGB_CHANNEL_VALUE_DATA_TYPE = numpy.uint8

# Maximum distance between two face
# description vectors of the same person
# used for recognition. Compare with
# http://dlib.net/face_recognition.py.html.
DISTANCE_THRESHOLD_RECOGNITION = 0.575

# Maximum distance between two face
# description vectors of the same person
# used for clustering. Compare with
# http://dlib.net/face_clustering.py.html.
DISTANCE_THRESHOLD_CLUSTERING = 0.45

# Number of skipped frames between two
# consecutive samples of a video file.
NUM_SKIPPED_FRAMES = 24

# Define default LSH parameters
# (using random vectors).
NUM_RANDOM_HASH_FUNCTIONS = 6
NUM_RANDOM_HASH_TABLES = 7
RANDOM_BUCKET_WIDTH = 0.95

# Define default LSH parameters
# (using PCA vectors).
NUM_PCA_HASH_FUNCTIONS = 8
NUM_PCA_HASH_TABLES = 7
PCA_BUCKET_WIDTH = 0.2

# Project name.
PROJECT_NAME = "face_amnesia"

# Absolute path to folder in which
# images and videos are stored.
MEDIA_FOLDER_PATH = os.path.join(os.path.expanduser("~"), PROJECT_NAME, "media")
# Create corresponding directory (if necessary).
os.makedirs(MEDIA_FOLDER_PATH, exist_ok=True)

# Absolute path to folder in which
# description vectors are stored.
DATA_POINT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), PROJECT_NAME, "data_points")
# Create corresponding directory (if necessary).
os.makedirs(DATA_POINT_FOLDER_PATH, exist_ok=True)

# Logger.
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s - %(message)s\n")
handler.setFormatter(formatter)
LOGGER = logging.getLogger(PROJECT_NAME)
LOGGER.addHandler(handler)
