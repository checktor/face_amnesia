# coding=utf-8
#
# Module to load binary models.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import os

import dlib

# Pre-trained models provided by
# Davis E. King as part of dlib project.
# See http://dlib.net/files/ for details.
FACE_DETECTION_MODEL = "mmod_human_face_detector.dat"
SHAPE_PREDICTION_5_POINT_MODEL = "shape_predictor_5_face_landmarks.dat"
SHAPE_PREDICTION_68_POINT_MODEL = "shape_predictor_68_face_landmarks.dat"
FACE_DESCRIPTION_MODEL = "dlib_face_recognition_resnet_model_v1.dat"


def _get_model_bin_path(bin_name: str) -> str:
    """
    Get path to specified binary model.
    :param bin_name: str - File name of binary model.
    :return: str - File path of binary model.
    """
    return os.path.join(os.getcwd(), "face_amnesia", "models", "bin", bin_name)


def get_hog_face_detector():
    """
    Load HOG-based model configured to find human
    faces that are looking towards the camera. See
    http://dlib.net/imaging.html#get_frontal_face_detector
    for further details.
    :return: HOG-based face detection model.
    """
    return dlib.get_frontal_face_detector()


def get_cnn_face_detector():
    """
    Deserialize CNN-based model configured to find
    human faces that are looking towards the camera.
    :return: CNN-based face detection model.
    """
    return dlib.cnn_face_detection_model_v1(_get_model_bin_path(FACE_DETECTION_MODEL))


def get_5_point_shape_predictor():
    """
    Deserialize model configured to predict
    the shape of a face by localizing the
    positions of five facial landmarks, e.g.
    eyes or nose. See
    http://dlib.net/imaging.html#shape_predictor
    for details.
    :return: 5 point shape prediction model.
    """
    return dlib.shape_predictor(_get_model_bin_path(SHAPE_PREDICTION_5_POINT_MODEL))


def get_68_point_shape_predictor():
    """
    Deserialize model configured to predict
    the shape of a face by localizing the
    positions of 68 facial landmarks, e.g.
    the corners of eyes or nose. See
    http://dlib.net/imaging.html#shape_predictor
    for details.
    :return: 68 point shape prediction model.
    """
    return dlib.shape_predictor(_get_model_bin_path(SHAPE_PREDICTION_68_POINT_MODEL))


def get_face_descriptor():
    """
    Deserialize model configured to convert
    coordinates of facial landmarks to
    128-dimensional vector making sure
    that data points of the same person
    have a shorter pairwise distance.
    :return: Face description model.
    """
    return dlib.face_recognition_model_v1(_get_model_bin_path(FACE_DESCRIPTION_MODEL))
