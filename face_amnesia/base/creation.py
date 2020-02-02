# coding=utf-8
#
# Data point creation module.
#
# Copyright: 2020 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import cv2
import os

import dlib
import numpy

from face_amnesia import settings
from face_amnesia.base.data_set import DataSet
from face_amnesia.media import image, video
from face_amnesia.models import loader
from face_amnesia.utils import calculation


class Creation:
    """
    Class to create vectors from face images using
    pre-trained classifier provided by dlib project.
    See http://dlib.net/ for details.
    """

    def __init__(self,
                 use_cnn: bool = False,
                 use_68_points: bool = True,
                 upsample: int = 0):
        """
        Initialize pre-trained dlib classifiers
        according to specified settings.
        In detail:
            - face_detector: Classifier to find
                bounding boxes surrounding
                each face in provided image.
            - shape_predictor: Classifier to detect positions
                of facial features such as eyes or nose.
            - face_descriptor: Classifier to convert facial
                features to a 128-dimensional vector.
        :param use_cnn: bool - Indicate
            if a CNN-based model should be used to find
            human faces. Use a HOG-based detection
            model in case of False (default: False).
        :param use_68_points: bool - Indicate
            if a 68 point model should be used to predict the
            shape of a human face. Use a 5 point prediction
            model in case of False (default: True).
        :param upsample: int - Number of times the
            image resolution should be increased
            to find smaller faces (default: 0).
        """
        # Store number of times the image resolution should be increased
        # so that the face detector is able to find smaller faces.
        if upsample >= 0:
            self.resolution = upsample
        else:
            # Negative resolution values are not allowed, so fall back to default.
            settings.LOGGER.warning("Invalid resolution value, using 0.")
            self.resolution = 0

        # Face detector.
        # ==============

        # Get face detection classifier.
        if use_cnn:
            # Use CNN-based model.
            self.face_detector = loader.get_cnn_face_detector()
        else:
            # Use HOG-based model.
            self.face_detector = loader.get_hog_face_detector()

        # Shape predictor.
        # ================

        # Get shape prediction classifier.
        if use_68_points:
            # Use 68 point model.
            self.shape_predictor = loader.get_68_point_shape_predictor()
        else:
            # Use 5 point model.
            self.shape_predictor = loader.get_5_point_shape_predictor()

        # Face descriptor.
        # ================

        # Get face description model.
        self.face_descriptor = loader.get_face_descriptor()

    def _get_face_descriptions(self, img_rgb: numpy.ndarray) -> list:
        """
        Extract bounding boxes surrounding each face in
        given image and compute its description vector.
        :param img_rgb: numpy.ndarray - RGB pixel matrix of image.
        :return: list(tuple) - Description vector and
            pixel coordinates of top left / bottom right
            corner concerning all detected faces in image.
        """
        # Compute bounding boxes.
        boxes = self.face_detector(img_rgb, self.resolution)
        # Make sure that the data type of received
        # bounding boxes is actually a dlib.rectangles
        # instance containing a number of dlib.rectangle
        # objects. The reason is that the CNN-based face
        # detection model returns a dlib.mmod_rectangles
        # instance which itself contains a number of
        # dlib.mmod_rectangle objects. The latter is just
        # a wrapper of a dlib.rectangle instance with a
        # confidence score. See
        # http://dlib.net/python/index.html#dlib.mmod_rectangle
        # for details.
        if isinstance(boxes, dlib.rectangles):
            bounding_boxes = boxes
        else:
            bounding_boxes = dlib.rectangles()
            for box in boxes:
                bounding_boxes.append(box.rect)
        face_descriptions = list()
        for box in bounding_boxes:
            # Compute position of facial features
            # concerning currently chosen face.
            facial_features = self.shape_predictor(img_rgb, box)
            # Compute corresponding description
            # vector and convert it to NumPy array.
            vector = self.face_descriptor.compute_face_descriptor(img_rgb, facial_features)
            description_vector = numpy.array(vector, dtype=settings.VECTOR_ENTRY_DATA_TYPE)
            # Get and adjust corner coordinates of
            # current bounding box because in rare
            # cases it may be that the received
            # coordinates lie outside the image.
            top_left = image.get_top_left(box, img_rgb)
            bottom_right = image.get_bottom_right(box, img_rgb)
            # Store description vector and its
            # bounding box coordinates as a tuple.
            face_descriptions.append((description_vector, top_left, bottom_right))
        return face_descriptions

    def _process_image(self, img_file_path: str) -> tuple:
        """
        Extract all faces in given image file and
        compute corresponding description vectors.
        :param img_file_path: str - Path to image file.
        :return: (list, list) - List of data
            points and another list of
            corresponding metadata structures.
            Return empty lists in case no
            description vectors could be retrieved.
        """
        # Read specified image file.
        img_bgr = image.read_image_from_file(img_file_path)
        if img_bgr.size < 1:
            # Error: image file could not be
            # read, therefore return empty lists
            # to indicate that no description
            # vectors could be retrieved.
            return list(), list()

        # Convert retrieved image to RGB color encoding.
        img_rgb = image.swap_color_encoding(img_bgr)
        # Get description vector and
        # bounding box coordinates of all
        # detected faces in current image.
        face_descriptions = self._get_face_descriptions(img_rgb)
        # Create empty list to store description vectors.
        vector_list = list()
        # Create empty list to store metadata structures.
        metadata_list = list()
        # Fill both lists with corresponding data of detected faces.
        for face_data_tuple in face_descriptions:
            # Extract and store description vector of current face.
            vector = face_data_tuple[0]
            vector_list.append(vector)
            # Extract and store metadata of current face.
            top_left = face_data_tuple[1]
            bottom_right = face_data_tuple[2]
            # Create metadata structure for current
            # face in image file. In this case, this
            # is a tuple containing file name and top
            # left / bottom right bounding box corners.
            img_metadata = DataSet.create_metadata(os.path.basename(img_file_path),
                                                   top_left,
                                                   bottom_right)
            metadata_list.append(img_metadata)
        return vector_list, metadata_list

    def _process_video(self,
                       video_file_path: str,
                       num_skipped_frames: int) -> tuple:
        """
        Extract all faces in given video file and
        compute corresponding description vectors.
        :param video_file_path: str - Path to video file.
        :param num_skipped_frames: int - Number of skipped
            frames between two consecutive samples.
        :return: (list, list) - List of data
            points and another list of
            corresponding metadata structures.
            Return empty lists in case no
            description vectors could be retrieved.
        """
        # Open video file.
        cap = cv2.VideoCapture(video_file_path)
        # Adjust given number of frames between
        # two samples to indicate the number of
        # necessary forward steps in video file.
        num_skipped_frames += 1
        # Create empty list to store description vectors.
        vector_list = list()
        # Create empty list to store metadata structures.
        metadata_list = list()
        # Create list to store description vectors
        # of last sample. Should be used as cache
        # to avoid multiple insertions of description
        # vectors corresponding to the same person.
        vectors_in_last_sample = list()
        # Read frames with specified sample rate.
        next_frame_index = 0
        ret = True
        while ret:
            # Jump to next frame index.
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
            # Read frame.
            ret, frame_bgr = cap.read()
            # Check if retrieved pixel data is valid.
            if frame_bgr is None:
                # Error: current frame is invalid,
                # therefore skip it and continue
                # with next sample (if possible).
                continue

            # Convert retrieved frame to RGB color encoding.
            frame_rgb = image.swap_color_encoding(frame_bgr)
            # Get current frame number and convert
            # it to corresponding frame index.
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_index = frame_number - 1
            # Get description vector of all
            # detected faces in current frame.
            face_descriptions = self._get_face_descriptions(frame_rgb)
            # Fill description vector and metadata lists
            # with corresponding data of detected faces.
            for face_tuple in face_descriptions:
                # Extract description vector of current face.
                vector = face_tuple[0]
                # Compare current vector to those detected in last
                # frame. When its distance exceeds a certain threshold,
                # assume a new face which needs to be stored.
                present_in_last_sample = False
                for last_vector in vectors_in_last_sample:
                    distance = calculation.get_distance(last_vector, vector)
                    if numpy.less_equal(distance, settings.DISTANCE_THRESHOLD_RECOGNITION):
                        present_in_last_sample = True
                        break
                if not present_in_last_sample:
                    # Store description vector of current face.
                    vector_list.append(vector)
                    # Extract and store corresponding metadata.
                    top_left = face_tuple[1]
                    bottom_right = face_tuple[2]
                    # Create metadata structure for current face in
                    # specified video file frame. In this case, this
                    # is a tuple containing file name, frame index
                    # and top left / bottom right bounding box corners.
                    video_metadata = DataSet.create_metadata(os.path.basename(video_file_path),
                                                             frame_index,
                                                             top_left,
                                                             bottom_right)
                    metadata_list.append(video_metadata)
            # Update description vectors of last
            # frame to those currently retrieved.
            vectors_in_last_sample.clear()
            for vector, _top_left, _bottom_right in face_descriptions:
                vectors_in_last_sample.append(vector)
            # Update frame index.
            next_frame_index += num_skipped_frames
        # Close video file.
        cap.release()
        return vector_list, metadata_list

    def process_file(self,
                     file_path: str,
                     num_skipped_frames: int = -1) -> DataSet:
        """
        Extract all faces in given image or video file
        and compute corresponding description vectors.
        :param file_path: str - Path to image or video file.
        :param num_skipped_frames: int - Number of skipped
            frames between two consecutive samples. In case
            of a negative number, this value is adjusted to
            ensure a sample rate of one sample per second.
            Is ignored in case of an image (default: -1).
        :return: DataSet - Data points and
            corresponding metadata structures.
            Return empty data set if no vectors
            could be retrieved.
        """
        # Make sure that given file path is an
        # absolute path and not a relative one.
        file_path = os.path.abspath(file_path)
        # Check if given file is an image or a video.
        if image.is_valid_image_file(file_path):
            vector_list, metadata_list = self._process_image(file_path)
            new_data = DataSet(vector_list, metadata_list)
            return new_data
        elif video.is_valid_video_file(file_path):
            # A negative number of skipped frames
            # between two consecutive samples is
            # invalid. In this case, use current
            # frame rate to sample once a second.
            if num_skipped_frames < 0:
                frame_rate = video.get_frame_rate(file_path)
                # Check if frame rate could be retrieved.
                if frame_rate > 0:
                    # Frame rate could be retrieved.
                    num_skipped_frames = frame_rate - 1
                else:
                    # Frame rate is invalid, so fall back
                    # to default defined in settings module.
                    settings.LOGGER.warning(
                        "Invalid number of frames between two samples. Use {} instead.".format(
                            settings.NUM_SKIPPED_FRAMES))
                    num_skipped_frames = settings.NUM_SKIPPED_FRAMES
            vector_list, metadata_list = self._process_video(file_path, num_skipped_frames)
            new_data_set = DataSet(vector_list, metadata_list)
            return new_data_set
        else:
            # Error: file is invalid.
            settings.LOGGER.warning("Invalid file: {}".format(file_path))
            return DataSet(list(), list())
