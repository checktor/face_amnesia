# coding=utf-8
#
# Module to manage data points and corresponding metadata.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
from abc import ABC, abstractmethod

import cv2
import numpy

from face_amnesia import settings
from face_amnesia.media import image, video

# Image extraction modes.
RAW_IMG = 0
DRAW_BOX = 1
CUT_OUT_BOX = 2

# String patterns.
SEPARATOR = "==================================================\n"


def are_valid_data_points(data_points: list) -> bool:
    """
    Check if all data points in given list are valid.
    :param data_points: list(numpy.ndarray) - Data points.
    :return: bool - True if all given data
        points are valid and False otherwise.
    """
    if isinstance(data_points, list):
        if len(data_points) <= 0:
            return True
        if not isinstance(data_points[0], numpy.ndarray):
            return False
        first_shape = data_points[0].shape
        if len(first_shape) != 1:
            return False
        if first_shape[0] <= 0:
            return False
        first_dtype = data_points[0].dtype
        if first_dtype != settings.VECTOR_ENTRY_DATA_TYPE:
            return False
        for point in data_points:
            if not isinstance(point, numpy.ndarray):
                return False
            current_shape = point.shape
            if first_shape != current_shape:
                return False
            current_dytpe = point.dtype
            if first_dtype != current_dytpe:
                return False
        # No errors could be found.
        return True
    else:
        # Invalid data type.
        return False


class BaseDataSet(ABC):
    """Abstract base class to manage data points and corresponding metadata."""

    @staticmethod
    @abstractmethod
    def create_metadata(*args):
        """
        Abstract base function to create metadata structure.
        :param args: tuple - Arguments.
        :return: New metadata structure.
        """
        pass

    @staticmethod
    @abstractmethod
    def merge_metadata(source_metadata, target_metadata):
        """
        Abstract base function to combine and
        deduplicate given metadata structures.
        :param source_metadata: Metadata structure.
        :param target_metadata: Metadata structure.
        :return: Combined metadata structure.
        """
        pass


class DataSet(BaseDataSet):
    """
    Concrete class to manage data points and corresponding metadata.

    Each data point is represented as a single NumPy array containing
    64 bit floating-point numbers (as specified in settings module).
    The vectors are stored in a list which can be seen as a matrix
    whose rows represent the data points. The corresponding metadata
    of each vector is organized as a set in another list whose index
    matches that of corresponding data point. The set structure allows
    to store multiple metadata entries per data point while automatically
    removing duplicates.

    Note that this structure implicitly assumes a single metadata entry
    to be hashable which is naturally true for tuples but not for lists
    or dictionaries. In this case, the metadata structure needs to be
    converted to something hashable (e.g. by converting a dictionary to
    a tuple using tuple(d.items())).

    In order to use non-hashable metadata structures without conversion,
    adjust create_metadata() and merge_metadata() methods correspondingly.
    """

    def __init__(self, data_points: list, metadata: list):
        """
        Initialize data points and corresponding metadata structures.
        :param data_points: list(numpy.ndarray) - Data points.
        :param metadata: list(set) - Metadata structures.
        """
        self.data_points = data_points
        self.metadata = metadata

    @property
    def data_points(self) -> list:
        return self._data_points

    @data_points.setter
    def data_points(self, new_data_points: list):
        """
        Set given data points.
        :param new_data_points: list(numpy.ndarray) - Data points.
        :raise ValueError: In case of invalid data points.
        """
        # Check if given data points are valid.
        if not are_valid_data_points(new_data_points):
            raise ValueError

        self._data_points = new_data_points

    @property
    def metadata(self) -> list:
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata: list):
        """
        Set given metadata structures.
        :param new_metadata: list(set) - Metadata structures.
        :raise ValueError: In case of invalid metadata structures.
        """
        if isinstance(new_metadata, list):
            if len(new_metadata) == len(self.data_points):
                # Number of metadata structures matches
                # the number of data points. Therefore,
                # metadata is valid and can be stored.
                self._metadata = new_metadata
            elif len(new_metadata) <= 0:
                # No metadata specified, so simply store an
                # empty metadata structure for each data point.
                new_metadata = list()
                for i in range(len(self.data_points)):
                    new_metadata.append(set())
                self._metadata = new_metadata
            else:
                # Given data structure does not contain the correct
                # number of metadata instances, so raise a ValueError.
                raise ValueError
        else:
            # Given data structure does not represent
            # valid metadata, so raise a ValueError.
            raise ValueError

    def __len__(self) -> int:
        """
        Use number of data points as length of current instance.
        :return: int - Number of data points.
        """
        return len(self.data_points)

    def _index_is_valid(self, index: int) -> bool:
        """
        Check if given index is valid.
        :param index: int - Index.
        :return: bool - True if index
            is valid and False otherwise.
        """
        return 0 <= index < len(self)

    def get_vector_at_index(self, index: int) -> numpy.ndarray:
        """
        Get data point at specified index.
        :param index: int - Index of data point.
        :return: numpy.ndarry - Data point at given index.
        :raise IndexError: In case of an invalid index.
        """
        if not self._index_is_valid(index):
            raise IndexError

        return self.data_points[index]

    def get_metadata_at_index(self, index: int) -> set:
        """
        Get metadata structure at specified index.
        :param index: int - Index of metadata structure.
        :return: list - Metadata structure at given index.
        :raise IndexError: In case specified index is invalid.
        """
        if not self._index_is_valid(index):
            raise IndexError

        return self.metadata[index]

    def get_data_at_index(self, index: int) -> tuple:
        """
        Get data point and corresponding metadata structure at specified index.
        :param index: int - Index.
        :return: tuple - Data point and corresponding metadata structure.
        :raise IndexError: In case specified index is invalid.
        """
        if not self._index_is_valid(index):
            raise IndexError

        return self.get_vector_at_index(index), self.get_metadata_at_index(index)

    def _append_data_points(self, new_data_points: list):
        """
        Append given data points to current instance.
        :param new_data_points: list - New data points.
        :raise ValueError: In case of invalid or non-matching data points.
        """
        # Check if provided data set is empty.
        # In this case, there is nothing to do.
        if len(new_data_points) <= 0:
            return

        if len(self.data_points) <= 0:
            # There are no previously
            # stored data point.
            self._data_points += new_data_points
        else:
            # There are previously stored data points,
            # so check if shape of new data is matching.
            first_data_point = new_data_points[0]
            if first_data_point.size > 0 and first_data_point.shape == self.data_points[0].shape:
                self._data_points += new_data_points
            else:
                raise ValueError

    def _append_metadata(self, new_metadata: list):
        """
        Append given metadata structures to current instance.
        :param new_metadata: list(set) - New metadata structures.
        """
        # Check if provided metadata is empty.
        # In this case, there is nothing to do.
        if len(new_metadata) <= 0:
            return

        self.metadata += new_metadata

    def add_data_point(self, new_data_point: numpy.ndarray, new_metadata: set):
        """
        Add given data point and corresponding
        metadata structure to current instance.
        :param new_data_point: numpy.ndarray - Data point.
        :param new_metadata: set(tuple) - Metadata structure.
        :raise ValueError: In case of invalid or non-matching data.
        """
        # Check if given data point is valid.
        if not are_valid_data_points([new_data_point]):
            raise ValueError

        # Check if given metadata is valid.
        if not isinstance(new_metadata, set):
            raise ValueError

        # Append data point.
        self._append_data_points([new_data_point])

        # Append metadata.
        self._append_metadata([new_metadata])

    def add_data_set(self, new_data_set):
        """
        Add given data set to current instance.
        :param new_data_set: DataSet - Data points
            and corresponding metadata structures.
        :raise ValueError: In case of invalid
            or non-matching data.
        """
        # Check if given data is valid.
        if not isinstance(new_data_set, DataSet):
            raise ValueError

        # Append data points.
        self._append_data_points(new_data_set.data_points)

        # Append metadata.
        self._append_metadata(new_data_set.metadata)

    @staticmethod
    def create_metadata(*args) -> set:
        """
        Create new metadata structure by simply
        adding given argument tuple to a set.
        :param args: tuple - Arguments.
        :return: set(tuple) - New metadata structure.
        """
        new_metadata = set()
        new_metadata.add(args)
        return new_metadata

    @staticmethod
    def merge_metadata(source_metadata: set, target_metadata: set) -> set:
        """
        Combine both metadata structures to a single one excluding duplicates.
        :param source_metadata: set(tuple) - Metadata structure.
        :param target_metadata: set(tuple) - Metadata structure.
        :return: set(tuple) - Combined metadata structure excluding duplicates.
        """
        return source_metadata.union(target_metadata)

    def get_metadata_as_string(self, headline: str = "") -> str:
        """
        Get string representation of all metadata
        structures in current instance.
        :param headline: str - Headline in
            order to keep metadata of different
            data sets apart (default: "").
        :return: str - Representation of metadata
            structures in current instance.
        """
        # Store parts of final string in a list to
        # reduce the number of string concatenations.
        string_parts = list()
        # Add specified description
        # label as headline.
        string_parts.append(headline)
        string_parts.append('\n')
        string_parts.append(SEPARATOR)
        string_parts.append(SEPARATOR)
        # Iterate through all data points
        # and retrieve its metadata structure.
        for i in range(len(self.data_points)):
            metadata = self.get_metadata_at_index(i)
            # Iterate through each entry
            # in current metadata structure.
            for md in metadata:
                # Append file path.
                string_parts.append("\nFile name: {}\n".format(md[0]))
                # Check if metadata structure
                # belongs to an image or a video.
                if len(md) == 4:
                    # Current metadata entry belongs to video file.
                    string_parts.append("Frame number: {}".format(md[1]))
                    string_parts.append("\nBounding box:\n\tTop left: {}\n\tBottom right: {}".format(md[2], md[3]))
                elif len(md) == 3:
                    # Current metadata entry belongs to image file.
                    string_parts.append("\nBounding box:\n\tTop left: {}\n\tBottom right: {}".format(md[1], md[2]))
                else:
                    # Current metadata entry is unknown.
                    settings.LOGGER.warning("Skipped unknown metadata entry.\n{}".format(str(md)))
                string_parts.append("\n\n")
            string_parts.append(SEPARATOR)
        return "".join(string_parts)

    def get_images_at_index(self, index: int, extraction_mode: int = RAW_IMG) -> list:
        """
        Get all images represented by data point at specified index.
        :param index: int - Index of data point.
        :param extraction_mode: int - Specify how
            represented images should be extracted.
            Options:
                RAW_IMG = 0: Get original image.
                DRAW_BOX = 1: Draw bounding box around recognized face.
                CUT_OUT_BOX = 2: Cut recognized face out of image.
            Default: RAW_IMG.
        :return: list - BGR pixel matrix of all images
            represented by data point at specified index.
        """
        # Get metadata at specified index.
        metadata = self.get_metadata_at_index(index)
        # Create list to store corresponding images.
        current_images = list()
        # Iterate through each entry
        # in current metadata structure.
        for md in metadata:
            # Extract file path.
            file_path = os.path.join(settings.MEDIA_FOLDER_PATH, md[0])
            # Check if metadata structure
            # belongs to an image or a video.
            if len(md) == 4:
                # Current metadata entry belongs to a video.
                frame_index = md[1]
                # Read corresponding frame.
                img_bgr = video.read_frame_from_file(file_path, frame_index)
                # Check if pixel matrix of
                # current frame could be retrieved.
                if img_bgr.size < 1:
                    # Pixel matrix of current frame is
                    # invalid, so skip this iteration.
                    settings.LOGGER.warning("Metadata points to invalid file.\n{}".format(md))
                    continue
                # Get bounding box.
                top_left = md[2]
                bottom_right = md[3]
                if extraction_mode == DRAW_BOX:
                    # Draw bounding box in retrieved image.
                    image.draw_bounding_box(img_bgr, top_left, bottom_right)
                elif extraction_mode == CUT_OUT_BOX:
                    # Extract bounding box from retrieved frame.
                    img_bgr = image.extract_bounding_box(img_bgr, top_left, bottom_right)
                # Append frame to result data.
                current_images.append(img_bgr)
            elif len(md) == 3:
                # Current metadata entry belongs
                # to an image, so read it.
                img_bgr = image.read_image_from_file(file_path)
                # Get bounding box.
                top_left = md[1]
                bottom_right = md[2]
                if extraction_mode == DRAW_BOX:
                    # Draw bounding box in retrieved image.
                    image.draw_bounding_box(img_bgr, top_left, bottom_right)
                elif extraction_mode == CUT_OUT_BOX:
                    # Extract bounding box from retrieved image.
                    img_bgr = image.extract_bounding_box(img_bgr, top_left, bottom_right)
                # Append image to result data.
                current_images.append(img_bgr)
            else:
                # Current metadata entry is unknown.
                settings.LOGGER.warning("Skipped unknown metadata entry.\n{}".format(str(md)))
        return current_images

    def get_images(self, extraction_mode: bool = RAW_IMG) -> list:
        """
        Get all available images represented by current instance.
        :param extraction_mode: int - Specify how
            represented images should be extracted.
            Options:
                RAW_IMG = 0: Get original image.
                DRAW_BOX = 1: Draw bounding box around recognized face.
                CUT_OUT_BOX = 2: Cut recognized face out of image.
            Default: RAW_IMG.
        :return: list - BGR pixel matrix of all
            images represented by current data.
        """
        # Create list to store pixel
        # matrices concerning each data point.
        matching_images = list()
        # Iterate through all data points and
        # retrieve its corresponding pixel data.
        for i in range(len(self.data_points)):
            current_images = self.get_images_at_index(i, extraction_mode)
            matching_images.append(current_images)
        return matching_images

    def write_images(self,
                     output_path: str,
                     extraction_mode: int = RAW_IMG,
                     cluster_in_subdirs: bool = False):
        """
        Locate all images represented by current instance
        and write them to specified output folder.
        :param output_path: str - Path to output folder.
        :param extraction_mode: int - Specify how
            represented images should be extracted.
            Options:
                RAW_IMG = 0: Get original image.
                DRAW_BOX = 1: Draw bounding box around recognized face.
                CUT_OUT_BOX = 2: Cut recognized face out of image.
            Default: RAW_IMG.
        :param cluster_in_subdirs: bool - Indicate if multiple
            images associated to the same data point should
            be stored inside a subdirectory (default: False).
        """
        # Create specified output folder (if necessary).
        os.makedirs(output_path, exist_ok=True)

        # Iterate through current data set vector by vector.
        for i in range(len(self.data_points)):
            # Reset output folder.
            current_output_folder = output_path
            # Get metadata structure and corresponding
            # pixel matrices of data point at current index.
            metadata = self.get_metadata_at_index(i)
            images = self.get_images_at_index(i, extraction_mode)
            # Eventually create subdirectory representing current cluster.
            if cluster_in_subdirs:
                current_output_folder = os.path.join(current_output_folder, "retrieved_cluster_{}".format(i))
                os.makedirs(current_output_folder, exist_ok=True)
            # Iterate through current metadata structure.
            for j, md in enumerate(metadata):
                # Get single metadata structure
                # and corresponding pixel matrix.
                img = images[j]
                # Get source file name.
                src_file_name = md[0]
                # Check if metadata structure belongs to an image or a video.
                if len(md) == 4:
                    # Current metadata entry belongs to a video file, so use
                    # corresponding frame index as part of the image file name.
                    src_frame_index = md[1]
                    top_left = md[2]
                    bottom_right = md[3]
                    img_path = os.path.join(current_output_folder,
                                            "{}_{}_{}-{}_{}-{}.jpg".format(src_file_name,
                                                                           src_frame_index,
                                                                           top_left[0], top_left[1],
                                                                           bottom_right[0], bottom_right[1]))
                elif len(md) == 3:
                    # Current metadata entry belongs to an image file.
                    top_left = md[1]
                    bottom_right = md[2]
                    img_path = os.path.join(current_output_folder,
                                            "{}_{}-{}_{}-{}.jpg".format(src_file_name,
                                                                        top_left[0], top_left[1],
                                                                        bottom_right[0], bottom_right[1]))
                else:
                    # Current metadata entry is unknown.
                    settings.LOGGER.warning("Skipped unknown metadata entry.\n{}".format(str(md)))
                    continue
                # Write image.
                cv2.imwrite(img_path, img)
