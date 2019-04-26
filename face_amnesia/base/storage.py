# coding=utf-8
#
# Vector storage module.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import os
import pickle

import numpy

from face_amnesia.base.data_set import DataSet

# File name extensions.
DATA_POINT_FILE_NAME_EXTENSION = ".npy"
METADATA_FILE_NAME_EXTENSION = ".dat"


def write_data_structure_to_file(file_path: str,
                                 data_structure,
                                 use_numpy: bool = False) -> bool:
    """
    Write given data structure to specified file.
    :param file_path: str - Path to file.
    :param data_structure: Data structure.
    :param use_numpy: bool - Indicate if NumPy
        package should be used to serialize
        data structure (default; False).
    :return: bool - True if data structure
        could be written and False otherwise.
    """
    fd = None
    success = False
    try:
        if use_numpy:
            numpy.save(file_path, data_structure)
        else:
            fd = open(file_path, "wb")
            pickle.dump(data_structure, fd)
        success = True
    except OSError:
        pass
    finally:
        if fd:
            fd.close()
        return success


def read_data_structure_from_file(file_path: str, use_numpy: bool = False):
    """
    Read data structure from specified file.
    :param file_path: str - Path to file.
    :param use_numpy: bool - Indicate if NumPy
        package should be used to deserialize
        data structure (default; False).
    :return: Retrieved data structure.
        Return None in case of an error.
    """
    fd = None
    data_structure = None
    try:
        fd = open(file_path, "rb")
        if use_numpy:
            data_structure = numpy.load(fd)
        else:
            data_structure = pickle.load(fd)
    except OSError:
        pass
    finally:
        if fd:
            fd.close()
        return data_structure


def write_data_set(storage_file_path: str, data_set: DataSet) -> bool:
    """
    Write given DataSet instance to specified storage
    folder by creating two separate files of the same
    name but with different extensions. Use specified
    file path and append ".npy" for matrix data and
    ".dat" for corresponding metadata before saving
    it to defined directory.
    :param storage_file_path: str - Path to storage file.
    :param data_set: DataSet - Data points and
        corresponding metadata structures.
    :return: bool - True if data set could be
        written and False otherwise.
    """
    # Check if specified path is not a string.
    if not isinstance(storage_file_path, str):
        return False

    # Check if specified path is empty.
    if storage_file_path == "":
        return False

    # Check if specified path is actually a directory.
    if os.path.isdir(storage_file_path):
        return False

    # Check if given data set is invalid.
    if not isinstance(data_set, DataSet):
        return False

    # Create path to data point file.
    matrix_file_path = storage_file_path + DATA_POINT_FILE_NAME_EXTENSION
    # Write data points (if possible).
    res = write_data_structure_to_file(matrix_file_path, data_set.data_points, use_numpy=True)
    if not res:
        return False

    # Create path to metadata file.
    metadata_file_path = storage_file_path + METADATA_FILE_NAME_EXTENSION
    # Write metadata (if possible).
    return write_data_structure_to_file(metadata_file_path, data_set.metadata)


def read_data_set(storage_file_path: str) -> DataSet:
    """
    Read DataSet instance from storage folder.
    Use given file path and append ".npy" for
    matrix data and ".dat" for corresponding
    metadata before reading it from defined
    directory.
    :param storage_file_path: str - Path to storage
        file (ignoring file name extension).
    :return: DataSet - Retrieved data points
        and corresponding metadata.
        Return empty DataSet instance
        in case of an error.
    """
    # Check if specified path is not a string.
    if not isinstance(storage_file_path, str):
        return DataSet(list(), list())

    # Check if specified path is actually a directory.
    if os.path.isdir(storage_file_path):
        return DataSet(list(), list())

    # Create path to data point file.
    data_point_file_path = storage_file_path + DATA_POINT_FILE_NAME_EXTENSION
    # Check if specified file exists.
    if not os.path.isfile(data_point_file_path):
        # Error: data points file does not exist,
        # therefore return empty DataSet instance.
        return DataSet(list(), list())
    # Read data points.
    data_points = read_data_structure_from_file(data_point_file_path, use_numpy=True)
    if data_points is None:
        # Error: data point file could not be read.
        return DataSet([], [])
    else:
        # Data points could be read,
        # so convert them to a list.
        data_points = list(data_points)

    # Create path to metadata file.
    metadata_file_path = storage_file_path + METADATA_FILE_NAME_EXTENSION
    # Read metadata (if possible).
    metadata = read_data_structure_from_file(metadata_file_path)
    if metadata is None:
        # Error: metadata file could not be read.
        return DataSet(data_points, list())

    # Return retrieved data set.
    return DataSet(data_points, metadata)


def get_all_data_set_file_paths(root_dir: str, excluded_dirs: list = []) -> list:
    """
    Get file paths of all data sets stored
    in specified root directory (or its
    subdirectories). Note that received paths
    are always relative to given root.
    :param root_dir: str - Path to directory.
    :param excluded_dirs: list(str) - Names
        of subdirectories which should
        be excluded (default: []).
    :return: list(str) - File paths to
        all data sets in specified
        directory (or its subdirectories).
    """
    # Create list to store data set file paths.
    data_set_file_paths = list()
    # Iterate through subdirectories of given root folder.
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Prune the search tree by removing subdirectories
        # marked for exclusion from list of current directory
        # names. Set topdown parameter to True in order
        # to make the search sensitive for such changes.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        # Search for matrix data and store
        # corresponding file path without extension.
        for f in files:
            if f.endswith(DATA_POINT_FILE_NAME_EXTENSION):
                path = os.path.join(root, os.path.splitext(f)[0])
                data_set_file_paths.append(path)
    return data_set_file_paths


def get_all_data_sets(root_dir: str) -> DataSet:
    """
    Retrieve all data points in specified
    directory (or its subdirectories).
    :param root_dir: str - Path to directory.
    :return: DataSet - Retrieved data points
        and corresponding metadata structures.
    """
    # Get all data set file paths in
    # specified directory (or its subdirectories).
    data_set_file_paths = get_all_data_set_file_paths(root_dir)
    # Retrieve corresponding data sets and
    # concatenate them to a single DataSet instance.
    new_data_set = DataSet([], [])
    for file_path in data_set_file_paths:
        current_data_set = read_data_set(file_path)
        new_data_set.add_data_set(current_data_set)
    return new_data_set
