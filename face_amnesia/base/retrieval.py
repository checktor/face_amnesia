# coding=utf-8
#
# Data point retrieval module.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
import sys
from abc import ABC, abstractmethod

import numpy
from sklearn.decomposition import PCA

from face_amnesia import settings
from face_amnesia.base import storage
from face_amnesia.base.data_set import DataSet
from face_amnesia.processing import comparison, deduplication
from face_amnesia.utils import benchmarking, calculation

# File and folder name patterns.
LINEAR_STRUCTURE_FOLDER_NAME = "linear"
LSH_STRUCTURE_FOLDER_NAME = "lsh"
LINEAR_FILE_NAME_PATTERN = "data"
PARAMETER_FILE_NAME_PATTERN = "parameter"
LSH_TABLE_NAME_PATTERN = "table"
LSH_BUCKET_NAME_PATTERN = "bucket"

# Number of hash tables and
# hash functions per table used
# for parameter optimization.
MIN_NUM_HASH_FUNCTIONS = 20
MIN_NUM_HASH_TABLES = 10

# Error messages.
DATA_SET_INVALID = "data set is invalid"
QUERY_VECTOR_EMPTY_OR_INVALID = "query vector is empty or invalid"
DATA_DIMENSION_NOT_MATCHING = "data dimension is not matching"

# Currently, "linear" and "lsh" subdirectories
# are used to store internal information. Therefore,
# it should be excluded from eventual append_folder
# queries of other retrieval structure instances.
EXCLUDED_STRUCTURE_DIRS = [LINEAR_STRUCTURE_FOLDER_NAME,
                           LSH_STRUCTURE_FOLDER_NAME]


def filter_data_set(data_set: DataSet,
                    query_point: numpy.ndarray,
                    distance_threshold: float) -> DataSet:
    """
    Filter given DataSet instance by only keeping data points
    whose distance to query vector is below specified threshold.
    :param data_set: DataSet - Data points and corresponding metadata structures.
    :param query_point: numpy.ndarray - Query vector.
    :param distance_threshold: float - Maximum distance to query point.
    :return: DataSet - Data points and corresponding metadata
        structures within specified distance to query point.
    """
    # Create new DataSet to store filtered data
    # points and corresponding metadata structures.
    new_data_set = DataSet([], [])

    # Extract each vector from specified data set
    # and check its distance to query point. If it
    # falls below given threshold, store it as a result.
    for i in range(len(data_set)):
        vector, metadata = data_set.get_data_at_index(i)
        distance = calculation.get_distance(query_point, vector)
        if numpy.less_equal(distance, distance_threshold):
            new_data_set.add_data_point(vector, metadata)

    # Return new DataSet instance.
    return new_data_set


def adjust_data_set_file_path(file_path: str) -> str:
    """
    Check and adjust given data set file path to be
    usable with current retrieval structure (if necessary).
    :param file_path: str - Path to data set file.
    :return: str - Adjusted data set file path.
    """
    path_and_ext = os.path.splitext(file_path)
    is_data_points_file = path_and_ext[1] == storage.DATA_POINT_FILE_NAME_EXTENSION
    is_metadata_file = path_and_ext[1] == storage.METADATA_FILE_NAME_EXTENSION
    if is_data_points_file or is_metadata_file:
        return path_and_ext[0]
    else:
        return file_path


class Base(ABC):
    """Abstract base class to retrieve vectors."""

    def __init__(self, storage_path: str):
        """
        Base function to initialize retrieval structure.
        :param storage_path: str - Path to storage folder.
        """
        # Use given storage path as root directory of
        # current retrieval structure. Process data
        # points in-memory in case of an invalid argument.
        if isinstance(storage_path, str) and storage_path != "":
            # Storage folder is provided and valid,
            # therefore handle data points out-of-core.
            os.makedirs(storage_path, exist_ok=True)
            self.storage_root_dir_path = storage_path
            self.in_memory = False
        else:
            # Storage folder is not provided or invalid,
            # therefore handle data points in-memory.
            self.storage_root_dir_path = ""
            self.in_memory = True

        # Initialize data
        # dimension with 0.
        self.dimension = 0

    def get_root_dir_path(self) -> str:
        """
        Get path to root directory
        of current retrieval structure.
        :return: str - Path to root directory
            of current retrieval structure.
        """
        return self.storage_root_dir_path

    @abstractmethod
    def write_parameters_to_file(self) -> bool:
        """
        Base function to write current
        retrieval structure parameters to file.
        :return: bool - True if parameters
            could be written and False otherwise.
        """
        pass

    @abstractmethod
    def read_parameters_from_file(self) -> bool:
        """
        Base function to read retrieval
        structure parameters from file.
        :return: bool - True if parameters
            could be read and False otherwise.
        """
        pass

    @abstractmethod
    def append(self,
               data_set: DataSet,
               source_file_path: str = ""):
        """
        Base function to append given data set
        to currently managed retrieval structure.
        Indicate path to corresponding source
        file (if known) to avoid multiple insertions
        of data points from the same source.
        :param data_set: DataSet - Data points
            and corresponding metadata structures.
        :param source_file_path: str - Path to source
            file of given data set (default: "").
        """
        pass

    def append_file(self, file_path: str):
        """
        Base function to append data
        set from specified file.
        :param file_path: str - Path to data set file.
        :raise ValueError: In case
            specified file path is invalid
            or does not contain valid data.
        """
        # Check if specified file path is invalid.
        if not isinstance(file_path, str):
            raise ValueError("file path is invalid")

        # Check and adjust given data
        # set file path (if necessary).
        file_path = adjust_data_set_file_path(file_path)

        # Read provided data and append
        # it to current retrieval structure.
        new_data_set = storage.read_data_set(file_path)
        if len(new_data_set) > 0:
            self.append(new_data_set, file_path)
        else:
            raise ValueError("file does not contain valid data")

    def append_folder(self, folder_path: str):
        """
        Base function to append all
        data sets in specified folder.
        :param folder_path: str - Path
            to data set folder.
        :raise ValueError: In case
            specified folder path is invalid
            or does not contain valid data.
        """
        # Check if folder path is invalid.
        if not isinstance(folder_path, str) or not os.path.isdir(folder_path):
            raise ValueError("folder path is invalid")

        # Get all data set
        # files in given folder.
        data_set_file_paths = storage.get_all_data_set_file_paths(folder_path, EXCLUDED_STRUCTURE_DIRS)
        for file_path in data_set_file_paths:
            self.append_file(file_path)

    @abstractmethod
    def query(self,
              query_point: numpy.ndarray,
              distance_threshold: float) -> DataSet:
        """
        Base function to query currently available data points
        in order to find those within specified distance threshold.
        :param query_point: numpy.ndarray - Query vector.
        :param distance_threshold: float - Maximum distance to query point.
        :return: DataSet - Data points and corresponding metadata
            structures within specified distance to query point.
        """
        pass


class Linear(Base):
    """Concrete class to retrieve vectors using linear search."""

    def __init__(self, storage_path: str = ""):
        """
        Initialize linear retrieval module.
        :param storage_path: str - Path to
            storage folder. Manage data points
            in main memory if given path is
            invalid or empty (default: "").
        """
        # Call to base constructor.
        super().__init__(storage_path)

        # Initialize cache structure to easily
        # recognize already appended files.
        self.file_name_cache = set()

        # Initialize attribute to store
        # data set paths already appended.
        self.file_path_storage = None

        # Initialize attribute to
        # store data points in-memory.
        self.data_point_storage = None

        # Read parameter file (if necessary).
        if not self.in_memory:
            res = self.read_parameters_from_file()
            if not res:
                # There is no matching parameter
                # file in specified folder, so
                # handle its content as input data.
                self.append_folder(self.storage_root_dir_path)

    def get_structure_dir_path(self) -> str:
        """
        Get path to directory containing the
        data structure actually used for retrieval.
        :return: str - Path to data structure
            actually used for retrieval.
        """
        return os.path.join(self.storage_root_dir_path, LINEAR_STRUCTURE_FOLDER_NAME)

    def get_parameter_file_path(self) -> str:
        """
        Get path to parameter file of
        linear retrieval structure.
        :return: str - Path to parameter file
            of current retrieval structure.
        """
        return os.path.join(self.get_root_dir_path(),
                            "{}_{}{}".format(LINEAR_STRUCTURE_FOLDER_NAME,
                                             PARAMETER_FILE_NAME_PATTERN,
                                             storage.METADATA_FILE_NAME_EXTENSION))

    def write_parameters_to_file(self) -> bool:
        """
        Write current linear
        structure parameters to file.
        :return: bool - True if parameters
            could be written and False otherwise.
        """
        params = (self.dimension,
                  self.file_path_storage)
        return storage.write_data_structure_to_file(self.get_parameter_file_path(), params)

    def read_parameters_from_file(self) -> bool:
        """
        Read linear retrieval structure parameters from file.
        :return: bool - True if parameters
            could be read and False otherwise.
        """
        # Read parameters from file.
        params = storage.read_data_structure_from_file(self.get_parameter_file_path())
        # Check if retrieved parameters are
        # valid for linear retrieval structure.
        if params is not None and len(params) == 2:
            self.dimension = params[0]
            self.file_path_storage = params[1]
            self.file_name_cache = set([os.path.basename(path) for path in self.file_path_storage])
            return True
        else:
            return False

    def append(self,
               data_set: DataSet,
               source_file_path: str = ""):
        """
        Append given data set to currently managed linear
        retrieval structure. Indicate path to corresponding
        source file (if known) to avoid multiple insertions
        of data points from the same source. Accordingly, do
        nothing if given data is empty or already present.
        :param data_set: Data points and
            corresponding metadata structures.
        :param source_file_path: str - Path to
            source file of given data set (default: "").
        :raise ValueError: In case given
            data set is invalid.
        """
        # Check if provided data set is invalid.
        if not isinstance(data_set, DataSet):
            raise ValueError(DATA_SET_INVALID)

        # Check if provided data set is empty.
        if len(data_set) <= 0:
            return

        # Check if specified file
        # has already been appended.
        file_name = os.path.basename(source_file_path)
        if file_name in self.file_name_cache:
            return

        # Check if data points are added for the first
        # time. If so, create corresponding data structure.
        if self.dimension <= 0:
            # Update dimension attribute
            # according to given data.
            self.dimension = data_set.data_points[0].shape[0]
            if self.in_memory:
                # Create empty DataSet instance to
                # store data points in main memory.
                self.data_point_storage = DataSet([], [])
            else:
                # Create folder to store data
                # points in secondary memory.
                os.makedirs(self.get_structure_dir_path(), exist_ok=True)
                # Create list to store paths
                # of appended data files.
                self.file_path_storage = []

        # Append given data points to currently
        # managed data set only if its dimension matches.
        if self.dimension == data_set.data_points[0].shape[0]:
            # Data dimension is matching.
            if self.in_memory:
                # Append given data set
                # to main memory structure.
                self.data_point_storage.add_data_set(data_set)
                # Update file name cache (if necessary).
                if source_file_path:
                    self.file_name_cache.add(os.path.basename(source_file_path))
            else:
                if source_file_path:
                    # Source file path is provided,
                    # so simply add its name to file
                    # name cache and its corresponding
                    # absolute path to file path storage.
                    self.file_name_cache.add(os.path.basename(source_file_path))
                    self.file_path_storage.append(os.path.abspath(source_file_path))
                else:
                    # Source file path is not provided,
                    # so write given data set to a corresponding
                    # file in current retrieval structure folder.
                    num = len(storage.get_all_data_set_file_paths(self.get_structure_dir_path()))
                    file_name = "{}_{}".format(LINEAR_FILE_NAME_PATTERN, str(num))
                    file_path = os.path.join(self.get_structure_dir_path(), file_name)
                    storage.write_data_set(file_path, data_set)
                    # Update file name cache
                    # and file path storage.
                    self.file_name_cache.add(file_name)
                    self.file_path_storage.append(file_path)
                # Write updated parameters to file.
                self.write_parameters_to_file()
        else:
            # Data dimension is not matching.
            raise ValueError(DATA_DIMENSION_NOT_MATCHING)

    def _query_main_memory(self,
                           query_point: numpy.ndarray,
                           distance_threshold: float) -> DataSet:
        """
        Query data points managed in main memory to get
        all vectors within given distance to query point.
        :param query_point: numpy.ndarray - Query vector.
        :param distance_threshold: float - Maximum
            distance to query point.
        :return: DataSet - Data points and corresponding metadata
            within specified distance to query point.
            Return empty data set in case of an error.
        """
        # Create new DataSet to store nearby data
        # points and corresponding metadata structures.
        new_data_set = DataSet([], [])

        if self.data_point_storage is None:
            # No data points available,
            # so return an empty data set.
            return new_data_set

        # Filter current data set only keeping vectors
        # within given distance threshold to query point.
        return filter_data_set(self.data_point_storage, query_point, distance_threshold)

    def _query_secondary_memory(self,
                                query_point: numpy.ndarray,
                                distance_threshold: float) -> DataSet:
        """
        Query data points managed in secondary memory to
        get all vectors within given distance to query point.
        :param query_point: numpy.ndarray - Query vector.
        :param distance_threshold: float - Maximum distance to query point.
        :return: DataSet - Data points and corresponding metadata
            within specified distance to query point.
            Return empty data set in case of an error.
        """
        # Create new DataSet to store nearby data
        # points and corresponding metadata structures.
        new_data_set = DataSet([], [])

        if self.file_path_storage is None:
            # No data points available,
            # so return an empty data set.
            return new_data_set

        # Extract every data point from specified files
        # and check its distance to query point. If it falls
        # below given distance threshold, store it as a result.
        for file_path in self.file_path_storage:
            current_data_set = storage.read_data_set(file_path)
            filtered_data_set = filter_data_set(current_data_set, query_point, distance_threshold)
            new_data_set.add_data_set(filtered_data_set)

        # Return new DataSet instance.
        return new_data_set

    def query(self,
              query_point: numpy.ndarray,
              distance_threshold: float) -> DataSet:
        """
        Query current data set in order to find data
        points within specified distance threshold.
        :param query_point: numpy.ndarray - Query vector.
        :param distance_threshold: distance_threshold: float - Maximum distance to query point.
        :return: DataSet - Data points and corresponding metadata
            structures within specified distance to query point.
        """
        # Check if given query vector is invalid.
        if not isinstance(query_point, numpy.ndarray) or query_point.size <= 0:
            raise ValueError(QUERY_VECTOR_EMPTY_OR_INVALID)

        if self.in_memory:
            return self._query_main_memory(query_point, distance_threshold)
        else:
            return self._query_secondary_memory(query_point, distance_threshold)


class Lsh(Base):
    """
    Concrete class to retrieve vectors using LSH structure
    optimized for 1- or 2-norm as distance measure. Is
    based on projections of given data points on random
    vectors and cutting corresponding line into parts
    of same length to define hash buckets.
    See Mayur Datar, Nicole Immorlica, et. al.:
    "Locality-Sensitive Hashing Scheme Based on
    p-Stable Distributions" (2004) for details.
    """

    def __init__(self,
                 storage_path: str = "",
                 use_pca: bool = False,
                 num_hash_functions: int = 0,
                 num_hash_tables: int = 0,
                 bucket_width: float = 0):
        """
        Initialize LSH retrieval module. Use
        default parameters provided in settings
        module in case of missing or invalid arguments.
        :param storage_path: str - Path to
            storage folder. Manage data points
            in main memory if given path is
            invalid or empty (default: "").
        :param use_pca: bool - Indicate if PCA
            should be used in order to compute
            projection vectors (default: False).
        :param num_hash_functions: int - Number
            of concatenated hash functions per
            hash table (corresponds to parameter
            k in corresponding LSH literature).
            Defaults to value specified in
            settings module.
        :param num_hash_tables: int - Number of
            independent hash tables (corresponds
            to parameter L in corresponding LSH
            literature). Defaults to value
            specified in settings module.
        :param bucket_width: float - Width of
            line section defining a single
            LSH value. Defaults to value
            specified in settings module.
        """
        # Call to base constructor.
        super().__init__(storage_path)

        # Store PCA flag.
        self.use_pca = use_pca

        # Store and validate number of hash functions.
        if num_hash_functions <= 0:
            # Number of hash functions is not provided
            # or invalid, so fall back to default.
            if self.use_pca:
                self.current_num_hash_functions = settings.NUM_PCA_HASH_FUNCTIONS
            else:
                self.current_num_hash_functions = settings.NUM_RANDOM_HASH_FUNCTIONS
        else:
            self.current_num_hash_functions = num_hash_functions

        # Store and validate number of hash tables.
        if num_hash_tables <= 0:
            # Number of hash tables is not provided
            # or invalid, so fall back to default.
            if self.use_pca:
                self.current_num_hash_tables = settings.NUM_PCA_HASH_TABLES
            else:
                self.current_num_hash_tables = settings.NUM_RANDOM_HASH_TABLES
        else:
            self.current_num_hash_tables = num_hash_tables

        # Store and validate bucket width.
        if numpy.less_equal(bucket_width, 0):
            # Bucket width is not provided or
            # invalid, so fall back to default.
            if self.use_pca:
                self.current_bucket_width = settings.PCA_BUCKET_WIDTH
            else:
                self.current_bucket_width = settings.RANDOM_BUCKET_WIDTH
        else:
            self.current_bucket_width = bucket_width

        # Initialize cache structure to easily
        # recognize already appended files.
        self.file_name_cache = set()

        # Initialize a list of projection
        # vectors. A single entry contains
        # one or multiple vectors whose
        # concatenation defines the hash
        # function for a single hash table.
        self.projection_vectors = []

        # Initialize a list of projection
        # offsets. A single entry contains
        # one or multiple floating-point
        # numbers defining an offset for
        # corresponding projection vector.
        self.projection_offsets = []

        # Initialize main memory hash tables with None.
        self.hash_tables = None

        # Read parameter file (if necessary):
        if not self.in_memory:
            res = self.read_parameters_from_file()
            if not res:
                # There is no matching parameter
                # file in specified folder, so
                # handle its content as input data.
                self.append_folder(self.storage_root_dir_path)

    def get_structure_dir_path(self) -> str:
        """
        Get path to directory containing the
        data structure actually used for retrieval.
        :return: str - Path to data structure
            actually used for retrieval.
        """
        return os.path.join(self.storage_root_dir_path, LSH_STRUCTURE_FOLDER_NAME)

    def get_parameter_file_path(self) -> str:
        """
        Get path to parameter file
        of LSH retrieval structure.
        :return: str - Path to parameter file
            of current retrieval structure.
        """
        return os.path.join(self.get_root_dir_path(),
                            "{}_{}{}".format(LSH_STRUCTURE_FOLDER_NAME,
                                             PARAMETER_FILE_NAME_PATTERN,
                                             storage.METADATA_FILE_NAME_EXTENSION))

    def write_parameters_to_file(self) -> bool:
        """
        Write current retrieval LSH
        structure parameters to file.
        :return: bool - True if parameters
            could be written and False otherwise.
        """
        params = (self.dimension,
                  self.file_name_cache,
                  self.projection_vectors,
                  self.projection_offsets,
                  self.current_num_hash_functions,
                  self.current_num_hash_tables,
                  self.current_bucket_width,
                  self.use_pca)
        return storage.write_data_structure_to_file(self.get_parameter_file_path(), params)

    def read_parameters_from_file(self) -> bool:
        """
        Read LSH retrieval structure parameters from file.
        :return: bool - True if parameters
            could be read and False otherwise.
        """
        # Read parameters from file.
        params = storage.read_data_structure_from_file(self.get_parameter_file_path())
        # Check if retrieved parameters are
        # valid for LSH retrieval structure.
        if params is not None and len(params) == 8:
            self.dimension = params[0]
            self.file_name_cache = params[1]
            self.projection_vectors = params[2]
            self.projection_offsets = params[3]
            self.current_num_hash_functions = params[4]
            self.current_num_hash_tables = params[5]
            self.current_bucket_width = params[6]
            self.use_pca = params[7]
            return True
        else:
            return False

    def _initialize_hash_tables(self, data_set: DataSet):
        """
        Initialize hash tables
        according to given data set.
        :param data_set: DataSet - Data points
            and corresponding metadata structures.
        """
        if self.in_memory:
            # Adjust number of hash tables and
            # hash functions per table to a
            # certain minimum value. This allows
            # to easily optimize parameter values
            # without having to recreate projection
            # vectors on each run.
            num_functions = max(self.current_num_hash_functions, MIN_NUM_HASH_FUNCTIONS)
            num_tables = max(self.current_num_hash_tables, MIN_NUM_HASH_TABLES)
            # Create corresponding number
            # of in-memory hash tables
            # implemented as a dictionary.
            for _i in range(num_tables):
                self.hash_tables.append(dict())
        else:
            num_functions = self.current_num_hash_functions
            num_tables = self.current_num_hash_tables

        if len(data_set) <= 1 or not self.use_pca:
            # Create dynamic hash tables based
            # on one or multiple random vectors.
            for _i in range(num_tables):
                # Create random vectors.
                random_vectors = calculation.get_random_vectors(num_functions, self.dimension)
                self.projection_vectors.append(random_vectors)
                # Create NumPy array containing
                # corresponding number of uniformly
                # distributed floats in the half-open
                # range [0, bucket_width). Note that
                # the original paper proposes to use
                # the closed range [0, bucket_width].
                # However, the half-open variant is
                # easily computed with NumPy and
                # should not make any difference.
                random_offsets = \
                    numpy.random.uniform(0, self.current_bucket_width, size=(1, num_functions))[0]
                self.projection_offsets.append(random_offsets)
        else:
            # Create static hash table based
            # on principal components of current
            # data set computed by scikit-learn's
            # PCA module. See
            # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            # for details.
            pca = PCA()
            pca.fit(data_set.data_points)
            pc = pca.components_
            # Adjust number of functions to number
            # of principal components (if necessary).
            if self.current_num_hash_functions > len(pc):
                settings.LOGGER.warning("Not enough principal components available, use {}.".format(len(pc)))
                self.current_num_hash_functions = len(pc)
            if num_functions > len(pc):
                num_functions = len(pc)
            # Extract corresponding number of vectors.
            chosen_pc = numpy.take(pc, [i for i in range(num_functions)], axis=0)
            for _i in range(num_tables):
                # Store principal components
                # as projection vectors.
                self.projection_vectors.append(chosen_pc)
                # Create corresponding number of random offsets.
                random_offsets = \
                    numpy.random.uniform(0, self.current_bucket_width, size=(1, num_functions))[0]
                self.projection_offsets.append(random_offsets)

        # Write current parameters to file (if necessary).
        if not self.in_memory:
            self.write_parameters_to_file()

    def _update_hash_tables(self,
                            num_hash_functions: int,
                            num_hash_tables: int):
        """
        Update currently used hash tables.
        :param num_hash_functions: int - New
            number of concatenated hash
            functions per hash table.
        :param num_hash_tables: int - New
            number of independent hash tables.
        """
        if self.in_memory:
            # Update number of hash functions.
            if num_hash_functions <= MIN_NUM_HASH_FUNCTIONS:
                self.current_num_hash_functions = num_hash_functions
            else:
                settings.LOGGER.warning(
                    "Cannot adjust number of hash functions, keep {}.".format(self.current_num_hash_functions))

            # Update number of hash tables.
            if num_hash_tables <= MIN_NUM_HASH_TABLES:
                self.current_num_hash_tables = num_hash_tables
            else:
                settings.LOGGER.warning(
                    "Cannot adjust number of hash tables, keep {}.".format(self.current_num_hash_tables))

            # Get all data points.
            current_data_set = DataSet([], [])
            for _hash, data in self.hash_tables[0].items():
                current_data_set.add_data_set(data)
            # Delete current hash tables.
            for table in self.hash_tables:
                table.clear()
            # Create new hash tables.
            self._add_to_hash_tables(current_data_set)
        else:
            # Out-of-core variant is not yet supported.
            settings.LOGGER.warning("Cannot adjust hash tables.")

    def update_params(self,
                      num_hash_functions: int,
                      num_hash_tables: int):
        """
        Replace current parameters with given
        ones and update hash tables correspondingly.
        Only change a parameter value if given update
        is valid.
        :param num_hash_functions: int - New
            number of concatenated hash
            functions per hash table.
        :param num_hash_tables: int - New
            number of independent hash tables.
        """
        # Check new number of hash functions.
        if num_hash_functions <= 0:
            num_hash_functions = self.current_num_hash_functions

        # Check new number of hash tables.
        if num_hash_tables <= 0:
            num_hash_tables = self.current_num_hash_tables

        self._update_hash_tables(num_hash_functions, num_hash_tables)

    def append(self,
               data_set: DataSet,
               source_file_path: str = ""):
        """
        Append given data set to currently managed LSH
        retrieval structure. Indicate path to corresponding
        source file (if known) to avoid multiple insertions
        of data points from the same source. Accordingly, do
        nothing if given data is empty or already present.
        :param data_set: DataSet - Data points
            and corresponding metadata structures.
        :param source_file_path: str - Path to
            source file of given data set (default: "").
        :raise ValueError: In case given
            data set is invalid.
        """
        # Check if provided data set is invalid.
        if not isinstance(data_set, DataSet):
            raise ValueError(DATA_SET_INVALID)

        # Check if provided data set is empty.
        if len(data_set) <= 0:
            return

        # Check if specified file
        # has already been appended.
        file_name = os.path.basename(source_file_path)
        if file_name in self.file_name_cache:
            return

        # Check if data points are added for the first
        # time. If so, create corresponding data structure.
        if self.dimension <= 0:
            # Update dimension attribute
            # according to given data.
            self.dimension = data_set.data_points[0].shape[0]
            if self.in_memory:
                # Create list to store
                # in-memory hash tables.
                self.hash_tables = []
            else:
                # Create hash tables in secondary
                # memory using different folders.
                os.makedirs(self.get_structure_dir_path(), exist_ok=True)
                for index in range(self.current_num_hash_tables):
                    path = self._get_path_to_hash_table_folder(index)
                    os.makedirs(path, exist_ok=True)
            # Update hash functions.
            self._initialize_hash_tables(data_set)

        # Append given data points to currently managed
        # hash tables only if its dimension matches.
        if self.dimension == data_set.data_points[0].shape[0]:
            self._add_to_hash_tables(data_set)
            if source_file_path:
                self.file_name_cache.add(os.path.basename(source_file_path))
        else:
            # Data dimension is not matching.
            raise ValueError(DATA_DIMENSION_NOT_MATCHING)

    def query(self,
              query_point: numpy.ndarray,
              distance_threshold: float) -> DataSet:
        """
        Query current hash tables in order to find
        data points within specified distance threshold.
        :param query_point: query_point: numpy.ndarray - Query vector.
        :param distance_threshold: float - Maximum distance to query point.
        :return: DataSet - Data points and corresponding metadata
            structures within specified distance to query point.
        """
        # Check if given query vector is invalid.
        if not isinstance(query_point, numpy.ndarray) or query_point.size <= 0:
            raise ValueError(QUERY_VECTOR_EMPTY_OR_INVALID)

        # Create empty data set to store retrieved vectors.
        retrieved_data_set = DataSet([], [])

        # Get hash values of current query
        # point according to all hash tables.
        hash_values = self._get_hash_values(query_point)

        # Retrieve data points in corresponding buckets.
        for hash_table_index, hash_value in enumerate(hash_values):
            current_data_set = self.read_from_bucket(hash_table_index, hash_value)
            retrieved_data_set.add_data_set(current_data_set)

        # Remove data points outside given
        # distance threshold and eventually
        # present duplicates. Ignore the latter
        # step in case of only one hash table.
        if self.in_memory:
            # In case data points are stored in-memory,
            # removing duplicates is much more efficient
            # than filtering the data set for nearby points.
            # Therefore, run deduplication before filtering.
            if self.current_num_hash_tables > 1:
                retrieved_data_set = deduplication.deduplicate(retrieved_data_set, sort_by_memory_address=True)
            retrieved_data_set = filter_data_set(retrieved_data_set, query_point, distance_threshold)
        else:
            # In case data points are stored out-of-core,
            # filtering is much more efficient than removing
            # duplicates. Therefore, run filtering first.
            retrieved_data_set = filter_data_set(retrieved_data_set, query_point, distance_threshold)
            # In an out-of-core situation, sorting by memory
            # address is not applicable to remove duplicates.
            # In this case, use sorting by vector entries.
            if self.current_num_hash_tables > 1:
                retrieved_data_set = deduplication.deduplicate(retrieved_data_set, sort_by_memory_address=False)

        return retrieved_data_set

    def _get_path_to_hash_table_folder(self, hash_table_index: int) -> str:
        """
        Get path to specified hash table folder.
        :param hash_table_index: int - Index of hash table.
        :return: str - Path to hash table folder.
        """
        return os.path.join(self.get_structure_dir_path(),
                            "{}_{}".format(LSH_TABLE_NAME_PATTERN, str(hash_table_index)))

    def _get_bucket_file_name(self, hash_value: str) -> str:
        """
        Get file name (without extension) of specified bucket.
        :param hash_value: str - Hash value of bucket.
        :return: str - Bucket file name (without extension).
        """
        bucket_file_name = "{}_{}".format(LSH_BUCKET_NAME_PATTERN, hash_value)
        return bucket_file_name

    def _get_path_to_bucket(self,
                            hash_table_index: int,
                            hash_value: str) -> str:
        """
        Get path to specified hash table bucket.
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: str - Path to hash table bucket.
        """
        return os.path.join(self._get_path_to_hash_table_folder(hash_table_index),
                            self._get_bucket_file_name(hash_value))

    def _get_hash_values(self, vector: numpy.ndarray) -> list:
        """
        Compute hash value of given vector for each hash table.
        :param vector: numpy.ndarray - Data point.
        :return: list(str) - Hash value of given vector concerning each hash table.
        """
        hash_values = []
        for i in range(self.current_num_hash_tables):
            # Create list to store string representation
            # of hash value received by each hash function.
            hash_components = []
            for j in range(self.current_num_hash_functions):
                # Compute hash value according
                # to currently chosen hash function.
                nominator = numpy.add(numpy.dot(self.projection_vectors[i][j], vector), self.projection_offsets[i][j])
                hash_int = int(numpy.divide(nominator, self.current_bucket_width))
                # Convert hash value integer
                # to corresponding string.
                if hash_int < 0:
                    hash_str = "n{}".format(str(abs(hash_int)))
                else:
                    hash_str = "p{}".format(str(hash_int))
                # Add current hash value string to list of components.
                hash_components.append(hash_str)
            # Concatenate list of hash components to single hash value.
            hash_values.append(''.join(hash_components))
        return hash_values

    def _write_to_internal_bucket(self,
                                  data_set: DataSet,
                                  hash_table_index: int,
                                  hash_value: str) -> bool:
        """
        Append given data points to specified in-memory hash table bucket.
        :param data_set: DataSet - Data points and corresponding metadata structures.
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: bool - True if data set could be written and False otherwise.
        """
        # Get specified in-memory hash table.
        hash_table = self.hash_tables[hash_table_index]

        # Create corresponding hash bucket (if necessary).
        if hash_value not in hash_table:
            hash_table[hash_value] = DataSet([], [])

        # Append given data set.
        hash_table[hash_value].add_data_set(data_set)
        return True

    def _write_to_external_bucket(self,
                                  data_set: DataSet,
                                  hash_table_index: int,
                                  hash_value: str) -> bool:
        """
        Append given data points to specified out-of-core hash table bucket.
        :param data_set: DataSet - Data points and corresponding metadata structures.
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: bool - True if data set could be written and False otherwise.
        """
        # Create path to specified hash
        # table bucket and read its content.
        bucket_path = self._get_path_to_bucket(hash_table_index, hash_value)
        bucket_content = self._read_from_external_bucket(hash_table_index, hash_value)

        # Append given data set.
        bucket_content.add_data_set(data_set)

        # Write current bucket back to secondary memory.
        res = storage.write_data_set(bucket_path, bucket_content)
        return res

    def write_to_bucket(self,
                        data_set: DataSet,
                        hash_table_index: int,
                        hash_value: str) -> bool:
        """
        Write given data points to specified hash
        table bucket overwriting eventually present data.
        :param data_set: DataSet - Data points and corresponding metadata structures.
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: bool - True if data set could be written and False otherwise.
        """
        if self.in_memory:
            return self._write_to_internal_bucket(data_set, hash_table_index, hash_value)
        else:
            return self._write_to_external_bucket(data_set, hash_table_index, hash_value)

    def _read_from_internal_bucket(self,
                                   hash_table_index: int,
                                   hash_value: str) -> DataSet:
        """
        Get all data points stored in specified hash
        table bucket (internally stored in main memory).
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: DataSet - Retrieved data points and corresponding metadata.
            Return empty DataSet instance in case of an error.
        """
        # Get specified hash table from main memory.
        hash_table = self.hash_tables[hash_table_index]
        # Check if specified bucket exists.
        if hash_value in hash_table:
            # Specified bucket exists,
            # so return its content.
            return hash_table[hash_value]
        else:
            # Specified bucket does not exist,
            # so return an empty DataSet instance.
            return DataSet([], [])

    def _read_from_external_bucket(self,
                                   hash_table_index: int,
                                   hash_value: str) -> DataSet:
        """
        Get all data points stored in specified hash table
        bucket (externally stored in secondary memory).
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: DataSet - Retrieved data points and corresponding metadata.
            Return empty DataSet instance in case of an error.
        """
        # Create path to specified hash table bucket.
        bucket_path = self._get_path_to_bucket(hash_table_index, hash_value)
        # Read and return corresponding DataSet instance.
        return storage.read_data_set(bucket_path)

    def read_from_bucket(self,
                         hash_table_index: int,
                         hash_value: str) -> DataSet:
        """
        Get all data points stored in specified hash table bucket.
        :param hash_table_index: int - Index of hash table.
        :param hash_value: str - Hash value of bucket.
        :return: DataSet - Retrieved data points and corresponding metadata.
            Return empty DataSet instance in case of an error.
        """
        if self.in_memory:
            return self._read_from_internal_bucket(hash_table_index, hash_value)
        else:
            return self._read_from_external_bucket(hash_table_index, hash_value)

    def _distribute_to_hash_values(self, data_set: DataSet) -> list:
        """
        Distribute data points to corresponding hash
        values in order to insert all data points
        belonging to the same hash bucket in one step.
        :param data_set: DataSet - Data points and
            corresponding metadata structures.
        :return: list(dict) - Each dictionary maps
            hash value to corresponding data points
            according to one specific hash table.
        """
        # Create temporary hash table structure.
        tmp_hash_tables = list()
        for _i in range(self.current_num_hash_tables):
            tmp_hash_tables.append(dict())

        # Insert each data point to each
        # temporary hash table structure.
        for i in range(len(data_set)):
            # Get current data point and corresponding metadata.
            vector, metadata = data_set.get_data_at_index(i)
            # Compute hash value concerning each hash table.
            hash_values = self._get_hash_values(vector)
            # Insert current vector into specified buckets.
            for j, hash_value in enumerate(hash_values):
                # Get current hash table.
                hash_table = tmp_hash_tables[j]
                # Create corresponding bucket (if necessary).
                if hash_value not in hash_table:
                    hash_table[hash_value] = DataSet([], [])
                # Insert vector and corresponding
                # metadata in current hash table.
                hash_table[hash_value].add_data_point(vector, metadata)

        # Return temporary hash table structure.
        return tmp_hash_tables

    def _add_to_hash_tables(self, data_set: DataSet):
        """
        Add each data point in given
        data set to every hash table.
        :param data_set: DataSet - Data points
            and corresponding metadata structures.
        """
        # Cluster given data set in temporary hash table buckets.
        tmp_hash_tables = self._distribute_to_hash_values(data_set)

        # Append data points in each temporary
        # bucket to corresponding real bucket.
        for hash_table_index, tmp_hash_table in enumerate(tmp_hash_tables):
            for hash_value, tmp_bucket in tmp_hash_table.items():
                self.write_to_bucket(tmp_bucket, hash_table_index, hash_value)


def get_optimal_lsh_params(sample_data_set: DataSet,
                           query_data_set: DataSet,
                           bucket_width: float,
                           max_num_missing_elements: int,
                           distance_threshold: float) -> tuple:
    """
    Optimize parameters of LSH retrieval using given
    sample data set. Run linear retrieval on each of
    the query points in order to determine desired
    output. Then adjust LSH parameters to achieve the
    same result (considering specified number of
    allowed errors) with minimum possible effort.
    :param sample_data_set: DataSet - Sample data
        points and corresponding metadata structures.
    :param query_data_set: DataSet - Query data
        points and corresponding metadata structures.
    :param bucket_width: float -  Width of line
        section defining a single LSH value.
    :param max_num_missing_elements: int - Maximum
        number of missing elements in retrieved
        data points compared to desired output.
    :param distance_threshold: float - Maximum
        distance to query point.
    :return: tuple(int) - Optimal number
        of hash functions and hash tables.
    """
    # Use linear retrieval structure
    # to determine desired results.
    linear = Linear()
    linear.append(sample_data_set)
    desired_data_set = DataSet([], [])
    for i in range(len(query_data_set)):
        result_data_set = linear.query(query_data_set.get_vector_at_index(i), distance_threshold)
        desired_data_set.add_data_set(result_data_set)

    # Retrieval structure parameters which should be tested.
    num_hash_tables_options = numpy.arange(1, MIN_NUM_HASH_TABLES)
    num_hash_functions_options = numpy.arange(1, MIN_NUM_HASH_FUNCTIONS)

    # Optimal parameters found so far.
    optimal_num_hash_tables = -1
    optimal_num_hash_functions = -1
    optimal_running_time_estimate = sys.maxsize

    # Create LSH handler.
    lsh = Lsh(bucket_width=bucket_width)
    lsh.append(sample_data_set)

    for hash_tables in num_hash_tables_options:
        for hash_functions in num_hash_functions_options:
            lsh.update_params(hash_functions, hash_tables)
            current_running_times = []
            current_result_data_set = DataSet([], [])
            for i in range(len(query_data_set)):
                start = benchmarking.get_current_time_micros()
                result_data_set = lsh.query(query_data_set.get_vector_at_index(i), distance_threshold)
                end = benchmarking.get_current_time_micros()
                current_running_times.append(end - start)
                current_result_data_set.add_data_set(result_data_set)
            current_running_time_estimate = numpy.max(current_running_times)
            _num_irrelevant, num_missing = comparison.compare(current_result_data_set, desired_data_set)
            if num_missing <= max_num_missing_elements:
                if current_running_time_estimate < optimal_running_time_estimate:
                    optimal_num_hash_functions = hash_functions
                    optimal_num_hash_tables = hash_tables
                    optimal_running_time_estimate = current_running_time_estimate
    print("Width:", bucket_width)
    fd = open("result.txt", "a")
    fd.write("Bucket width: {}\n".format(str(bucket_width)))
    fd.write("Allowed error: {}\n".format(str(max_num_missing_elements)))
    fd.write("Optimal number of hash functions: {}\n".format(str(optimal_num_hash_functions)))
    fd.write("Optimal number of hash tables: {}\n".format(str(optimal_num_hash_tables)))
    fd.write("Optimal running time estimate: {}\n\n".format(str(optimal_running_time_estimate)))
    fd.close()
    return optimal_num_hash_functions, optimal_num_hash_tables
