# coding=utf-8
#
# Mathematical module.
#
# Use available NumPy functionality
# to process vector and matrix data.
# Note that NumPy creates deep copies
# of given arrays and matrices if a
# conversion to its inherent data
# structure is necessary. Therefore,
# the data returned by provided
# functions are deep copies if
# corresponding inputs are not
# provided as NumPy arrays.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import numpy

from face_amnesia import settings


def get_norm(matrix: numpy.ndarray, order: int = 2):
    """
    Compute norm according to each row of given matrix.
    :param matrix: numpy.ndarray - Data matrix
        storing one data point per row.
    :param order: Indicate which norm should
        be used. For order = 1 the 1-norm is
        used, whereas order = 2 leads to a
        2-norm (Euclidean norm).
    :return: Norm according to rows of given matrix.
    :raise TypeError: In case matrix data type is invalid.
    :raise ValueError: In case matrix is empty, has
        the wrong shape or contains invalid entries.
    """
    # Convert given matrix to NumPy structure (if necessary)
    # which implicitly creates a deep copy. May raise a
    # ValueError or a TypeError in case of an error.
    if not isinstance(matrix, numpy.ndarray):
        matrix = numpy.array(matrix, dtype=settings.VECTOR_ENTRY_DATA_TYPE)

    # Check if matrix is empty.
    if matrix.size < 1:
        raise ValueError

    # Check if norm is invalid.
    if order <= 0:
        raise ValueError

    # Check matrix dimension.
    dimension = len(matrix.shape)
    if dimension == 1:
        # Matrix is actually a vector.
        return numpy.linalg.norm(matrix, ord=order, axis=0)
    elif dimension == 2:
        # Matrix is two-dimensional.
        return numpy.linalg.norm(matrix, ord=order, axis=1)
    else:
        # Consider inputs of three or
        # more dimensions as invalid.
        raise ValueError


def get_distance(first_vector: numpy.ndarray,
                 second_vector: numpy.ndarray,
                 order: int = 2):
    """
    Compute distance between given vectors.
    :param first_vector: numpy.ndarray - Data point.
    :param second_vector: numpy.ndarray - Data point.
    :param order: int - Indicate which norm
        should be used to compute the distance.
        For order = 1 the 1-norm is used, whereas
        order = 2 leads to a 2-norm (Euclidean norm).
    :return: Distance between both vectors.
    :raise TypeError: In case vector data types are invalid.
    :raise ValueError: In case vectors are empty, have a
        wrong / non-matching shape or contain invalid entries.
    """
    # Convert first vector to NumPy structure (if necessary)
    # which implicitly creates a deep copy. May raise a
    # ValueError or a TypeError in case of an error.
    if not isinstance(first_vector, numpy.ndarray):
        first_vector = numpy.array(first_vector, dtype=settings.VECTOR_ENTRY_DATA_TYPE)

    # Convert second vector to NumPy structure (if necessary)
    # which implicitly creates a deep copy. May raise a
    # ValueError or a TypeError in case of an error.
    if not isinstance(second_vector, numpy.ndarray):
        second_vector = numpy.array(second_vector, dtype=settings.VECTOR_ENTRY_DATA_TYPE)

    # Check if vector shapes are non-matching.
    if first_vector.shape != second_vector.shape:
        raise ValueError

    # Compute Euclidean distance.
    return get_norm(numpy.subtract(first_vector, second_vector), order)


def get_mean(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Compute mean vector according
    to rows of given matrix.
    :param matrix: numpy.ndarray - Data matrix
        storing one data point per row.
    :return: numpy.ndarray - Mean vector
        according to rows of given matrix.
    :raise TypeError: In case matrix
        data type is invalid.
    :raise ValueError: In case vector
        is empty, has the wrong shape
        or contains invalid entries.
    """
    # Convert matrix to NumPy structure (if necessary)
    # which implicitly creates a deep copy. May raise a
    # ValueError or a TypeError in case of an error.
    if not isinstance(matrix, numpy.ndarray) or matrix.dtype != settings.VECTOR_ENTRY_DATA_TYPE:
        matrix = numpy.array(matrix, dtype=settings.VECTOR_ENTRY_DATA_TYPE)

    # Check if matrix is empty.
    if matrix.size < 1:
        raise ValueError

    # Check matrix dimension.
    dimension = len(matrix.shape)
    if dimension == 1:
        # Matrix is actually a vector.
        return matrix
    elif dimension == 2:
        # Matrix is two-dimensional.
        return numpy.mean(matrix, axis=0)
    else:
        # Consider inputs of three or
        # more dimensions as invalid.
        raise ValueError


def get_random_vectors(num_vectors: int,
                       dimension: int,
                       use_cauchy: bool = False) -> numpy.ndarray:
    """
    Get specified number of random vectors with
    given dimension. Use Cauchy or normal (Gaussian)
    distribution with the following parameters: a = 1
    in case of Cauchy and mean = 0 / standard deviation = 1
    in case of Gaussian distribution.
    :param num_vectors: int - Number of random vectors.
    :param dimension: int - Dimension of each vector.
    :param use_cauchy: bool - Indicate if a Cauchy
        distribution (in case of True) or a normal
        (Gaussian) distribution (otherwise) should
        be used (default: False).
    :return: numpy.ndarray - Matrix containing one
        random vector of corresponding size per row.
    """
    # Check if desired number of random
    # vectors is zero or negative.
    if num_vectors <= 0:
        raise ValueError

    # Check if desired dimension of
    # each vector is zero or negative.
    if dimension <= 0:
        raise ValueError

    shape = (num_vectors, dimension)
    if use_cauchy:
        # Use Cauchy distribution (a = 1).
        return numpy.random.standard_cauchy(size=shape)
    else:
        # Use Gaussian distribution (mean: 0, standard deviation: 1).
        return numpy.random.normal(size=shape)
