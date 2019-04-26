# coding=utf-8
#
# Image processing module.
#
# Assume an image to be stored as a NumPy matrix
# whose entries contain the color encoding of the
# pixel in corresponding position. Each color is
# represented by 3 values (using 8 bit unsigned
# integers as specified in settings module) matching
# RGB or BGR values of specified pixel (therefore
# ignoring the alpha channel). This can be seen as a
# 3D matrix whose third dimension contains 3 values.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import imghdr

import cv2
import dlib
import numpy

from face_amnesia import settings
from face_amnesia.utils import io

# Drawing constants.
BOUNDING_BOX_LINE_WIDTH = 4
BOUNDING_BOX_COLOR = (0, 0, 255)
BOUNDING_BOX_FONT = cv2.FONT_HERSHEY_PLAIN


def is_valid_image_file(img_file: str) -> bool:
    """
    Check if given file is a valid image and can be read by current user.
    :param img_file: str - Path to image file.
    :return: bool - True if file is a valid and readable image and False otherwise.
    """
    # Check if given path actually points
    # to a file readable by current user.
    if not io.is_readable_file(img_file):
        # Error: could not read image file.
        return False

    # Check if file is actually an image.
    if not imghdr.what(img_file):
        # Error: file is not an image.
        return False

    # No errors could be found.
    return True


def is_valid_pixel_matrix(img: numpy.ndarray) -> bool:
    """
    Check if given pixel matrix is valid.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: bool - True if pixel matrix is valid and False otherwise.
    """
    # Check pixel matrix structure.
    if not isinstance(img, numpy.ndarray):
        return False

    # Check matrix shape and entry data type.
    return len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == settings.RGB_CHANNEL_VALUE_DATA_TYPE


def read_image_from_file(img_file: str) -> numpy.ndarray:
    """
    Read BGR pixel matrix from given image file.
    :param img_file: str - Path to image file.
    :return: numpy.ndarray - BGR pixel matrix of image.
        Return empty array in case of invalid data.
        Raises ValueError in case of an invalid image file.
    """
    # Check if given file is valid image
    # file and can be read by current user.
    if not is_valid_image_file(img_file):
        raise ValueError

    # Read image file.
    img_bgr = cv2.imread(img_file)
    # Check if retrieved pixel data is valid.
    if img_bgr is None:
        # Error: could not read image file.
        return numpy.empty(0)

    # Return pixel matrix.
    return img_bgr


def swap_color_encoding(img: numpy.ndarray) -> numpy.ndarray:
    """
    Turn color encoding around, i.e. convert from BGR to RGB (or vice versa).
    In other words, using this function twice restores original color encoding.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: numpy.ndarray - Converted pixel matrix of image.
        Raises a ValueError in case of an invalid pixel matrix.
    """
    # Check if given pixel matrix is valid.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_converted


def get_image_height(img: numpy.ndarray) -> int:
    """
    Get height of given image.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: Image height (in pixels).
        Raises a ValueError in case of an invalid pixel matrix.
    """
    # Check if given pixel matrix is valid.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    return img.shape[0]


def get_image_width(img: numpy.ndarray) -> int:
    """
    Get width of given image.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: Image width (in pixels).
        Raises a ValueError in case of an invalid pixel matrix.
    """
    # Check if given pixel matrix is valid.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    return img.shape[1]


def _adjust_coordinates(x: int, y: int, img: numpy.ndarray) -> tuple:
    """
    Adjust given pixel coordinates
    to size of specified image making
    sure that x / y coordinate is strictly
    smaller than width / height of image.
    :param y: int - y value of pixel coordinate.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: tuple(int, int) - Adjusted coordinates.
    """
    # x coordinate should be smaller than image width.
    img_width = get_image_width(img)
    if x >= img_width:
        x = img_width - 1
    # y coordinate should be smaller than image height.
    img_height = get_image_height(img)
    if y >= img_height:
        y = img_height - 1
    return x, y


def get_top_left(box: dlib.rectangle, img: numpy.ndarray) -> tuple:
    """
    Extract coordinates of bounding box's top left corner making
    sure that corresponding pixel lies within specified image.
    :param box: dlib.rectangle - Bounding box.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: tuple - x and y coordinates of top left corner.
        Raises ValueError in case of an invalid pixel matrix or bounding box.
    """
    # Check bounding box.
    if box is None:
        raise ValueError

    # Check pixel matrix.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    # Coordinate values should not be negative.
    x = max(0, box.left())
    y = max(0, box.top())

    # Coordinate values should not
    # exceed the image's width / height.
    return _adjust_coordinates(x, y, img)


def get_bottom_right(box: dlib.rectangle, img: numpy.ndarray) -> tuple:
    """
    Extract coordinates of bounding box's bottom right corner making
    sure that corresponding pixel lies within specified image.
    :param box: dlib.rectangle - Bounding box.
    :param img: numpy.ndarray - Pixel matrix of image.
    :return: tuple - x and y coordinates of bottom right corner.
        Raises ValueError in case of an invalid pixel matrix or bounding box.
    """
    # Check bounding box.
    if box is None:
        raise ValueError

    # Check pixel matrix.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    # Coordinate values should not be negative.
    x = max(0, box.right())
    y = max(0, box.bottom())

    # Coordinate values should not
    # exceed the image's width / height.
    return _adjust_coordinates(x, y, img)


def extract_bounding_box(img: numpy.ndarray,
                         top_left: tuple,
                         bottom_right: tuple) -> numpy.ndarray:
    """
    Cut specified bounding box out of given image.
    :param img: numpy.ndarray - Pixel matrix of image.
    :param top_left: tuple - Coordinates of bounding box's top left corner.
    :param bottom_right: tuple - Coordinates of bounding box's bottom right corner.
    :return: numpy.ndarray - Pixel matrix of cut out part of image.
        Raises ValueError in case of an invalid pixel matrix or bounding box.
    """
    # Check pixel matrix.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    # Check bounding box coordinates.
    if not top_left or not bottom_right:
        raise ValueError

    bounding_box = img[top_left[1]:(bottom_right[1] + 1), top_left[0]:(bottom_right[0] + 1)]
    return bounding_box


def draw_bounding_box(img: numpy.ndarray,
                      top_left: tuple,
                      bottom_right: tuple,
                      text: str = ""):
    """
    Draw specified bounding box and a description text into given image.
    :param img: numpy.ndarray - Pixel matrix of image.
    :param top_left: tuple - Coordinates of bounding box's top left corner.
    :param bottom_right: tuple - Coordinates of bounding box's bottom right corner.
    :param text: str - Description text (default: "").
    :return: Does not return anything but directly adjusts given pixel data.
        Raises ValueError in case of an invalid pixel matrix.
    """
    # Check pixel matrix.
    if not is_valid_pixel_matrix(img):
        raise ValueError

    # Check bounding box coordinates.
    if not top_left or not bottom_right:
        raise ValueError

    cv2.rectangle(img, top_left, bottom_right,
                  BOUNDING_BOX_COLOR, BOUNDING_BOX_LINE_WIDTH)
    cv2.putText(img, text, top_left,
                BOUNDING_BOX_FONT, 2,
                BOUNDING_BOX_COLOR, BOUNDING_BOX_LINE_WIDTH)
