# coding=utf-8
#
# Video processing module.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import shlex
import subprocess

import cv2
import numpy

from face_amnesia.utils import io


def is_valid_video_file(video_file: str) -> bool:
    """
    Check if given file is a valid
    video and can be read by current user.
    :param video_file: str - Path to video file.
    :return: bool - True if file is a valid
        and readable video and False otherwise.
        Raises subprocess.CalledProcessError
        if file format could not be examined.
    """
    # Check if given path actually points
    # to a file readable by current user.
    if not io.is_readable_file(video_file):
        # Error: could not read video file.
        return False

    # TODO: Video file names with whitespaces
    #  cannot be recognized as valid videos.
    # Run shell command to check if file is actually a video.
    raw_command = "file -i {}".format(video_file)
    converted_command = shlex.split(raw_command)
    result = subprocess.run(converted_command, stdout=subprocess.PIPE)
    # Check return code of shell command. If it is non-zero,
    # a subprocess.CalledProcessError is raised. Do not catch
    # this exception because it indicates a general problem with
    # the operating system which should be reported to the user.
    result.check_returncode()
    # Evaluate result given by standard output.
    result_string = str(result.stdout)
    first_line = result_string.split('\n')[0]
    video_type = first_line.split(' ')[1]
    if "video" not in video_type:
        return False

    # Check if video file can be opened.
    cap = cv2.VideoCapture(video_file)
    res = cap.isOpened()
    # Close video file.
    cap.release()
    if res:
        # Video file could be opened.
        return True
    else:
        # Error: video file could not be opened.
        return False


def get_frame_rate(video_file: str) -> int:
    """
    Get frame rate of given video file.
    :param video_file: str - Path to video file.
    :return: int - Frame rate of given video file.
        Return 0 if frame rate could not be determined.
        Raises ValueError if video file is invalid.
    """
    # Check if given video file is valid and readable.
    if not is_valid_video_file(video_file):
        raise ValueError

    # Open video file.
    cap = cv2.VideoCapture(video_file)
    # Get frame rate.
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    # Close video file.
    cap.release()
    # Return frame rate.
    return frame_rate


def read_frame_from_file(video_file: str, frame_index: int) -> numpy.ndarray:
    """
    Read BGR pixel data of specified frame from given video file.
    :param video_file: str - Path to video file.
    :param frame_index: int - Index of desired frame.
    :return: numpy.ndarray - BGR pixel matrix of specified frame.
        Return empty array if specified frame does not exist or cannot be read.
        Raises ValueError if video file is invalid.
    """
    # Check if given video file is valid and readable.
    if not is_valid_video_file(video_file):
        raise ValueError

    # Open video file.
    cap = cv2.VideoCapture(video_file)
    # Check if given frame index is negative.
    if frame_index < 0:
        return numpy.empty(0)
    # Jump to position in video
    # specified by given frame index.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # Read corresponding frame.
    ret, frame = cap.read()
    # Close video file.
    cap.release()
    if ret:
        # Specified frame could be retrieved.
        return frame
    else:
        # Error: frame could not be retrieved.
        return numpy.empty(0)
