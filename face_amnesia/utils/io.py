# coding=utf-8
#
# IO module.
#
# Copyright: 2020 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import os


def is_readable_file(file_path: str) -> bool:
    """
    Check if given file is valid and readable by current user.
    :param file_path: str - Path to file.
    :return: bool - True if file is valid and readable by current user and False otherwise.
    """
    # Check if specified file is valid.
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        # Error: file is not valid.
        return False

    # Check if file is readable.
    if not os.access(file_path, os.R_OK):
        # Error: file is not readable.
        return False

    # No errors could be found.
    return True
