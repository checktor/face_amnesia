# coding=utf-8
#
# Helper functions for unit test.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import os
import shutil
import stat


def create_non_readable_file(src_file_path: str, des_file_path: str):
    # Create a copy of specified source file.
    shutil.copyfile(src_file_path, des_file_path)
    # Make it non-readable.
    os.chmod(des_file_path, stat.S_IWUSR)


def create_non_writable_file(src_file_path: str, des_file_path: str):
    # Create a copy of specified source file.
    shutil.copyfile(src_file_path, des_file_path)
    # Make it non-writable.
    os.chmod(des_file_path, stat.S_IRUSR)
