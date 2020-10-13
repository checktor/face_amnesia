# coding=utf-8
#
# Benchmarking module.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import time


def get_current_time_micros() -> int:
    """
    Get current time in microseconds.
    :return: int - Current time in microseconds.
    """
    return int(round(time.time() * 1000000))
