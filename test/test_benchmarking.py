# coding=utf-8
#
# Unit test for benchmarking module.
#
# Copyright: 2020 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import unittest

from face_amnesia.utils import benchmarking


class TestBenchmarking(unittest.TestCase):

    def test_get_current_time_micros(self):
        first_time = benchmarking.get_current_time_micros()
        second_time = benchmarking.get_current_time_micros()
        self.assertIsInstance(first_time, int)
        self.assertIsInstance(second_time, int)
        self.assertTrue(first_time > 0)
        self.assertTrue(second_time > 0)
        self.assertTrue((second_time - first_time) > 0)
        self.assertTrue((first_time - second_time) < 0)
