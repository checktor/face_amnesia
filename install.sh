#!/bin/bash
#
# Script to install pre-compiled versions
# of OpenCV and dlib (without GPU support).
#
# Tested with Ubuntu 18.04.
#
# Copyright: 2020 C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

# Update.
sudo apt-get update
sudo apt-get upgrade
sudo apt-get autoremove

# Install Python.
sudo apt-get install python3

# Install pip.
sudo apt-get install python3-pip

# Install NumPy.
pip3 install numpy

# Install scikit-learn.
pip3 install scikit-learn

# Install OpenCV without GUI functionality.
# Corresponding opencv_contrib package contains
# "extra" modules, i.e. modules that are unstable
# or not well-tested. They are not necessary
# to run the current version of this package.
pip3 install opencv-python-headless
# pip3 install opencv-contrib-python-headless

# Install OpenCV (and eventually its
# contrib modules) with GUI functionality.
# The current version of this package
# does not use any GUI functionality.
# pip3 install opencv-python
# pip3 install opencv-contrib-python

# Install cmake in order to build dlib.
sudo apt-get install cmake

# Install dlib.
pip3 install dlib
