# Dockerfile using image of Ubuntu 18.04.
# and compiling OpenCV and dlib by hand.
#
# Tested with Docker 18.09.9.
#
# Copyright: 2020 Christian Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

# Use image of Ubuntu 18.04.
FROM ubuntu:18.04

# Use image of Ubuntu 18.04 with CUDA
# and cuDNN in order to use GPU support.
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install Ubuntu packages.
RUN apt-get update && \
	apt-get -y install \
		# Prerequisites.
		build-essential \
		cmake \
		wget \
		unzip \
		pkg-config \
		# Python.
		python3 \
		python3-dev \
		python3-pip \
		# Image processing libraries.
		libjpeg-dev \
		libpng-dev \
		libtiff-dev \
		# Video processing and streaming libraries.
		libxvidcore-dev \
		libx264-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libgstreamer-plugins-base1.0-dev \
		# GUI support (if necessary).
		# libgtk-3-dev \
		# Optimizations.
		libopenblas-dev \
		liblapacke-dev \
		libatlas-base-dev \
		gfortran \
		libtbb2 \
		libtbb-dev && \
	# Cleanup.
	apt-get clean && \
	rm -r /var/lib/apt/lists/*

# Install Python packages.
RUN pip3 install --no-cache-dir setuptools numpy scikit-learn

# Create folder to store external libraries.
RUN mkdir /home/lib

# Install OpenCV.
# ===============

# Get OpenCV.
RUN cd /home/lib/ && \
    wget https://github.com/opencv/opencv/archive/4.3.0.zip && \
	unzip 4.3.0.zip && \
	rm 4.3.0.zip

# Get OpenCV contrib modules (necessary to use GPU support).
# RUN cd /home/lib && \
# 	wget https://github.com/opencv/opencv_contrib/archive/4.3.0.zip && \
#	unzip 4.3.0.zip && \
#	rm 4.3.0.zip

# Build OpenCV.
RUN cd /home/lib/opencv-4.3.0/ && \
	mkdir build && \
	cd build/ && \
	cmake -DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DENABLE_PRECOMPILED_HEADERS=OFF ..
		# Add the following flags in order to enable GPU support.
        # -DWITH_CUDA=ON \
        # -DENABLE_FAST_MATH=1 \
        # -DCUDA_FAST_MATH=1 \
       	# -DWITH_CUBLAS=1 \
		# -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.3.0/modules

# Compile OpenCV.
RUN cd /home/lib/opencv-4.3.0/build/ && \
	# Adjust -j flag according to the
	# number of CPU cores available.
	make -j $(nproc) && \
	make install && \
	ldconfig

# Install dlib.
# =============

# Get dlib.
RUN cd /home/lib/ && \
	wget http://dlib.net/files/dlib-19.19.tar.bz2 && \
	tar -xjf dlib-19.19.tar.bz2 && \
	rm dlib-19.19.tar.bz2

# Build and compile dlib.
RUN cd /home/lib/dlib-19.19/ && \
	mkdir build && \
	cd build/ && \
	# In the following command, add "-DUSE_SSE2_INSTRUCTIONS=1",
	# "-DUSE_SSE4_INSTRUCTIONS=1" or "-DUSE_AVX_INSTRUCTIONS=1"
	# in case your CPU supports SSE2, SSE4 or AVX instructions
	# and "-DDLIB_USE_CUDA=1" if GPU support should be enabled.
	cmake .. && \
	cmake --build . --config Release && \
	cd .. && \
	# Analogous to above, append "--yes USE_SSE2_INSTRUCTIONS",
	# "--yes USE_SSE4_INSTRUCTIONS" or "--yes USE_AVX_INSTRUCTIONS"
	# to use SSE2, SSE4 or AVX instructions and "--yes DLIB_USE_CUDA"
	# to enable GPU support. However, on some systems these
	# options are on by default and need not to be provided.
	python3 setup.py install

# Add source data.
ADD . /home/face_amnesia
