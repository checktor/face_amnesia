# face_amnesia
Face detection and retrieval in image and video files.

## Requirements

### Language
* Python 3.5 (and later)

### Dependencies
* NumPy (http://www.numpy.org/)
    * basic numerical computations
* scikit-learn (https://scikit-learn.org/stable/)
    * PCA
* OpenCV (https://opencv.org/)
    * basic image and video processing
* dlib (http://dlib.net/)
    * face detection and recognition models

## Installation

### Use pre-compiled libraries
Run provided [install script](install.sh) to use pre-compiled versions
of OpenCV and dlib (without GPU support) available via pip. Note that
corresponding pre-built OpenCV package is unofficial but works fine.
See comments in install script for further details.

### Use Dockerfile
Run provided [Dockerfile](Dockerfile) to compile OpenCV and dlib by hand
using the official Ubuntu 18.04. image. Note that it is possible to enable
hardware-dependent optimization such as AVX instructions or CUDA support.
See comments in Dockerfile for further details.

## Testing
Run provided [test script](test.sh) to execute unit and integration tests.

## Usage
Command-line interfaces for data point creation and retrieval available
in project's root directory.

    cd /path/to/face_amnesia/

Data points and corresponding source files are stored in and served from
a separate `face amnesia` folder in current user's home directory which
may be identical to project's root directory.

### Data point creation
Create face description vectors from media file (image, video) or whole
directory.

    python3 create.py /path/to/media/file/or/directory/
    
Available options:
    
    python3 create.py --help

### Data point retrieval
Retrieve previously stored data points corresponding to faces similar
to given one.

    python3 retrieve.py /path/to/query/media/file/
    
Available options:
    
    python3 retrieve.py --help

## Further reading
* Mayur Datar, Nicole Immorlica, et. al.: "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" (2004)
* Chris Biemann: "Chinese Whispers - an Efficient Graph Clustering Algorithm and its Application to Natural Language Processing Problems" (2006)

## Further tools
* Adam Geitgey: face_recognition (https://github.com/ageitgey/face_recognition)

## Troubleshooting
Compiling dlib by hand is usually straightforward. Building OpenCV from
source, however, may need some adjustments. The official guides on this
topic are useful and can be found on the web:

* http://dlib.net/compile.html
* https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html

The following paragraphs will provide additional information on
compilation problems which may occur on some architectures.

### Compilation: CBLAS / LAPACK headers could not be found  
Possible workaround:

    sudo cp /usr/include/lapacke*.h /usr/include/openblas/
    
### Compilation: GCC version 8 is not supported
Possible workaround:

    export CC=/path/to/gcc-7/compiler

Building dlib's Python bindings with CUDA support needs an additional
argument to setup.py call:
    
    --set CUDA_HOST_COMPILER=/path/to/gcc-7/compiler
