#!/usr/bin/python3
# coding=utf-8
#
# Command-line interface to
# create face description vectors.
#
# Copyright: 2019 Christian Hecktor (christian.hecktor@arcor.de).
# Licence: GNU General Public License v3.0.

import argparse
import os
import shutil

from face_amnesia import settings
from face_amnesia.base import storage
from face_amnesia.base.creation import Creation
from face_amnesia.processing import clustering
from face_amnesia.utils import benchmarking


def create(media_path: str,
           num_upsample: int,
           use_compression: bool,
           use_cnn: bool,
           use_overwrite: bool):
    """
    Main entry point of face
    description vector module.
    :param media_path: str - Path to
        image / video file or whole directory.
    :param num_upsample: int - Number of
        times the image resolution should
        be increased to find smaller faces.
    :param use_compression: bool - Indicate if
        data points of each input file
        should be compressed via clustering.
    :param use_cnn: bool - Indicate if CNN-based
        face detection model should be used.
    :param use_overwrite: bool - Indicate if already
        present data points of specified image
        or video files should be overwritten.
    """
    # Validate given input path.
    if not media_path:
        # Error: no input path specified.
        settings.LOGGER.error("No input path specified.")
        exit(1)

    processed_file_names = []
    if not use_overwrite:
        # Get name of all files already processed.
        processed_file_paths = storage.get_all_data_set_file_paths(settings.DATA_POINT_FOLDER_PATH)
        for path in processed_file_paths:
            processed_file_names.append(os.path.basename(path))

    # Initialize vector creator.
    creator = Creation(use_cnn=use_cnn,
                       use_68_points=True,
                       upsample=num_upsample)

    # Initialize list of file paths not yet processed.
    pending_file_paths = []

    # Process given input path.
    if os.path.isdir(media_path):
        # Given input path is a directory, so iterate
        # through all its subdirectories searching
        # for image or video files not yet processed.
        for root, _dirs, files in os.walk(media_path):
            for f in files:
                if f not in processed_file_names:
                    pending_file_paths.append(os.path.join(root, f))
    elif os.path.isfile(media_path):
        # Given path is a single file, so add
        # it to list of image or video file
        # paths if it is not yet processed.
        if os.path.basename(media_path) not in processed_file_names:
            pending_file_paths.append(media_path)
    else:
        # Error: given input path is invalid.
        settings.LOGGER.error("Input path is invalid: {}".format(media_path))
        exit(1)

    # Print corresponding info.
    if len(pending_file_paths) <= 0:
        print("Everything up-to-date.")
    else:
        print("Found new files not yet processed:")
        for file in pending_file_paths:
            print(file)
        print()

    for file in pending_file_paths:
        start = benchmarking.get_current_time_micros()
        data_set = creator.process_file(file)
        end = benchmarking.get_current_time_micros()
        print(file)
        print("Running time: {} ms.\n".format((end - start) / 1000))
        if len(data_set) <= 0:
            # Error: no vectors could be extracted.
            settings.LOGGER.warning("Could not extract any data points: {}".format(file))
        else:
            # If desired, compress resulting data set.
            if use_compression:
                data_set = clustering.cluster(data_set, settings.DISTANCE_THRESHOLD_CLUSTERING)
            # If necessary, copy current image or video
            # file to media file directory in order to
            # make sure that referenced file names in
            # metadata structures can be resolved.
            dst = os.path.join(settings.MEDIA_FOLDER_PATH, os.path.basename(file))
            if file != dst:
                shutil.copyfile(file, dst)
            # Write data set to data point folder.
            storage.write_data_set(os.path.join(settings.DATA_POINT_FOLDER_PATH, os.path.basename(file)), data_set)


if __name__ == "__main__":
    # Command-line argument parser.
    ap = argparse.ArgumentParser(description="Command-line interface to create face description vectors.")

    # Options.
    ap.add_argument("media_file",
                    type=str,
                    help="path to media file (image, video) or whole directory \
                    which should be used as input to create face description vectors"
                    )
    ap.add_argument("-u",
                    "--upsample",
                    type=int,
                    required=False,
                    default=0,
                    help="number of times the image or frame \
                    resolution should be increased to find smaller faces"
                    )
    ap.add_argument("--compress",
                    action='store_true',
                    required=False,
                    help="use clustering to compress retrieved data points of each input file"
                    )
    ap.add_argument("--cnn",
                    action='store_true',
                    required=False,
                    help="use CNN-based detection model to identify the position of faces"
                    )
    ap.add_argument("-o",
                    "--overwrite",
                    action='store_true',
                    required=False,
                    help="overwrite already present data points of specified image or video files"
                    )

    # Parse current arguments.
    args = vars(ap.parse_args())

    # Get input path.
    media_file = args.get("media_file")

    # Get upsampling parameter.
    upsample = args.get("upsample")

    # Get compression flag.
    compress = args.get("compress")

    # Get CNN flag.
    cnn = args.get("cnn")

    # Get force flag.
    overwrite = args.get("overwrite")

    # Call to main entry point.
    create(media_file, upsample, compress, cnn, overwrite)
