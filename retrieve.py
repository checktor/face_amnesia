#!/usr/bin/python3
# coding=utf-8
#
# Command-line interface to retrieve
# previously stored data points.
#
# Copyright: C. Hecktor (checktor@posteo.de).
# Licence: GNU General Public License v3.0.

import argparse
import os

from face_amnesia import settings
from face_amnesia.base import retrieval
from face_amnesia.base.creation import Creation
from face_amnesia.utils import benchmarking

# Retrieval mode constants.
LINEAR = "linear"
LSH = "lsh"


def retrieve(media_path: str,
             retrieval_mode: str,
             extracted_img_path: str):
    """
    Main entry point of data point retrieval module.
    :param media_path: str - Path to image or video file.
    :param retrieval_mode: str - Retrieval mode.
    :param extracted_img_path: str - Path to folder
        in which extracted images should be stored.
    """
    # Validate given media file.
    if not media_path:
        # Error: no media file specified.
        settings.LOGGER.error("No image or video file specified.")
        return
    if os.path.isdir(media_path):
        # Error: specified file is actually a directory.
        settings.LOGGER.error("Given image or video file is actually a directory: {}".format(media_path))
        return

    # Initialize data point creator and extract
    # all description vectors of given image file.
    creator = Creation()
    query_data_set = creator.process_file(media_path)

    if retrieval_mode == LINEAR:
        # Use linear search to retrieve nearby data points.
        handler = retrieval.Linear(settings.DATA_POINT_FOLDER_PATH)
    elif retrieval_mode == LSH:
        # Use LSH to retrieve nearby data points.
        handler = retrieval.Lsh(settings.DATA_POINT_FOLDER_PATH)
    else:
        # Unknown retrieval mode, so fall back to default.
        settings.LOGGER.warning('Given retrieval mode is unknown, using "{}".'.format(LINEAR))
        handler = retrieval.Linear(settings.DATA_POINT_FOLDER_PATH)

    # Use current time in microseconds
    # as unique key of current retrieval.
    retrieval_key = str(benchmarking.get_current_time_micros())

    # Retrieve nearby data points for each vector in query data.
    for i in range(len(query_data_set)):
        vector = query_data_set.get_vector_at_index(i)
        start = benchmarking.get_current_time_micros()
        result_data_set = handler.query(vector, settings.DISTANCE_THRESHOLD_RECOGNITION)
        end = benchmarking.get_current_time_micros()
        print("Running time: {} ms.\n".format((end - start) / 1000))
        if extracted_img_path:
            # Write images referenced by resulting
            # metadata to specified folder (if necessary).
            if len(result_data_set) > 0:
                query_point_label = "query_point_{}".format(i)
                output_path = os.path.join(extracted_img_path, retrieval_key, query_point_label)
                print("Write images to {}".format(output_path))
                result_data_set.write_images(output_path, extraction_mode=1)
        else:
            # Write resulting metadata to standard output.
            print(result_data_set.get_metadata_as_string(retrieval_key))


if __name__ == "__main__":
    # Command-line argument parser.
    ap = argparse.ArgumentParser(description="Command-line interface to retrieve previously stored data points.")

    # Options.
    ap.add_argument("media_file",
                    type=str,
                    help="path to image or video file which should \
                    be used as reference to query similar data points"
                    )
    ap.add_argument("-m",
                    "--mode",
                    type=str,
                    required=False,
                    choices=[LINEAR, LSH],
                    default=LINEAR,
                    help="indicate which retrieval mode should be used (default: {})".format(LINEAR)
                    )
    ap.add_argument("-e",
                    "--extracted_img_folder",
                    type=str,
                    required=False,
                    help="path to folder in which extracted images or frames should be stored"
                    )

    # Parse current arguments.
    args = vars(ap.parse_args())

    # Get image path.
    media_file = args.get("media_file")

    # Get retrieval mode.
    mode = args.get("mode")

    # Get image extraction flag.
    extracted_img_folder = args.get("extracted_img_folder")

    # Call to main entry point.
    retrieve(media_file, mode, extracted_img_folder)
