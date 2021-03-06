import numpy as np
import argparse
import cv2 as cv
import sys

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='Path to output image containing ArUCo tag')
ap.add_argument('-i', '--id', type=int, required=True, help='ID of ArUCo Tag to generate')
ap.add_argument('-t', '--type', type=str, default='DICT_ARUCO_ORIGINAL', help='type of ArUCo tag to generate')
args = vars(ap.parse_args())

# Define name of each possible ArUCo tag OpenCV supports (21 below)
ARUCO_DICT = {"DICT_4X4_50": cv.aruco.DICT_4X4_50,
              "DICT_4X4_100": cv.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv.aruco.DICT_4X4_250,
              "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv.aruco.DICT_5X5_50,
              "DICT_5X5_100": cv.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv.aruco.DICT_5X5_250,
              "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv.aruco.DICT_6X6_50,
              "DICT_6X6_100": cv.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv.aruco.DICT_6X6_250,
              "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv.aruco.DICT_7X7_50,
              "DICT_7X7_100": cv.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv.aruco.DICT_7X7_250,
              "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11}

# verify that the supplied ArUCo exists and is supported by OpenCv
if ARUCO_DICT.get(args['type'], None) is None:
    print(f'ArUCo tag of {args["type"]} is not supported')
    sys.exit(0)

# Load ArUCo dictionary
arucodic = cv.aruco.Dictionary_get(ARUCO_DICT[args['type']])

# Allocate memory for output ArUCO tag and draw ArUCo tag on output image
print(f'Generating ArUCO Tag type {args["type"]} with ID {args["id"]}')
tag = np.zeros((300, 300, 1), dtype='uint8')
cv.aruco.drawMarker(arucodic, args['id'], 300, tag, 1)  # Last 1 is for padding it can be more then 1 but not less i.e 0

# Write Generated ArUCO tag on disk and display on screen
cv.imwrite(args['output'], tag)
cv.imshow('ArUCo Tag', tag)
cv.waitKey(0)
cv.destroyAllWindows()
