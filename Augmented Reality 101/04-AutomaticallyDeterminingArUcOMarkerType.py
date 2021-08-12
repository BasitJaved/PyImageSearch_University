import argparse
import imutils
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image containing ArUCo Tag')
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

# Load image from disk and resize it
print('[INFO] Loading Image...')
image = cv.imread(args['image'])
image = imutils.resize(image, width=600)

# loop over they type of ArUCo Dictionaries
for (arucoName, arucoDict) in ARUCO_DICT.items():
    # Load ArUCo Dictionary, grab ArUCo parameters and attempt to detect markers for current Dictionary
    arucoDict = cv.aruco.Dictionary_get(arucoDict)
    arucoParam = cv.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParam)

    # if at least one ArUCo marker was detected display ArUCo name to terminal
    if len(corners) > 0:
        print(f'[Info] detected {len(corners)} for {arucoName}')
