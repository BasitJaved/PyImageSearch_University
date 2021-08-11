import argparse
import imutils
import cv2 as cv
import sys

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image containing ArUCo Tag')
ap.add_argument('-t', '--type', type=str, default='DICT_ARUCO_ORIGINAL', help='Type of ArUCo Tag to detect')
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

# Load input image from disk and resize it
print('Loading Image ...')
image = cv.imread(args['image'])
image = imutils.resize(image, width=600)

# Verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args['type'], None) is None:
    print(f'ArUCo Tag of {args["type"]} is not supported')
    sys.exit(0)

# load the ArUCO dictionary, grab ArUCo Parameters and detect the markers
print(f'Detecting {args["type"]} tags...')
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args['type']])
arucoParms = cv.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParms)

# Verify that atleast one ArUCo tag was detected
if len(corners)>0:
    # Flatten the ArUCo IDs list
    ids = ids.flatten()

    # Loop over the detected ArUCo cornets
    for (markerCorner, markerID) in zip(corners, ids):
        # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topleft, topright, bottomright, bottomleft) = corners

        # convert each (x, y) coordinate pairs into  integers
        topleft = (int(topleft[0]), int(topleft[1]))
        topright = (int(topright[0]), int(topright[1]))
        bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
        bottomright = (int(bottomright[0]), int(bottomright[1]))

        # draw the bounding box of the ArUCo detection
        cv.line(image, topleft, topright, (0, 255, 0), 2)
        cv.line(image, topright, bottomright, (0, 255, 0), 2)
        cv.line(image, bottomright, bottomleft, (0, 255, 0), 2)
        cv.line(image, bottomleft, topleft, (0, 255, 0), 2)

        # Compute and draw center x,y coordinates of ArUCo markets
        cX = int((topleft[0]+bottomright[0])/2.0)
        cY = int((topleft[1]+bottomright[1])/2.0)
        cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)

        # Draw ArUCo Marker ID on image
        cv.putText(image, str(markerID), (topleft[0], topleft[1]-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f'ArUCo marker ID {markerID}')

        # show image
        cv.imshow('Image', image)
        cv.waitKey(0)

cv.destroyAllWindows()


