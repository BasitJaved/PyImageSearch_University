from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2 as cv
import sys


def find_color_card(image):
    # load ArUCo dictionary, grab ArUCo parameters and detect markers in input image
    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Try and extract coordinates of Color Correction card
    try:
        ids = ids.flatten()

        # extract top-left marker
        i = np.squeeze(np.where(ids==923))  # finding the index where id is 923
        topLeft = np.squeeze(corners[i])[0]  # using that index to find the corner

        # extract top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]

        # extract bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]

        # extract bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]

    # If we could not find the color correction card or any id in it we gracefully return
    except:
        return None

    # Build the list of reference points and apply a prospective transform to obtain a top-down
    # birds-eye-view of color matching card
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)

    # Return the color matching card to calling function
    return card

# Construct the argument Parser and Parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-r', '--reference', type=str, default='reference.jpg', help='Path to reference image')
ap.add_argument('-i', '--image', type=str, default='01.jpg', help='Path to input image to apply correction to')
args = vars(ap.parse_args())

# load reference and input image
ref = cv.imread(args['reference'])
img = cv.imread(args['image'])

# Resize the reference and input image
ref = imutils.resize(ref, width=500)
img = imutils.resize(img, width=500)

# Display reference and input image
cv.imshow('reference', ref)
cv.waitKey(0)
cv.imshow('image', img)
cv.waitKey(0)

# find the color matching card in image
print('[INFO] finding color matching cards...')
refCard = find_color_card(ref)
imgCard = find_color_card(img)

# if color matching card is not found in either reference image or input image, gracefully exit
if refCard is None or imgCard is None:
    print('[INFO] could not find color matching card')
    sys.exit(0)

# show the color matching card in reference image and input image respectively
cv.imshow('Reference Color Card', refCard)
cv.imshow('Input Color Card', imgCard)

# applying histogram matching from the color matching card in the reference image to the color matching card
# in input image
print('[INFO] matching images')
imgCard = exposure.match_histograms(imgCard, refCard, multichannel=True)

# show input color matching card after histogram matching
cv.imshow('Input Color Card after matching', imgCard)
cv.waitKey(0)
cv.destroyAllWindows()
