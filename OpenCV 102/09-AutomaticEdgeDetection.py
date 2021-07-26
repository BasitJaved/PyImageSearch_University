# Thresholding and edge detection are both segmentation techniques, in thresholding our goal is to segment every single
# Pixel in the image into foreground or background, edge detection does not want to segment every single pixel
# it just tells us where the boarders of the objects are
import cv2 as cv
import argparse
import numpy as np
import glob


def auto_canny(img, sigma=0.4):
    # Compute Median of single channel pixel intensities
    v = np.median(img)

    # Apply automatic Canny Edge detection using Computed Median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv.Canny(img, lower, upper)

    return edge


# Construct the argument Parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='images', help='Path to input File')
args = vars(ap.parse_args())

# Loop over the images
for imagepath in glob.glob(args['image']+'/*.jpg'):
    # load image, convert it to gray scale and blur it slightly
    image = cv.imread(imagepath)
    cv.imshow('Original', image)
    cv.waitKey(0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)

    # Apply canny edge detection using wide, tight and automatically determined threshold
    wide = cv.Canny(blur, 10, 200)
    tight = cv.Canny(blur, 225, 250)
    auto = auto_canny(blur)
    cv.imshow('Edges', np.hstack([wide, tight, auto]))
    cv.waitKey(0)
