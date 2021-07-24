# Thresholding is the binarization of an image.
# we seek to convert a grayscale image to a binary image, where the pixels are either 0 or 255.
# A simple thresholding example would be selecting a threshold value T, and then setting all
# pixel intensities less than T to 0, and all pixel values greater than T to 255. In this way,
# we are able to create a binary representation of the image.

import cv2 as cv
import argparse

# Construct the Argument Parser and Parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='coin5.jpg', help='Path to input image')
args = vars(ap.parse_args())

# read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# Convert the image to Gray Scale and blur it slightly
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (7, 7), 0)
cv.imshow('Blurred Image', blur)
cv.waitKey(0)

# Apply Basic thresholding, first parameter will be image, second is our threshold
# check if pixel value is greater then our threshold (200) we set it to be white (255) otherwise it
# will be black (0)
# Threshold is selected by hit and trial need to find what will be best value for our particular project
(T, thresh) = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
cv.imshow('THRESH BINARY', thresh)
cv.waitKey(0)

# Apply Basic thresholding, first parameter will be image, second is our threshold
# check if pixel value is greater then our threshold (200) we set it to be black otherwise white
# this is inverse of thresh_binary
(T, threshInv) = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
cv.imshow('THRESH BINARY INV', threshInv)
cv.waitKey(0)

# Visualize masked region
masked = cv.bitwise_and(image, image, mask=threshInv)
cv.imshow('Masked', masked)
cv.waitKey(0)

# Applying Otsu's automatic thresholding which automatically determines best threshold value
(T, otsuthreshInv) = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow('Otsu THRESH BINARY INV', otsuthreshInv)
print(f'Otsu thresholding Value is {T}')
cv.waitKey(0)

# Visualize masked region
masked2 = cv.bitwise_and(image, image, mask=otsuthreshInv)
cv.imshow('Masked2', masked2)
cv.waitKey(0)
