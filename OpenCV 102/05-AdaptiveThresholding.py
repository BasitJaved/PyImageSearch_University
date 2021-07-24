# Thresholding is the binarization of an image.
# we seek to convert a grayscale image to a binary image, where the pixels are either 0 or 255.
# A simple thresholding example would be selecting a threshold value T, and then setting all
# pixel intensities less than T to 0, and all pixel values greater than T to 255. In this way,
# we are able to create a binary representation of the image.
# adaptive thresholding considers a small set of neighboring pixels at a time,
# computes T for that specific local region, and then performs the segmentation.


import cv2 as cv
import argparse

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='jobs.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Read and Display input image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# Convert the image to gray scale and blur it slightly
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow('Blur Image', blur)
cv.waitKey(0)

# Apply simple thresholding with hardcoded threshold value
(T, threshInv) = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
cv.imshow('THRESH BINARY INV', threshInv)
cv.waitKey(0)

# Apply Otsu's thresholding
(T, otsuthreshInv) = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow('Otsu THRESH BINARY INV', otsuthreshInv)
print(T)
cv.waitKey(0)

# Instead of manually specifying the threshold value we can use adaptive thresholding to examin
# neighbourhoods of pixels and adaptively threshold each neighborhood
# First MEan Adaptive thresholding, here need to fine tune last 2 parameters (21 kernrl size, 10 C)
thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
cv.imshow('Mean Adaptive Threshold', thresh)
cv.waitKey(0)

# First Gaussian Adaptive thresholding, here need to fine tune last 2 parameters (21 kernrl size, 4 C)
thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
cv.imshow('Gaussian Adaptive Threshold', thresh)
cv.waitKey(0)
cv.destroyAllWindows()