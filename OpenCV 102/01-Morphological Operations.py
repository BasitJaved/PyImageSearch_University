import cv2 as cv
import numpy as np
import argparse

# initiate the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='image0.jpg', help='Path to input image')
args = vars(ap.parse_args())

# read and display image
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Original image', gray)
cv.waitKey(0)

# applying a series of erosion
# Erosion will eat away pixels
for i in range(0, 3):
    eroded = cv.erode(gray.copy(), None, iterations=i+1)
    cv.imshow(f'Eroded {i} times', eroded)
    cv.waitKey(0)

cv.destroyAllWindows()
cv.imshow('Original', gray)
cv.waitKey(0)

# applying a series of dilation
# Dilation will add pixels
for i in range(0, 3):
    eroded = cv.dilate(gray.copy(), None, iterations=i+1)
    cv.imshow(f'Dilated {i} times', eroded)
    cv.waitKey(0)

cv.destroyAllWindows()

# Applying opening which is applying Erosion followed by dilation to remove noise from image
# read and display image
image = cv.imread('image1.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Original image', gray)
cv.waitKey(0)

# initiating kernal sizes for erosion and dilation
kSize = [(3,3), (5,5), (7,7)]

# applying opening operation
for ks in kSize:
    # Construct a rectangular kernal from current size and apply opening operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ks)
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    cv.imshow(f'Opening : ({ks[0], ks[1]})', opening)
    cv.waitKey(0)

cv.destroyAllWindows()

# Applying Closing which is applying dilation followed by Erosion, it is opposite of opening
# read and Display image
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Original image', gray)
cv.waitKey(0)

# applying closing operation
for ks in kSize:
    # Construct a rectangular kernal from current size and apply opening operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ks)
    opening = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    cv.imshow(f'Closing : ({ks[0], ks[1]})', opening)
    cv.waitKey(0)

cv.destroyAllWindows()

# Applying Morphological gradient which is difference between dilation and Erosion, it is kind of a boundary
# detection between ROI and background
# read and Display image
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Original image', gray)
cv.waitKey(0)

# applying closing operation
for ks in kSize:
    # Construct a rectangular kernal from current size and apply opening operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ks)
    opening = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
    cv.imshow(f'Gradient : ({ks[0], ks[1]})', opening)
    cv.waitKey(0)

cv.destroyAllWindows()

# Bottom Hat/Black Hat morphological operation is difference between original/gray input image and closing
# it is used to reveal Dark region of an image on Bright background
image = cv.imread('car13.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Original image', gray)
cv.waitKey(0)

# Construct a rectangular kernal (13x5) and apply black hat operation which will enable us to find dark
# regions on a white background
rectKernal = cv.getStructuringElement(cv.MORPH_RECT, (13,5))
blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernal)
cv.imshow('Black Hat', blackhat)
cv.waitKey(0)

# Top Hat/White Hat morphological operation is difference between original/gray input image and opening
# it is used to reveal bright region of an image on dark background
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernal)
cv.imshow('White Hat', tophat)
cv.waitKey(0)
cv.destroyAllWindows()
