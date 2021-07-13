import cv2 as cv
import numpy as np
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='00.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('image', image)
cv.waitKey(0)

# examples of using add/subtract functions in OpenCV
add = cv.add(np.uint8([200]), np.uint8([100]))
subtract = cv.subtract(np.uint8([50]), np.uint8([100]))
print(f'Max of 255: {add}')
print(f'Min of 0: {subtract}')

# examples of adding/subtracting using numpy
add = np.uint8([200]) + np.uint8([100])
subtract = np.uint8([50]) - np.uint8([100])
print(f'Wrap around: {add}')
print(f'Wrap around: {subtract}')

# increasing the brightness by increasing pixel intensities by 100 using OpenCV add function
M = np.ones(image.shape, dtype='uint8')*100
bright = cv.add(image, M)
cv.imshow('Brighter Image', bright)
cv.waitKey(0)

# decreasing the brightness by decreasing pixel intensities by 50 using OpenCV subtract function
M = np.ones(image.shape, dtype='uint8')*50
dark = cv.subtract(image, M)
cv.imshow('Darker Image', dark)
cv.waitKey(0)

# increasing the brightness by increasing pixel intensities by 100 using numpy addition
M = np.ones(image.shape, dtype='uint8')*100
brighter = image + M
cv.imshow('Numpy addition', brighter)
cv.waitKey(0)

# decreasing the brightness by decreasing pixel intensities by 50 using numpy subtraction
M = np.ones(image.shape, dtype='uint8')*50
darker = image - M
cv.imshow('Numpy Subtraction', darker)
cv.waitKey(0)
cv.destroyAllWindows()
