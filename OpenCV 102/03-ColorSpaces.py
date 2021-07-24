import cv2 as cv
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='05.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# store each channel in a separate variable and display
B, G, R = cv.split(image)
cv.imshow('Blue', B)
cv.waitKey(0)
cv.imshow('Green', G)
cv.waitKey(0)
cv.imshow('Red', R)
cv.waitKey(0)
cv.destroyAllWindows()

# Show Original Image then Convert the image into HSV space and display it
cv.imshow('Original Image', image)
cv.waitKey(0)
# Converting to HSV
HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow('HSV', HSV)
cv.waitKey(0)

# Split the HSV image into individual channels and display it
H, S, V = cv.split(HSV)
cv.imshow('H', H)
cv.waitKey(0)
cv.imshow('S', S)
cv.waitKey(0)
cv.imshow('V', V)
cv.waitKey(0)
cv.destroyAllWindows()

# Show Original Image then Convert the image into L*a*b space and display it
cv.imshow('Original Image', image)
cv.waitKey(0)
# Converting to L*a*b
LAB = cv.cvtColor(image, cv.COLOR_BGR2LAB)
cv.imshow('LAB', LAB)
cv.waitKey(0)

# Split the L*a*b image into individual channels and display it
L, A, B = cv.split(LAB)
cv.imshow('L', L)
cv.waitKey(0)
cv.imshow('A', A)
cv.waitKey(0)
cv.imshow('B', B)
cv.waitKey(0)
cv.destroyAllWindows()

# Show Original Image then Convert the image into Gray and display it
cv.imshow('Original Image', image)
cv.waitKey(0)
# Converting to Gray
Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', Gray)
cv.waitKey(0)
cv.destroyAllWindows()
