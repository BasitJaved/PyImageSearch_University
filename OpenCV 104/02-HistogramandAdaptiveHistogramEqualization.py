import argparse
import cv2 as cv

###########################################################
########### SIMPLE EQUALIZATION ###########################
###########################################################

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='moon5.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Load image from disk, convert it to grayscale and display
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)

# Apply histogram Equalization and display equalized image
print('[Info] Performing Histogram Equalization')
equalized = cv.equalizeHist(gray)
cv.imshow('Equalized', equalized)
cv.waitKey(0)

# applying to color images by converting to L*a*b format
cimage = cv.imread('road2.jpg')
LAB = cv.cvtColor(cimage, cv.COLOR_BGR2Lab)
L, A, B = cv.split(LAB)
cv.imshow('L', L)
cv.waitKey(0)

# Apply histogram Equalization and display equalized image
print('[Info] Performing Histogram Equalization')
equalizedL = cv.equalizeHist(L)
cv.imshow('EqualizedL', equalizedL)
cv.waitKey(0)

# Merge L channel back and convert back to BGR
lab = cv.merge((equalizedL, A, B))
BGR = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
cv.imshow('Original', cimage)
cv.waitKey(0)
cv.imshow('BGR', BGR)
cv.waitKey(0)
cv.destroyAllWindows()

###########################################################
########### Adaptive EQUALIZATION #########################
###########################################################

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='moon5.jpg', help='Path to input image')
ap.add_argument('-c', '--clip', type=float, default=2.0, help='Threshold for contrast Limiting')  # use b/w 2-4
ap.add_argument('-t', '--tile', type=int, default=8,
                help='Tile grid size -- divides the image into tile x tile cells')
args = vars(ap.parse_args())

# Load image from disk, convert it to grayscale and display
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)

# apply CLAHE (Constrast Limited Adaptive Histogram Equalization)
clahe = cv.createCLAHE(clipLimit=args['clip'], tileGridSize=(args['tile'], args['tile']))
equalized = clahe.apply(gray)
cv.imshow('Equalized', equalized)
cv.waitKey(0)
cv.destroyAllWindows()
