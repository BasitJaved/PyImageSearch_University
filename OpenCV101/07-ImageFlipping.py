import cv2 as cv
import numpy as np
import argparse
import imutils

#construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type = str, default = '00.jpg', help = 'Path to input image')
args = vars(ap.parse_args())

#load and display image
image = cv.imread(args['image'])
cv.imshow('image', image)
cv.waitKey(0)

#Flip image Horizontally
print('[INFO] Flipping image Horizontally')
flipped_H = cv.flip(image, 1)
cv.imshow('Horizontally Flipped', flipped_H)
cv.waitKey(0)

#Flip image Vertically
print('[INFO] Flipping image Vertically')
flipped_V = cv.flip(image, 0)
cv.imshow('Vertically Flipped', flipped_V)
cv.waitKey(0)

#Flip image Horizontally and Vertically
print('[INFO] Flipping image Horizontally and Vertically')
flipped_HV = cv.flip(image, -1)
cv.imshow('Horizontally and Vertically Flipped', flipped_HV)
cv.waitKey(0)

cv.destroyAllWindows()