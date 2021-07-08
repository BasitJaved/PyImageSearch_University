import cv2 as cv
import numpy as np
import argparse
import imutils

#construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type = str, default = '00.jpg',
                help = 'Path to input image')
args = vars(ap.parse_args())

#Read image
image = cv.imread(args['image'])
cv.imshow('Image', image)
cv.waitKey(0)

#storing dimensions in to h and w variables
(h, w) = (image.shape[0], image.shape[1])
(cX, cY) = (w//2, h//2)                    #center point of image

#rotate the image 45 degrees arround the center point
M = cv.getRotationMatrix2D((cX, cY), 45, 1.0) #last value is scale of image increase it if need bigger image
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow('Rotated by 45 Degree', rotated)
cv.waitKey(0)


#rotate the image -90 degrees arround the center point
M = cv.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow('Rotated by -90 Degree', rotated)
cv.waitKey(0)

#rotate the image arround an arbitrary point instead of center of image
M = cv.getRotationMatrix2D((10, 10), 30, 0.5)
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow('Rotated by 30 Degree and Decrease Size', rotated)
cv.waitKey(0)

#rotate the image 180 degrees using imutils.rotate
rotated = imutils.rotate(image, 180)
cv.imshow('Rotated by 180 Degree', rotated)
cv.waitKey(0)

#rotate the image 45 degrees using imutils.rotate_bound
rotated = imutils.rotate_bound(image, 45)
cv.imshow('Rotated by 45 Degree', rotated)
cv.waitKey(0)

cv.destroyAllWindows()