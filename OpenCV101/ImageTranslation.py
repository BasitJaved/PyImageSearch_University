import cv2 as cv
import numpy as np
import argparse

#construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default= 'road2.jpg',
                help = 'path to input image')
args = vars(ap.parse_args())

#read and display image
image = cv.imread(args['image'])
cv.imshow('Image', image)
cv.waitKey(0)

#shift the image 25 pixels to the right and 50 pixels down
M = np.float32([[1, 0, 25], [0, 1, 50]])    #([[1, 0, shiftX], [0, 1, shiftY]])
shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow('Shifted Right', shifted)
cv.waitKey(0)

#shift the image 50 pixels to the left and 90 pixels up
M2 = np.float32([[1, 0, -50], [0, 1, -90]])    #([[1, 0, shiftX], [0, 1, shiftY]])
shifted_left = cv.warpAffine(image, M2, (image.shape[1], image.shape[0]))
cv.imshow('Shifted Left', shifted_left)
cv.waitKey(0)

#shift the 100 pixels down
M3 = np.float32([[1, 0, 0], [0, 1, 100]])    #([[1, 0, shiftX], [0, 1, shiftY]])
shifted_down = cv.warpAffine(image, M3, (image.shape[1], image.shape[0]))
cv.imshow('Shifted Down', shifted_down)
cv.waitKey(0)

#can also do like this
#import imutils
#shifted = imutils.translate(image, 0, 100)
#cv.imshow('Shifted Down', shifted)
#cv.waitKey(0)

cv.destroyAllWindows()