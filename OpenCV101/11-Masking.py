import cv2 as cv
import numpy as np
import argparse

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='06.jpg', help='Path to input image')
args = vars(ap.parse_args())

# load and display image
image = cv.imread(args['image'])
cv.imshow('image', image)
cv.waitKey(0)

# mask will be same size as our image but will have only 2 values 0 and 255
# mask value of 0 (background) are ignored while mask values of 255 (foreground)
# are allowed to be kept
mask = np.zeros(image.shape[:2], dtype='uint8')
cv.rectangle(mask, (155, 15), (520, 426), 255, -1)
cv.imshow('mask', mask)
cv.waitKey(0)

# apply mask to image
masked = cv.bitwise_and(image, image, mask=mask)
cv.imshow('masked', masked)
cv.waitKey(0)

# circular mask to get face only
Cmask = np.zeros(image.shape[:2], dtype='uint8')
cv.circle(Cmask, (340, 135), 125, 255, -1)
cv.imshow('Cmask', Cmask)
cv.waitKey(0)

# apply mask to image
Cmasked = cv.bitwise_and(image, image, mask=Cmask)
cv.imshow('Cmasked', Cmasked)
cv.waitKey(0)

cv.destroyAllWindows()
