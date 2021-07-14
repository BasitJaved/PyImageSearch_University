import cv2 as cv
import numpy as np
import imutils

# Read and display first image
image = cv.imread('road2.jpg')

#Resizing the image to 500 pixels wide but to prevent resized image from being skewed/distorted we need to
#calculate the aspect ratio of new width to old width
pix = 616.0
dim = (int(pix+200.0), int(pix))

#perform resizing of image
resizedIm1 = cv.resize(image, dim, interpolation=cv.INTER_AREA)
cv.imshow('Resized1', resizedIm1)
cv.waitKey(0)

# Read and display second image
image2 = cv.imread('road3.jpg')

#Resizing the image to 500 pixels wide but to prevent resized image from being skewed/distorted we need to
#calculate the aspect ratio of new width to old width
pix = 616.0
dim = (int(pix+200.0), int(pix))

#perform resizing of image
resizedIm2 = cv.resize(image2, dim, interpolation=cv.INTER_AREA)
cv.imshow('Resized2', resizedIm2)
cv.waitKey(0)

print(resizedIm1.shape)
print(resizedIm2.shape)

# adding both images using opencv add function
finalCV = cv.add(resizedIm1, resizedIm2)
cv.imshow('finalCV', finalCV)
cv.waitKey(0)

# adding both images using np add
finalnp = resizedIm1 + resizedIm2
cv.imshow('finalnp', finalnp)
cv.waitKey(0)
