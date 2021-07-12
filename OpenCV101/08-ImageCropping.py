import cv2 as cv
import numpy as np
import argparse
import imutils

#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type = str, default = '00.jpg', help = 'Path to input image')
args = vars(ap.parse_args())

#load and display image
image = cv.imread(args['image'])
cv.imshow('image', image)
cv.waitKey(0)

#creating a numpy array to explain the concept
I = np.arange(0, 25)

#converting it into a 2-dimentionaly array
I = I.reshape((5,5))
print(I)

#slicing
print(I[0:3, 0:2])

#slicing
print(I[3:, 3:])

#slicing
print(I[2::2, ::2])

#get image shape
print(image.shape)

#cropping in openCV is achieved by numpy array slicing so lets try and extract face from our image
#roi = [startY:endY, startX:endX, channels]
face = image[85:150, 249:299, :]
cv.imshow('Face', face)
cv.waitKey(0)

#extract upper body without face from our image
body = image[145:325, :, :]
cv.imshow('body', body)
cv.waitKey(0)

#extract Legs from our image
legs = image[325:570, 195:355, :]
cv.imshow('legs', legs)
cv.waitKey(0)
cv.destroyAllWindows()