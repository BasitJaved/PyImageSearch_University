import cv2 as cv
import numpy as np
import argparse
import imutils

#construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type = str, default = 'road2.jpg', help = 'Path to input image')
args = vars(ap.parse_args())

#load the input image and display it
image = cv.imread(args['image'])
cv.imshow('image', image)
cv.waitKey(0)

#Resizing the image to 500 pixels wide but to prevent resized image from being skewed/distorted we need to
#calculate the aspect ratio of new width to old width
pix = 500.0
r = pix/image.shape[1]                  #to get ration (new_width / old_width)
dim = (int(pix), int(image.shape[0]*r)) # (new_width, Original_image_height*ratio)

#perform resizing of image
resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
cv.imshow('Resized Width', resized)
cv.waitKey(0)

#Resizing the image to 500 pixels high but to prevent resized image from being skewed/distorted we need to
#calculate the aspect ratio of new height to old height
pix = 500.0
r = pix/image.shape[0]                  #to get ration (new_height / old_height)
dim = (int(image.shape[1]*r), int(pix)) #(Original_image_width*ratio, newHeight)

#perform resizing of image
resizedH = cv.resize(image, dim, interpolation=cv.INTER_AREA)
cv.imshow('Resized height', resizedH)
cv.waitKey(0)

#Resizing width using imutils.resize function that will autometically maintain aspect ratio
resizedImW = imutils.resize(image, width=416)
cv.imshow('Resized Width Imutils', resizedImW)
cv.waitKey(0)

#Resizing height using imutils.resize function that will autometically maintain aspect ratio
resizedImH = imutils.resize(image, height=316)
cv.imshow('Resized Height Imutils', resizedImH)
cv.waitKey(0)

#load new input image and display it
imageNew = cv.imread('00.jpg')
cv.imshow('image', imageNew)
cv.waitKey(0)

#construct list of interpolation methods in OpenCV
methods = [
    ('cv.INTER_NEAREST', cv.INTER_NEAREST),     #generally works better for decreasing image size
    ('cv.INTER_LINEAR', cv.INTER_LINEAR),       #generally works better for decreasing image size
    ('cv.INTER_AREA', cv.INTER_AREA),           #generally works better for decreasing image size
    ('cv.INTER_CUBIC', cv.INTER_CUBIC),         #generally works better for increasing image size
    ('cv.INTER_LANCZ0S4', cv.INTER_LANCZOS4)    #same as cubic generally works better for increasing image size
]

#loop over interpolation methods
for (name, method) in methods:
    #increase size of image using current interpolation method
    print(f'INFO {name}')
    resize = imutils.resize(imageNew, width = imageNew.shape[1]*2, inter=method)
    cv.imshow(f'Method: {name}', resize)
    cv.waitKey(0)





cv.destroyAllWindows()