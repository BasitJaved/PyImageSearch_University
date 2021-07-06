import cv2 as cv
import numpy as np
#import argparse

#initialize a canvas as 300x300 pixel image with 3 channels
canvas = np.zeros((300,300,3), dtype='uint8') #black image since all zeros

#draw a green line from top left corner of canvas to bottom right
green = (0, 255, 0)
cv.line(canvas, (0, 0), (300, 300), green)

#draw a 3 pixcel thick Red line from top left corner of canvas to bottom right
red = (0, 0, 255)
cv.line(canvas, (300, 0), (0, 300), red, thickness=3)

#draw a Green 50x50 pixel sequare rectangle starting at 10x10 and ending at 60x60
green = (0, 255, 0)
cv.rectangle(canvas, (10, 10), (60, 60), green)

#draw a red 150x25 pixel rectangle starting at 50x200 and ending at 200x225 with
#thickness 5
red = (0, 0, 255)
cv.rectangle(canvas, (50, 200), (200, 225), red, 5)

#draw a blue 25x75 pixel rectangle filled in
blue = (225, 0, 0)
cv.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv.imshow('Image', canvas)
cv.waitKey(0)

#initialize another canvas as 300x300 pixel image with 3 channels
canvas1 = np.zeros((300,300,3), dtype='uint8') #black image since all zeros

#save width and height of canvas in w and h
h = canvas1.shape[0]
w = canvas1.shape[1]
(centerX, centerY) = (h//2, w//2)

#white color
white = (255, 255, 255)

#loop over increasing radii from 0 pixels to 175 pixels in 25 pixel increment
for i in range(0, 175, 25):
    #draw a white circle with current radius size
    cv.circle(canvas1, (centerX, centerY), i, white)
cv.imshow('Image2', canvas1)
cv.waitKey(0)

#initialize another canvas as 300x300 pixel image with 3 channels
canvas2 = np.zeros((300,300,3), dtype='uint8') #black image since all zeros

#draw 25 random circles
for i in range(0, 25):
    #generate a random radius size between 5 amd 200
    #generate a random color
    #pick a random point on canvas2 to create circle
    radius = np.random.randint(5, high=200)
    color = np.random.randint(0, high=256, size=(3,)).tolist()
    pt = np.random.randint(0, high=300, size=(2,))

    #draw circle filled with random color
    cv.circle(canvas2, tuple(pt), radius, color, -1)
cv.imshow('Image3', canvas2)
cv.waitKey(0)
cv.destroyAllWindows()