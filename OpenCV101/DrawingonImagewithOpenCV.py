import cv2 as cv
import numpy as np
import argparse

#construct the aregument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default= 'road2.jpg',
                help = 'path to input image')
args = vars(ap.parse_args())

#load the input image
image = cv.imread(args['image'])

#draw a circle, 2 filled in circles and one rectangle
cv.circle(image, (150, 100), 100, (0,255,255), 3)
cv.circle(image, (250, 300), 100, (255,0,255), -1)
cv.circle(image, (550, 600), 100, (255,255,0), -1)
cv.rectangle(image, (450, 100), (800, 450), (0, 255, 0), -1)

#display image
cv.imshow('Image', image)
cv.waitKey(0)
cv.destroyAllWindows()