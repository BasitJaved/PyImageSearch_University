import numpy as np
import argparse
import cv2 as cv


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping pixel values to their adjusted gamma values
    invgamma = 1.0/gamma
    table = np.array([((i/255)**invgamma)*255 for i in np.arange(0, 256)]).astype('uint8')

    # apply gamma correction using lookup table
    return cv.LUT(image, table)

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='moun3.jpg', help='Path to input image')
args = vars(ap.parse_args())

# load and display the image
image = cv.imread(args['image'])

# apply gamma correction using various values of gamma
for gamma in np.arange(0.0, 5.0, 0.5):
    # ignore when gamma = 1 as there will be no change
    if gamma == 1:
        continue

    # apply gamma correction and show images
    gamma = gamma if gamma>0 else 0.1  # if gamma =0, invGamma will be 1.0/0.0
    adjusted = adjust_gamma(image, gamma=gamma)
    cv.putText(adjusted, f'g={gamma}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv.imshow('Images', np.hstack([image, adjusted]))
    cv.waitKey(0)

cv.destroyAllWindows()
