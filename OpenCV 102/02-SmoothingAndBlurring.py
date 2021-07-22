import cv2 as cv
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='00.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# Initiate a list of Kernel sizes so we can evaluate the relationship between kernel size and amount of blurring
kSize = [(3, 3), (7, 7), (9, 9), (15, 15)]
# loop over the kernel sizes and apply average blur to image using kernel sizes
for (kx, ky) in kSize:
    blurred = cv.blur(image, (kx, ky))
    cv.imshow(f'Average ({kx}, {ky})', blurred)
    cv.waitKey(0)
cv.destroyAllWindows()

# Display image again and using same kernels applying gaussian blur
# Most of the time we will use Gaussian blur as it gives more realistic results
cv.imshow('Original Image', image)
cv.waitKey(0)
# loop over the kernel sizes and apply Gaussian blur to image using kernel sizes
for (kx, ky) in kSize:
    blurred = cv.GaussianBlur(image, (kx, ky), 0)
    cv.imshow(f'Gaussian ({kx}, {ky})', blurred)
    cv.waitKey(0)
cv.destroyAllWindows()

# Display image again and applying Median blur
# Median blur is very very good for salt & pepper noise but other then that it is not used mostly
# For damaged images or photos captured under highly suboptimal conditions,
# a median blur can really help as a pre-processing step prior to passing the image
cv.imshow('Original Image', image)
cv.waitKey(0)
# loop over the kernel sizes and apply Median blur to image using kernel sizes
for k in (3, 7, 9, 15):
    blurred = cv.medianBlur(image, k)
    cv.imshow(f'Median ({k})', blurred)
    cv.waitKey(0)
cv.destroyAllWindows()

# Bilateral blur reduces the noise in image while still maintaining the edges, and it accomplishes this
# by introducing two Gaussian distributions.
# First Gaussian function only considers spatial neighbors, i.e pixels that appear close together in
# (x, y) coordinates
# The Second Gaussian then models the pixel intensity of neighborhood ensuring only pixels with similar
# intensities are included in computation of blur
# largest downside to this method is that it is considerably slower than its averaging,
# Gaussian, and median blurring counterparts.
# Display image again and construct a list of Bilateral filtering parameters
cv.imshow('Original Image', image)
cv.waitKey(0)
# params = [(11, 21, 7), (11, 41, 21), (11, 61, 39), (11, 81, 59)]  # (Diameter, SigmaColor, SigmaSpace)
params = [(15, 11, 7), (15, 33, 21), (15, 66, 42), (15, 99, 63)]  # (Diameter, SigmaColor, SigmaSpace)
# loop over the parameters and apply Bilateral blur
for (diameter, sigmaColor, sigmaSpace) in params:
    blurred = cv.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    cv.imshow(f'Blurred d={diameter}, sc={sigmaColor}, ss={sigmaSpace}', blurred)
    cv.waitKey(0)
cv.destroyAllWindows()
