import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='coin5.jpg', help='Path to input image')
ap.add_argument('-s', '--scharr', type=int, default=0, help='If using Scharr or not, 0 means not(Default)')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# Convert image to Gray Scale
# When computing gradient magnitude and orientation of an image we assume we are working with Gray Scale
# or single channel image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)
cv.waitKey(0)

# Set the kernel size, depending upon we are using Sobel operator of Scharr operator, then compute gradient
# Scharr kernel is more senstive to changes in gradient then Sobel kernel
# along the x,y-axis
ksize = -1 if args['scharr'] > 0 else 3  # -1 means use shcarr 3 means use Sobel
gX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize)

# gX, and gY (gradient magnitude images) are of floating point data type so we need to convert them back to
# unsigned 8-bit integer representation so other OpenCV functions can operate on them and visualize them
gX = cv.convertScaleAbs(gX)
gY = cv.convertScaleAbs(gY)

# combine gradient representation into a single image
combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)

# Display output images
cv.imshow('Sobel/Scharr X', gX)
cv.waitKey(0)
cv.imshow('Sobel/Scharr Y', gY)
cv.waitKey(0)
cv.imshow('Sobel/Scharr Combined', combined)
cv.waitKey(0)
cv.destroyAllWindows()

# Computing Megnitude and Orientation
# Compute gradient along x, y
gX = cv.Sobel(gray, ddepth=cv.CV_64F, dx=1, dy=0)
gY = cv.Sobel(gray, ddepth=cv.CV_64F, dx=0, dy=1)

# Compute gradient Magnitude and orientation
# The gradient magnitude is used to measure how strong the change in image intensity is.
# The gradient orientation is used to determine in which direction the change in intensity is pointing.
magnitude = np.sqrt((gX**2) + (gY**2))
orientation = np.arctan2(gY, gX) * (180/np.pi) % 180 # (180/np.pi) % 180 will convert it to degree
# % 180 at end will clamp the values between 0 to 180 degrees

# initialize a figure to display input grascale image, along with gradient magnitude and orientation
(fig, axs) = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 4))

# plot each of image
axs[0].imshow(gray, cmap='gray')
axs[1].imshow(magnitude, cmap='jet')
axs[2].imshow(orientation, cmap='jet')

# set title of each image
axs[0].set_title('GrayScale')
axs[1].set_title('Gradient Magnitude')
axs[2].set_title('Gradient Orientation [0, 180]')

# Loop over each axis and turn off x and y ticks
for i in range(0, 3):
    axs[i].get_xaxis().set_ticks([])
    axs[i].get_yaxis().set_ticks([])

# show plots
plt.tight_layout()
plt.show()
