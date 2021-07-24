import cv2 as cv
import argparse
import numpy as np
from skimage.exposure import rescale_intensity

def convolve(image, kernel):
    # grab the spatial dimensions of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Allocate memory for output image taking care to pad the borders of input image so spatial size is not reduced
    pad = (kW - 1)//2
    image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype='float32')

    # Loop over the input image sliding the kernel across each (x,y) coordinate from left to right and top to bottom
    for y in np.arange(pad, iH+pad):      # looping on rows
        for x in np.arange(pad, iW+pad):  # looping on columns
            # Extract the ROI of image by extracting the center region of current (x, y) coordinate dimensions
            roi = image[y-pad: y+pad+1, x-pad: x+pad+1]

            # Perform the actual convolution by taking element wise multiplication and then summing the matrix
            k = (roi * kernel).sum()

            # store the convlove value in output (x, y)-coordinate of output image
            output[y - pad, x - pad] = k

    # Rescale the image to be in range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output*255).astype('uint8')

    return output

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type = str, default='06.jpg', help = 'Path to input image')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv.waitKey(0)

# Conver the image to gray scale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)
cv.waitKey(0)

# construct average blurring kernals used to smooth an image
smallBlur = np.ones((7, 7), dtype='float')*(1.0/(7*7))
largeBlur = np.ones((21, 21), dtype='float')*(1.0/(21*21))

# Construct a sharpning filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype='int')

# Construct laplacian kernal used to detect edge like regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype='int')

# Construct a sobal x-axis kernal
sobalX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype='int')

# Construct a sobal y-axis kernal
sobalY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype='int')

# Construct the kernel bank, a list of kernels we will apply using our custom convolve function and OpenCVs
# filter 2D function
kernelBank = (
    ('Small Blur', smallBlur),
    ('Large Blur', largeBlur),
    ('Sharpen', sharpen),
    ('Laplacian', laplacian),
    ('Sobal-X', sobalX),
    ('Sobal-Y', sobalY)
)

# loop over the kernals
for (kernelName, kernel) in kernelBank:
    # Apply the kernels to gray scale image using custom convolve function and openCVs filter2D function
    print(f'applying {kernelName} kernal')
    convolveoutput = convolve(gray, kernel)
    cvoutput = cv.filter2D(gray, -1, kernel)

    # Display the image
    cv.imshow(f'{kernelName}-Convolve', convolveoutput)
    cv.imshow(f'{kernelName}-OpenCV', cvoutput)
    cv.waitKey(0)


cv.destroyAllWindows()

