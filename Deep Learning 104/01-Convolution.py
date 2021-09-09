from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2 as cv

def convolve(image, K):
    # Grab the spatial dimensions of image and kernal
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # Allocate memory for output image taking care to pad the orders of input image so the spetial size is not reduced
    pad = (kW - 1) // 2
    image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype='float')

    # loop over the input image sliding kernal across each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract thr roi of image by extracting the center region of current (x, y)-coordinates dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking element-wise multiplication between ROI and kernal then
            # summing the matrix
            k = (roi * K).sum()

            # Store the convolved value in output (x, y)-coordinate of output image
            output[y - pad, x - pad] = k

    # Rescale the output image to be in range [0, 255]
    output = rescale_intensity(output,  in_range=(0, 255))
    output = (output * 255).astype('uint8')

    return output

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = vars(ap.parse_args())

# construct the average blurring kernels used to smoothen the image
smallBlur = np.ones((7, 7), dtype='float') * (1.0/(7*7))
largeBlur = np.ones((21, 21), dtype='float') * (1.0/(21*21))

# construct a sharpening filter
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype='int')

# Construct a laplacian kernel used to detect edge like regions
laplacian = np.array(([0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]), dtype='int')

# Construct a Sobel x-axis kernel
sobel_x = np.array(([-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]), dtype='int')

# Construct a Sobel y-axis kernel
sobel_y = np.array(([-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]), dtype='int')

# Construct an emboss kernel
emboss = np.array(([-2, -1, 0],
                    [-1, 1, 1],
                    [0, 1, 2]), dtype='int')

# construct the kernel bank, a list of kernels we are going to apply using our custom convolve function and OpenCV's
# filter 2D function
kernelBank = (
    ('smallBlur', smallBlur),
    ('largeBlur', largeBlur),
    ('Sharpen', sharpen),
    ('laplacian', laplacian),
    ('SobelX', sobel_x),
    ('SobelY', sobel_y),
    ('Emboss', emboss)
)

# Load the input image and convert it into gray scale
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelname, K) in kernelBank:
    # apply the kernel to gray scale image using both convolve function and openCv's filter2D function
    print(f'[INFO] applying {kernelname} kernel')
    convolveOutput = convolve(gray, K)
    openCVOutput = cv.filter2D(gray, -1, K)

    # show the output images
    cv.imshow('Original', gray)
    cv.imshow(f'{kernelname} - Convolve', convolveOutput)
    cv.imshow(f'{kernelname} - OpenCV', openCVOutput)
    cv.waitKey(0)
    cv.destroyAllWindows()
