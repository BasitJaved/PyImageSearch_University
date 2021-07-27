import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def plot_histogram(img, title, mask=None):
    # Split the image into respective channels and initialize tuple of channel names with figure for plotting
    chans = cv.split(img)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Loop over image channels
    for (chan, color) in zip(chans, colors):
        # Create Histogram of current channel and plot it
        hist = cv.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


# Read and display image
image = cv.imread('road2.jpg')

# Convert and display image using matplotlib
# Matplotlib expect image in RGB format so need to convert from BGR to RGB
plt.figure()
plt.axis('Off')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

# plot histogram for image
plot_histogram(image, 'Histogram for Original Image')

# Construct a mask for our image, mask will be blask for the regions we want to ignore and white for regions
# we want to examin
mask = np.zeros(image.shape[:2], dtype='uint8')
cv.rectangle(mask, (65, 0), (870, 480), 255, -1)
cv.imshow('Mask', mask)
cv.waitKey(0)

# Display masked Region
masked = cv.bitwise_and(image, image, mask=mask)
cv.imshow('Masked', masked)
cv.waitKey(0)

# plot histogram for masked image
plot_histogram(image, 'Histogram for Masked Image', mask = mask)

# Show plots
plt.show()
