import matplotlib.pyplot as plt
import cv2 as cv
import argparse

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='road2.jpg', help='Path to input file')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original image', image)
cv.waitKey(0)

# Convert Image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)
cv.destroyAllWindows()

# Compute Grayscale Histogram
hist = cv.calcHist([gray], [0], None, [256], [0, 256])

# Convert and display image using matplotlib
# Matplotlib expect image in RGB format so need to conver from BGR to RGB
plt.figure()
plt.axis('Off')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

# Plot Histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(hist)
plt.xlim([0, 256])

# Normalize Histogram
hist /= hist.sum()

# Plot Normalize Histogram
plt.figure()
plt.title('Grayscale Histogram (Normalized)')
plt.xlabel('Bins')
plt.ylabel('% of Pixels')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
