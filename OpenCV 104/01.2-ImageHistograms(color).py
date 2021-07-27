import matplotlib.pyplot as plt
import cv2 as cv
import argparse

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='road3.jpg', help='Path to input file')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])

# Convert and display image using matplotlib
# Matplotlib expect image in RGB format so need to convert from BGR to RGB
plt.figure()
plt.axis('Off')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))


# Split the image into respective channels and initialize tuple of channel names with figure for plotting
chans = cv.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

# Loop over image channels
for (chan, color) in zip(chans, colors):
    # Create Histogram of current channel and plot it
    hist = cv.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

# Create a new figure and plot 2D-color histogram for green and blue channels
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and B')
plt.colorbar(p)

# Plot 2D-color histogram for Green and Red channels
ax = fig.add_subplot(132)
hist = cv.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and R')
plt.colorbar(p)

# Plot 2D-color histogram for Blue and Red channels
ax = fig.add_subplot(133)
hist = cv.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and R')
plt.colorbar(p)

# Examining the dimensionality of one of 2D histograms
print(f'2D Histogram shape: {hist.shape}, with {hist.flatten().shape[0]} values')

# Creating a 3D histogram utilizing all 3 channels with 8-bins in each direction, we cannot plot 3D histogram
# but theory is exactly like that of 2D histogram so we will just show shape of histogram
hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print(f'3D Histogram shape: {hist.shape}, with {hist.flatten().shape[0]} values')
plt.show()
