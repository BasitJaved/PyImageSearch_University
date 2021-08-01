from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='road3.jpg', help='Path to target image')
ap.add_argument('-s', '--ReferenceImage', type=str, default='mou.jpg', help='Path to source image')
args = vars(ap.parse_args())

# Load and display source and target images
reference = cv.imread(args['ReferenceImage'])
source = cv.imread(args['image'])
cv.imshow('Source', source)
cv.waitKey(0)
cv.imshow('Reference', reference)
cv.waitKey(0)

# Determine if we are performing multi-channel histogram matching  and perform histogram matching
print('[INFO] performing histogram matching')
multi = True if source.shape[-1]>1 else False
matched = exposure.match_histograms(source, reference, multichannel=multi)
cv.imshow('Matched', matched)
cv.waitKey(0)
cv.destroyAllWindows()

# Construct image to display histogram plots for each channel before and after histogram matching was applied
(fig, axes) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# loop over source image, reference image and output matched image
for (i, image) in enumerate((source, reference, matched)):
    # convert image from BGR to RGB format
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # loop over the names of channels in RGB order
    for (j, color) in enumerate(('red', 'green', 'blue')):
        # Compute histogram of current channel and plot it
        (hist, bins) = exposure.histogram(image[..., j], source_range='dtype')
        axes[j, i].plot(bins, hist/hist.max())

        # Compute the Comulative distribution function for current channel and plot it
        (cdf, bins) = exposure.cumulative_distribution(image[..., j])
        axes[j, i].plot(bins, cdf)

        # Set the y-axis label for current plot to be name of current color channel
        axes[j, 0].set_ylabel(color)

# Set axis titles
axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')

# Display output plots
plt.tight_layout()
plt.show()
