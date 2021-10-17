from skimage import feature
import numpy as np

class LocalBinaryPattrens:
    def __init__(self, numPoints, radius):
        # store num of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the local binary pattren representation of the image and use LBP representation to build histogram
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints+2))

        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        return hist