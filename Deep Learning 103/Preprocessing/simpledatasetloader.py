import numpy as np
import cv2 as cv
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store image preprocessor
        self.preprocessors = preprocessors

        # if preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class labels assuming that our path has following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and display each to image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as a feature vector by updating the data list followed by labels
            data.append(image)
            labels.append(label)

            # show an update every verbose images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f'[INFO] processed {i + 1}/{len(imagePaths)}')

        # return a tuple of data and labels
        return (np.array(data), np.array(labels))