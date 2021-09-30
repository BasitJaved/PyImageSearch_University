from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2 as cv
import os

def load_house_attributes(inputPath):
    # initialize the list of column names in CSV file and load it using pandas
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']
    df = pd.read_csv(inputPath, sep=' ', header=None, names=cols)

    # determine unique zipcodes and number of data points with each zip code
    zipcodes = df['zipcode'].value_counts().keys().tolist()
    counts = df['zipcode'].value_counts().to_list()

    # loop over each of unique zipcode and their corresponding count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zipcode count for this dataset is extreamly unbalanced so we will sanitize our data by removing
        # houses with less then 25 houses per zip code
        if count<25:
            idxs = df[df['zipcode'] == zipcode].index
            df.drop(idxs, inplace=True)

    return df

def process_house_attributes(df, train, test):

    # initialize the column names of continuous data
    continuous = ['bedrooms', 'bathrooms', 'area']

    # performing min-max scaling each continuous feature to range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.fit_transform(test[continuous])

    # one-hot encode zip code categorical data
    zipBinarizer = LabelBinarizer().fit(df['zipcode'])
    trainCategorical = zipBinarizer.transform(train['zipcode'])
    testCategorical = zipBinarizer.transform(test['zipcode'])

    # construct training and testing data points by categorical features with continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    return (trainX, testX)

def load_house_images(df, inputPath):
    # initialize our images array
    images = []

    # loop over the indexes of houses
    for i in df.index.values:
        # find the four images for the house and sort file paths, ensuring four are always in same order
        basePath = os.path.sep.join([inputPath, '{}_*'.format(i+1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialize our list of input images along with output images after combining 4 input images
        inputImages = []
        outputImages = np.zeros((64, 64, 3), dtype='uint8')

        # loop over the input house path
        for housePath in housePaths:
            # load the input image, resize it to 32x32 and then update list of input images
            image = cv.imread(housePath)
            image = cv.resize(image, (32, 32))
            inputImages.append(image)

        # tile the four images in output image such that first image goes in top-right corner, second in the
        # top left corner, third in bottom right corner and forth in bottom left corner
        outputImages[0:32, 0:32] = inputImages[0]
        outputImages[0:32, 32:64] = inputImages[1]
        outputImages[32:64, 32:64] = inputImages[2]
        outputImages[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of images the network will be trained on
        images.append(outputImages)

    return np.array(images)
