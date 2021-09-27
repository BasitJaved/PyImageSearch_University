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
