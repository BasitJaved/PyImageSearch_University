from Preprocessing.local_binary_pattrens import LocalBinaryPattrens
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from imutils import paths
import argparse
import time
import cv2 as cv
import os

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
args = vars(ap.parse_args())

# grab the image paths in input datset directory
imagePaths = list(paths.list_images(args['datset']))

# initialize the local binary pattrens descriptor along with the data and labels
print('[INFO] extracting featueres...')
desc = LocalBinaryPattrens(24, 8)
data = []
labels = []