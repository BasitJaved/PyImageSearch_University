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
imagePaths = list(paths.list_images(args['dataset']))

# initialize the local binary pattrens descriptor along with the data and labels
print('[INFO] extracting featueres...')
desc = LocalBinaryPattrens(24, 8)
data = []
labels = []

# loop over the data set of images
for imagePath in imagePaths:
    # Load image, convert it into grayscale and quantify it using LBPs
    image = cv.imread(imagePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # extract the labels from image path and then update the label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

# partition the data into training and testing sets 75-25
print('[INFO] Constructing training and testing split...')
(trainX, testX, trainY, testY) = train_test_split(data, labels, random_state=22, test_size=0.25)

# Construct the set of parameters to tune
parameters = [
    {'kernel': ['Linear'], 'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
    {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
    {'kernel': ['rbf'], 'gamma': ['auto', 'scale'], 'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]

# tune the hyperparameters via a cross-validation grid search
print('[INFO] Tuning hyperparameters via a grid search...')
grid = GridSearchCV(estimator=SVC(), param_grid=parameters, n_jobs=-1)
start = time.time()
grid.fit(trainX, trainY)
end = time.time()

# show the grid search information
print(f'[INFO] grid search took {end-start} seconds')
print(f'[INFO] grid search best score {grid.best_score_*100} ')
print(f'[INFO] grid search best parameters {grid.best_params_} ')

# grab the best model and evaluate it
print('[INFO] evaluating...')
model = grid.best_estimator_
predictions = model.predict(testX)
print(classification_report(testY, predictions))