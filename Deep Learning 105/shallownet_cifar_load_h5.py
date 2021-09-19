from Preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from Preprocessing.simplepreprocessor import SimplePreprocessor
from Preprocessing.simpledatasetloader import SimpleDatasetLoader
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to output model')
args = vars(ap.parse_args())

# Load dataset
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

# initialize label names for Cifar-10 dataset
classlabels  = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# grab the list of images
print('[INFO] loading images ...')
testx = np.array(list(testX))
idxs = np.random.randint(0, len(testX), size=(10,))
testX = testX[idxs]
print(testX.shape)

# load pretrained network
print('[INFO] Loading pre-trained network...')
model = load_model(args['model'])

# Make predictions on images
print('[INFO] Predicting...')
preds = model.predict(testX, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(testX):
    # load image, draw predictions and display it
    image = cv.imread(imagePath)
    cv.putText(image, f'Label: {classlabels[preds[i]]}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('Image', image)
    cv.waitKey(0)

cv.destroyAllWindows()
