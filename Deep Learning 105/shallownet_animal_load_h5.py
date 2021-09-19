from Preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from Preprocessing.simplepreprocessor import SimplePreprocessor
from Preprocessing.simpledatasetloader import SimpleDatasetLoader
#from conv.shallownet import ShallowNet
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-m', '--model', required=True, help='Path to output model')
args = vars(ap.parse_args())

# initialize class labels
classlabels = ['cat', 'dog', 'panda']

# grab the list of images
print('[INFO] loading images ...')
imagePaths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessor
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load tha dataset from disk and scale the raw pixel intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float')/255.0

# load pretrained network
print('[INFO] Loading pre-trained network...')
model = load_model(args['model'])

# Make predictions on images
print('[INFO] Predicting...')
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load image, draw predictions and display it
    image = cv.imread(imagePath)
    cv.putText(image, f'Label: {classlabels[preds[i]]}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('Image', image)
    cv.waitKey(0)

cv.destroyAllWindows()
