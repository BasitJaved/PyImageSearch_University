from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2 as cv
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-m', '--model', type = str, default='vgg16', help='Name of Pre-trained network to use')
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes inside Keras
MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'inception': InceptionV3,
    'xception': Xception,
    'resnet': ResNet50
}

# ensure a valid model name was provided via command line argument
if args['model'] not in MODELS.keys():
    raise AssertionError('The --model command line argument should be a key in Models dictionary')

# initialize input image shape (224x224) along with pre-processing funvtion
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

#if we are using InceptionV3 or Xception network, then we need to set the input image shape to (299x299)
# and use a different preprocessing function
if args['model'] in ('inception', 'xception'):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load network weights from disk
print(f'[INFO] Loading {args["model"]} ...')
Network = MODELS[args['model']]
model = Network(weights='imagenet')

# load the input image using Keras helper utility while ensuring image is resized to 'inputShape', the required
# input dimensions for the ImageNet pre-trained Network
print('[INFO] Loading and Preprocessing image ...')
image = load_img(args['image'], target_size=inputShape)
image = img_to_array(image)

# expanding the dimensions of input image to include the batch_size information
# current dimension (inputShape[0], inputShape[1], 3)
# after expanding (1, inputShape[0], inputShape[1], 3)
image = np.expand_dims(image, axis = 0)

# pre-process the image using appropriate function based on the model that has been loaded
image = preprocess(image)

# Classify the image
print(f'[INFO] Classifying the image with {args["model"]}...')
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display rank-5 predictions + probabliities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print(f'{i+1}., {label}: {prob*100:.2f}')

# load image via openCV, draw top predictions on image and display image on our screen
orig = cv.imread(args['image'])
(imagenetID, label, prob) = P[0][0]
cv.putText(orig, f'Label: {label}, {prob*100:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.imshow('Classification', orig)
cv.waitKey(0)
cv.destroyAllWindows()