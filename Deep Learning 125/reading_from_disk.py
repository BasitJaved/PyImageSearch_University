from Preprocessing.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from imutils import paths
import numpy as np
import argparse
import os

def load_images(imagePath):
    # read images from disk, decode it, resize it, and scale the pixel intensities to range [0, 1]
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (96, 96))/255.0

    # grab the label and encode it
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    oneHot = label == classNames
    encodedLabel = tf.argmax(oneHot)

    return (image, encodedLabel)

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset...')
args = vars(ap.parse_args())

# initialize batch size and number of steps
BS = 64
Num_steps = 1000

# grab the list of names in dataset directory and grab all unique classes
print('[INFO] loading image Paths...')
imagePaths = list(paths.list_images(args['dataset']))
classNames = np.array(sorted(os.listdir(args['dataset'])))

# build the data input pipline
print('[INFO] creating a tf.data input pipline...')
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = (dataset
           .shuffle(1024)
           .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .repeat()
           .batch(BS)
           .prefetch(tf.data.AUTOTUNE))

# create a standard image generator object
print('[INFO] Creating an ImageDataGenerator object...')
imageGen = ImageDataGenerator(rescale=1.0/255)
dataGen = imageGen.flow_from_directory(
    args['dataset'],
    target_size=(96, 96),
    batch_size=BS,
    class_mode='categorical',
    color_mode='rgb'
)

# benchmark the image data generator and display the number of data points generated, along with time taken
# to perform operation
totalTime = benchmark(dataGen, Num_steps)
print(f'[INFO] ImageDataGenerator generated {BS*Num_steps} images in {totalTime} seconds...')

# create a dataset iterator, benchmark the tf.data pipline and display the number of data points generated
# along with time taken
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, Num_steps)
print(f'[INFO] tf.data generated {BS*Num_steps} images in {totalTime} seconds...')