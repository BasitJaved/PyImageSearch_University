import matplotlib
matplotlib.use('Agg')

from Preprocessing.cancernet import CancerNet
from Preprocessing import config
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import os

def load_images(imagePaths):
    # read image from disk, decode it, convert the data type to floating input and resize it
    image = tf.io.read_file(imagePaths)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, config.Image_size)

    # parse the class label from file path
    label = tf.strings.split(imagePaths, os.path.sep)[-2]
    label = tf.strings.to_number(label, tf.int32)