from tensorflow.keras.layers.experimental import preprocessing
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os

def load_images(imagePaths):
    # read image from disk, decode it, convert the data type to floating input and resize it
    image = tf.io.read_file(imagePaths)
    image = tf.image.decode_jpg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (156, 156))

    # parse the class label from file path
    label = tf.strings.split(imagePaths, os.path.sep)[-2]

    return (image, label)

def augment_using_layers(images, labels, aug):
    # pass a batch of images through our augmentation pipline and return augmented images
    images = aug(images)

    return (images, labels)

def augment_using_ops(images, labels):
    # randomly flip the image horizontally, randomly flip the image vertically and rotate the image by 90 degrees
    # in counter clock-wise direction
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)

    return (images, labels)

# construct an argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input images dataset')
ap.add_argument('-a', '--augment', type=bool, default=False, help='Flag to apply Augmentation or not')
ap.add_argument('-t', '--type', choices=['layers', 'ops'], help='Method to be used to perform Augmentation')
args = vars(ap.parse_args())

#set Batch Size
BS = 8

#grab all image paths
imagePaths = list(paths.list_images(args['dataset']))

# build dataset and data input pipline
print('[INFO] loading the dataset...')
ds = tf.data.Dataset.from_tensor_slices(imagePaths)
ds = (ds
      .shuffle(len(imagePaths), seed=42)
      .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
      .batch(BS)
      )

# check if we should apply data Augmentation
if args['augment']:
    # check if we will be using layers to perform data augmentataion
    if args['augment'] == 'layers':
        # initialize our Sequential data Augmentation pipline
        aug = tf.keras.Sequential([
            preprocessing.RandomFlip('horizontal_and_vertical'),
            preprocessing.RandomZoom(
                height_factor=(-0.05, -0.15),
                width_factor=(-0.05, -0.15)),
            preprocessing.RandomRotation(0.3)
        ])

        # add data augmentation to our pipline
        ds = (ds
              .map(lambda x, y:augment_using_layers(x, y, aug), num_parallel_calls=tf.data.AUTOTUNE)
              )

    # Otherwise we will be using Tensorflow image operations to perform data augmentation
    else:
        ds = (ds
              .map(augment_using_ops, num_parallel_calls=tf.data.AUTOTUNE)
              )

# complete our data input pipline
ds = (ds
      .prefetch(tf.data.AUTOTUNE)
      )

# grab a batch of data
batch = next(iter(ds))

# initialize a figure
print('[INFO] visualizing first batch of dataset...')
title = 'With Data augmentation {}'.format('applied ({})'.format(args['type']) if args['augment'] else 'not applied')
fig = plt.figure(figsize=(BS, BS))
fig.suptitle(title)

# loop over the batch size
for i in range(0, BS):
    # grab the image and label from the batch
    (image, label) = (batch[0][i], batch[1][i])

    # Create a sub plot and plot image and label
    ax = plt.subplot(2, 4, i+1)
    plt.imshow(image.numpy())
    plt.title(label.numpy().decode('UTF-8'))
    plt.axis('off')

# show the plot
plt.tight_layout()
plt.show()
