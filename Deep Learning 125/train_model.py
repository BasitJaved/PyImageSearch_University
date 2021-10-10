import matplotlib
matplotlib.use('Agg')

from Preprocessing.cancernet import CancerNet
from Preprocessing import config
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib as plt
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

    return (image, label)

# tf decorator
@tf.function
def augment(image, label):
    # Perform random Horizontal and vertical flips
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    return (image, label)

# Consturct the argument Parser and Parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='Path to Output loss/accuracy plot')
args = vars(ap.parse_args())

# grab all the training, validation and testing dataset image paths
trainPaths = list(paths.list_images(config.Train_path))
valPaths = list(paths.list_images(config.Val_path))
testPaths = list(paths.list_images(config.Test_path))

# Calculate the total number of training images in each class and initialize a dictionary to store class weights
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals =trainLabels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# build the training dataset and data input pipline
trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
trainDS = (trainDS
           .shuffle(len(trainPaths))
           .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
           .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .batch(config.Bs)
           .prefetch(tf.data.AUTOTUNE))

# build the Validation dataset and data input pipline
valDS = tf.data.Dataset.from_tensor_slices(valPaths)
valDS = (valDS
           .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .batch(config.Bs)
           .prefetch(tf.data.AUTOTUNE))

# build the testing dataset and data input pipline
testDS = tf.data.Dataset.from_tensor_slices(testPaths)
testDS = (testDS
           .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .batch(config.Bs)
           .prefetch(tf.data.AUTOTUNE))

# initialize out CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3, classes=1)
opt = Adagrad(learning_rate=config.Init_lr, decay=config.Init_lr/config.Num_Epochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# initialize an early stopping callback to stop model from overfitting
es = EarlyStopping(monitor='val loss',
                   patience=config.Early_stopping_patience,
                   restore_best_weights=True)

# fit the model
H = model.fit(x = trainDS,
              validation_data=valDS,
              class_weight=classWeight,
              epochs=config.Num_Epochs,
              callbacks=[es],
              verbose=1)

# evaluate the model on test set
(_, acc) = model.evaluate(testDS)
print(f'[INFO] test accuracy: {acc*100}')

# Plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label = 'Train Loss')
plt.plot(H.history['val_loss'], label = 'Val Loss')
plt.plot(H.history['accuracy'], label = 'Train Acc')
plt.plot(H.history['val_accuracy'], label = 'Val Acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])
