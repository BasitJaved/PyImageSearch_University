from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--plot', type=str, default='training_plot.png', help='Path to Output loss/accuracy plot')
args = vars(ap.parse_args())

# define training hyper parameters
BS = 64
Epochs = 50

# load the cifar 10 dataset
print('[INFO] Loading training data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# initialize sequential data augmentation pipeline for training
trainingAug = tf.keras.Sequential([
    preprocessing.Rescaling(scale=1.0/255), #scaling pixcel intensities
    preprocessing.RandomFlip('horizontal_and_vertical'),
    preprocessing.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    preprocessing.RandomRotation(0.3)
])

# initialize a second augmentation pipline for testing
testAug = tf.keras.Sequential([
    preprocessing.Rescaling(scale=1.0/255)
    ])

# prepare training data pipline
trainDS = tf.data.Dataset.from_tensor_slices((trainX, trainY))
trainDS = (trainDS
           .shuffle(BS*100)
           .batch(BS)
           .map(lambda x, y: (trainingAug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(tf.data.AUTOTUNE)
           )

# prepare testing data pipline
testDS = tf.data.Dataset.from_tensor_slices((testX, testY))
testDS = (testDS
          .batch(BS)
          .map(lambda x, y: (testAug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(tf.data.AUTOTUNE)
          )

# initialize the model as a very basic CNN
print('[INFO] initializing Model')
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile the model
print('[INFO] Compiling Model...')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train model
print('[INFO] Training Model...')
H = model.fit(trainDS,
              validation_data=testDS,
              epochs=Epochs)

# show accuracy on testing set
(loss, accuracy) = model.evaluate(testDS)
print(f'[INFO] Accuracy:{accuracy}')

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label = 'train_loss')
plt.plot(H.history['val_loss'], label = 'val_loss')
plt.plot(H.history['accuracy'], label = 'train_acc')
plt.plot(H.history['val_accuracy'], label = 'val_acc')
plt.title('Training loss and accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accurcay')
plt.legend(loc='lower left')
plt.savefig(args['plot'])