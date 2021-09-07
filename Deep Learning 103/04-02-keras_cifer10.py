from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='Path to output loss/accuracy plot')
args = vars(ap.parse_args())

# Grab the Mnist dataset
print('[INFO] accessing CIFAR10...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# Each image in Cifar10 data is represented as a 32x32x3 image but in order to apply a standard neural network we
# must first flatten the image to simple list of 32x32x3=3072 pixels
trainX = trainX.reshape((trainX.shape[0], 32*32*3))
testX = testX.reshape((testX.shape[0], 32*32*3))

# Scale the data from 255 to the range [0, 1]
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

# Convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize label names for CIFAR10 dataset
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# define a 3072-1024-512-10 architecture using keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train model using SGD
print('[INFO] training network...')
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# evaluate the network
print('[INFO] evaluating the network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('keras_cifar10.png')