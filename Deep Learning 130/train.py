from Preprocessing import config
from Preprocessing.model import build_model
from Preprocessing import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import keras_tuner as kt
import numpy as np
import argparse
import cv2 as cv

# Construct Argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tuner', required=True, type=str, choices=['hyperband', 'random','bayesian'],
                help='type of hyper parameter tuner we will be using')
ap.add_argument('-p','--plot', required=True, help='path to output accuracy/loss plot')
args = vars(ap.parse_args())

# load Fashion Mnist dataset
print('[INFO] loading Fashion MNIST...')
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# add a channel dimension to dataset
print(trainX[0].shape)
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
print(trainX[0].shape)

# scale data to range of [0, 1]
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

# one hot encode training and testing labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize label names
label_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# initialize early stopping callback to prevent the model from overfitting/spending too much time with minimal gians
es = EarlyStopping(
        monitor='val_loss',
        patience=config.early_stopping_patience,
        restore_best_weights=True
)

# check if we will be using hyperband tuner
if args['tuner'] == 'hyperband':
    # instentiate hyperband tuner object
    print('[INFO]instentiating hyperband tuner object...')
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=config.epochs,
        factor=3,
        seed=42,
        directory=config.output_path,
        project_name=args['tuner']
    )
# check if we will be using random search tuner
elif args['tuner'] == 'random':
    # instentiate random search tuner object
    print('[INFO]instentiating random search tuner object...')
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        seed=42,
        directory=config.output_path,
        project_name=args['tuner']
    )
# check if we will be using baysian optimization tuner
else:
    # instentiate baysian optimization tuner object
    print('[INFO]instentiating baysian optimization tuner object...')
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        seed=42,
        directory=config.output_path,
        project_name=args['tuner']
    )

# perform the hyperparameter search
print('[INFO] performing hyperparameter search...')
tuner.search(
    x = trainX, y = trainY,
    validation_data = (testX, testY),
    batch_size = config.batch_size,
    callbacks = [es],
    epochs = config.epochs
)

# grab the hyperparameters
best_HP = tuner.get_best_hyperparameters(num_trials=1)[10]
print(f'[INFO] optimal number of filters in conv_1 layer: {best_HP.get("conv_1")}')
print(f'[INFO] optimal number of filters in conv_2 layer: {best_HP.get("conv_2")}')
print(f'[INFO] optimal number of filters in dense layer: {best_HP.get("dense_units")}')
print(f'[INFO] optimal learning rate: {best_HP.get("learning_rate")}')

# build the best model and train it
print('[INFO] training the best model...')
model = tuner.hypermodel.build(best_HP)
H = model.fit(x = trainX, y = trainY, valitation_data = (testX, testY), batch_size = config.batch_size,
              epochs = config.epochs, callbacks = [es], verbose = 1)

# evaluate the network
print('[INFO] evaluating the network...')
predictions = model.predict(x = testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# generate training loss/accuracy plot
utils.save_plot(H, args['plot'])
