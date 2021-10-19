import tensorflow as tf
tf.random.set_seed(42)

from Preprocessing.mlp import get_mlp_model
from tensorflow.keras.datasets import mnist

# load the mnist dataset
((trainX, trainY), (testX, testY)) = mnist.load_data()

# scale the data to the range [0, 1]
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

# initialize model with default hyper parameters
print('[INFO] initializing model...')
model = get_mlp_model()

#train the network
print('[INFO] Training Model...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=8, epochs=20)

# make predictions on test set and evaluate it
print('[INFO] Evaluating network...')
accuracy = model.evaluate(testX, testY)[1]
print(f'Accuracy: {accuracy*100}')