from conv.lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Grab Mnist dataset
print('[INFO] accessing MNIST dataset')
((trainX, trainY), (testX, testY)) = mnist.load_data()

# if we are using channels first ordering then reshape the design matrix
if K.image_data_format == 'channels_first':
    trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
    testX = testX.reshape((testX.shape[0], 1, 28, 28))
# otherwise
else:
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

# Scale the data to the range of [0, 1]
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

# Convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

# initialize the optimizer and model
print('[INFO] Compiling model ...')
opt = SGD(lr = 0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# training the network
print('[INFO] Training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=30, verbose=1)

# Evaluate the network
print('[INFO] Evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 30), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 30), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 30), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 30), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
