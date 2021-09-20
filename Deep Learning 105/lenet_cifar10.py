from conv.lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Grab Mnist dataset
print('[INFO] accessing cifar10 dataset')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# if we are using channels first ordering then reshape the design matrix
if K.image_data_format == 'channels_first':
    trainX = trainX.reshape((trainX.shape[0], 3, 32, 32))
    testX = testX.reshape((testX.shape[0], 3, 32, 32))
# otherwise
else:
    trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
    testX = testX.reshape((testX.shape[0], 32, 32, 3))

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
model = LeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# training the network
print('[INFO] Training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=50, verbose=1)

# Evaluate the network
print('[INFO] Evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 50), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 50), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 50), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 50), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
