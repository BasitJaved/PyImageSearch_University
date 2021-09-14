from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from conv.shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

# Convert labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize label names for Cifar-10 dataset
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize optimizer and model
print('[INFO] compiling Model...')
opt = SGD(lr = 0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss= 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train model
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=50, verbose=1)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=['cat', 'dog', 'panda']))

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
