from Preprocessing.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# load the MNIST dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1]
# each image is represented by an 8x8 = 64 dim feature vector
print('[INFO] loading MNIST (sample) dataset ...')
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())
print(f'[INFO] Samples: {data.shape[0]}, dim: {data.shape[1]}')

# construct the training and testing split
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# Convert labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print('[INFO] Training Network ...')
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print(f'[INFO] {nn}')
nn.fit(trainX, trainY, epochs=5000)

# Evaluate the network
print('[INFO] evaluating the network ...')
predictions = nn.predict(testX)
predictions = predictions.argmax(axis = 1)
print(classification_report(testY.argmax(axis = 1), predictions))
