from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DataPreprocessing.simplepreprocessor import SimplePreprocessor
from DataPreprocessing.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
args = vars(ap.parse_args())

# Grab the list of image paths
print('[INFO] Loading Images ...')
imagePaths = list(paths.list_images(args['dataset']))

# Initialize the image preprocessor, load dataset from disk, and reshape data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# loop over the set of regularizers
for i in (None, 'l1', 'l2'):
    # Train a SGD calssifier using softmax loss function and specified regularizer for 25 epochs
    print(f'[INFO] Training model with {i} penalty.')
    model = SGDClassifier(loss='log', penalty=i, max_iter=10, learning_rate='constant', tol=1e-3,
                          eta0=0.01, random_state=12)
    model.fit(trainX, trainY)

    # Evaluate the classifier
    acc = model.score(testX, testY)
    print(f'[INFO] {i} Penalty accuracy: {acc*100}')
