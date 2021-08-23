from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from DataPreprocessing.simplepreprocessor import SimplePreprocessor
from DataPreprocessing.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# Construct an argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, help='Number of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='Number of jobs for k-NN distance (-1 uses all '
                                                           'available cores)')
args = vars(ap.parse_args())

# grab the list of images we will be describing
print('[INFO] loading  Images ....')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load dataset from disk  and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.reshape((data.shape[0], 3072))  # 32*32*3

# show imformation on memory consumption of images
print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes/(1024*1024.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# divide the dataset into training and testing set (75/25)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate a k-NN classifier on raw pixel intensities
print('[INFO] evaluating a k-NN classifier ...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))