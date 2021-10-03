from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Preprocessing import datasets, models
import numpy as np
import argparse
import locale
import os

# construct the arggument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', type=str, required=True, help='Path to input dataset of house images')
args = vars(ap.parse_args())

# construct the path to input text file that contains information on each house in dataset and load dataset
print('[INFO] loading house attributes ...')
inputPath = os.path.sep.join([args['dataset'], 'HousesInfo.txt'])
df = datasets.load_house_attributes(inputPath)

# load house images and scale pixel intensities to range [0, 1]
print('[INFO] loading house images...')
images = datasets.load_house_images(df, args['dataset'])
images = images/ 255.0

# Construct the training and testing split
print('[INFO] constructing Training/Testing splits...')
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largset house in training set and use it to scale house prices to range [0, 1]
maxPrice = trainAttrX['price'].max()
trainY = trainAttrX['price']/maxPrice
testY = testAttrX['price']/maxPrice

# Create CNN and compile model using mean absolute percentage error as loss implying we seek to minimize
# the absolute percentage difference between our prices
model = models.create_cnn(64, 64, 3, regress=True)
opt = Adam(learning_rate=1e-3, decay= 1e-3/200)
model.compile(loss = 'mean_absolute_percentage_error', optimizer=opt)

# training model
print('[INFO] training model...')
model.fit(x = trainImagesX, y = trainY, validation_data=(testImagesX, testY), epochs=200, batch_size=8)

# Make predictions on testing data
print('[INFO] predicting the house price...')
preds = model.predict(testImagesX)

# compute the difference between predicted house prices and the actual house price, then compute the percentile
# difference and absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff/testY) * 100
absPercentDiff = np.abs(percentDiff)

# Compute mean and standard Deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# show some stats on our model
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
print(f'[INFO] Avg. House Price: {locale.currency(df["price"].mean(),grouping=True)}, '
      f'std House Price: {locale.currency(df["price"].std(), grouping=True)}')
print(f'mean: {mean}, std: {std}')
