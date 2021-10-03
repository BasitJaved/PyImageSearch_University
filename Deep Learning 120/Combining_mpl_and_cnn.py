from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model
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

# construct the path to input txt file that contains information on each house in dataset and then load dataset
print('[INFO] loading house attributes')
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

# process the house attributes data by performing min-max scaling on continuous data and one-hot encoding
# on categorical data and concatenating then together
print('[INFO] processing data...')
(trainAttrX, testAttrX) = datasets.process_house_attributes(df, trainAttrX, testAttrX)

# create mlp and cnn models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the output of both MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
print(combinedInput.shape)

# our final FC layer head will have two dense layers the final one being our regression head
x = Dense(4, activation='relu')(combinedInput)
x = Dense(1, activation='linear')(x)

# our final model will accept categorical/numerical data on mlp input and images on CNN input,
# outputting a single value
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute error as our loss implying that we seek to minimize the absolute
# percentage difference between our price *predictions* and actual price
opt = Adam(learning_rate=1e-3, decay=1e-3 / 200)
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

# training model
print('[INFO] training model...')
model.fit(x=[trainAttrX, trainImagesX], y = trainY, validation_data=([testAttrX, testImagesX], testY),
          epochs=200, batch_size=8)

# Make predictions on testing data
print('[INFO] predicting the house price...')
preds = model.predict([testAttrX, testImagesX])

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

