from Preprocessing import config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd

# load the dataset, seperate the features and labels, and perform a training and testing split using 85% of
# data for training and 15% for evaluation
print('[INFO] loading data...')
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, random_state=3, test_size=0.15)

# standardise the feature values by computing the mean, subtracting the mean from data points and then dividing
# by standard deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# train the model with no hyperparameter tuning
print('[INFO] training our support vector regression model...')
model = SVR()
model.fit(trainX, trainY)

# Evaluate our model using R^2-score (1.0 is best value)
print('[INFO] Evaluating...')
print(f'R2: {model.score(testX, testY)}')