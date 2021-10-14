from Preprocessing import config
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
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
kernel = ['linear', 'rbf', 'sigmoid', 'poly']
tolerance = loguniform(1e-6, 1e-3)
c = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol = tolerance, C=c)

# initialize a cross-validation fold and perform a grid-search to tune the hyperparameters
print('[INFO] grid searching over the hyperparameters...')
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1, cv = cvFold,
                                param_distributions=grid, scoring='neg_mean_squared_error')
searchResults = randomSearch.fit(trainX, trainY)

# Extract the best model and Evaluate it
bestModel = searchResults.best_estimator_
print('[INFO] Evaluating...')
print(f'R2: {bestModel.score(testX, testY)}')