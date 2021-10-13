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