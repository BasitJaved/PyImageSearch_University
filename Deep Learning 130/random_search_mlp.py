import tensorflow as tf
tf.random.set_seed(42)

from Preprocessing.mlp import get_mlp_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.datasets import mnist

# load mnist dataset
print('[INFO] downloading MNIST...')
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Scale data in the range of 0, 1
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

# wrap model into scikit-learn compatible classisifier
print('[INFO] initializing Model...')
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# define a gride of hyperparameter search space
hidden_layer_one=[256, 512, 784]
hidden_layer_two=[128, 256, 512]
learning_rate=[1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batch_size = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]

# create a dictionary from the hyperparameter grid
grid = dict(
    hidden_layer_one=hidden_layer_one,
    learning_rate=learning_rate,
    hidden_layer_two=hidden_layer_two,
    dropout=dropout,
    batch_size=batch_size,
    epochs=epochs
)

# initialize a random search with a 3 fold cross-validation and then start the hyperparameter search process
print('[INFO] performing random search...')
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3, param_distributions=grid, scoring='accuracy')
search_results = searcher.fit(trainX, trainY)

# summarize grid search information
best_score = search_results.best_score_
best_params = search_results.best_params_
print(f'[INFO] best score is {best_score} using {best_params}')

# extract best model, make predictions on data and show a classification report
print('[INFO] evaluating best model')
best_model = search_results.best_estimator_
accuracy = best_model.score(testX, testY)
print(f'Accuracy: {accuracy*100}')