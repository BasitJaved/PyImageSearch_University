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