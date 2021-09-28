from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation='relu'))
    model.add(Dense(4, activation='relu'))

    # check if regression node should be added
    if regress:
        model.add(Dense(1, activation='linear'))

    return model