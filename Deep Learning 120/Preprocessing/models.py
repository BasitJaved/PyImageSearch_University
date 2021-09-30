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

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize input share and channel dimension, assuming Tensorflow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is first CONV layer set input appropriately
        if i == 0:
            x = inputs

        # Conv => Relu => BN => Pooling
        x = Conv2D(f, (3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer matching no of nodes coming out of MLP
    x = Dense(4)(x)
    x = Activation('relu')(x)

    # check to see if regression node should be added
    if regress:
        x = Dense(1, activation='linear')(x)

    # construct the CNN
    model = Model(inputs, x)

    return model
