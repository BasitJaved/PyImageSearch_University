from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, MaxPool2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras import backend as K

class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be channels last
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using channels first then update the input shape
        if K.image_data_format()=='channels_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # Conv =>Relu => Pooling
        model.add(SeparableConv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (Conv =>Relu)*2  => Pooling
        model.add(SeparableConv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (Conv =>Relu)*3  => Pooling
        model.add(SeparableConv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # First and only set of FC => Relu layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # binary Classifier
        model.add(Dense(classes, activation='sigmoid', bias_initializer='zeros'))

        return model