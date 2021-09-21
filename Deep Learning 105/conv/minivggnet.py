from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):

        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using Channel first then update the input shape
        if K.image_data_format() == 'channel_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # First CONV => ReLU => Conv => ReLU => Pool layers set
        model.add(Conv2D(32, (3, 3), padding = 'same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Second CONV => ReLU => Conv => ReLU => Pool layers set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # First set of FC => ReLU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
    