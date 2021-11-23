from . import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_model(hp):
    # initialize the model along with input shape and channel
    model = Sequential()
    input_shape = config.input_shape
    chan_dim = -1

    # first CONV=> Relu => Pool layer set
    model.add(Conv2D(hp.Int('conv_1', min_value=32, max_value=96, step=32), (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Second CONV=> Relu => Pool layer set
    model.add(Conv2D(hp.Int('conv_2', min_value=64, max_value=128, step=32), (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # First set of FC => Relu Layer
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=256, max_value=768, step=256)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #softmax classifier
    model.add(Dense(config.num_classes))
    model.add(Activation('softmax'))

    # initialize the learning rate choices and optimizer
    lr = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    opt = Adam(learning_rate=lr)

    # compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model