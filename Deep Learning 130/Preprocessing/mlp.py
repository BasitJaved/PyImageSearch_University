from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def get_mlp_model(hidden_layer_one=784, hidden_layer_two=256, dropout=0.2, learning_rate=0.01):
    # initialize a sequential model and add layer to flatten the input data
    model = Sequential()
    model.add(Flatten())

    # Add two blocks of FC => ReLU => Dropout
    model.add(Dense(hidden_layer_one, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_two, activation='relu'))
    model.add(Dropout(dropout))

    # add softmax layer on top
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model