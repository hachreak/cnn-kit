
""" Keras Model."""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D

from ..layers.focus import Focus


def get_model(input_shape, output_shape):
    """Get the model."""
    model = Sequential()
    # load basic model
    model.add(Focus(filters=32, input_shape=input_shape))
    model.add(Focus(filters=32))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add new classification layers
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_shape, activation='softmax'))

    return model
