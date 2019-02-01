
"""Siamese network.

See original model: https://github.com/akshaysharma096/Siamese-Networks
"""

from keras.layers import Input, Lambda, Dense
from keras.models import Model

from ..layers import _lambda


def get_model(input_shape, model, output_shape, fun=None):
    """Get the model."""
    fun = fun or _lambda.abs_diff

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(fun)
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(output_shape, activation='sigmoid')(L1_distance)

    return Model(inputs=[left_input, right_input], outputs=prediction)
