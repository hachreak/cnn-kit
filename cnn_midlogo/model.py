
""" Keras Model."""

from keras.applications import inception_v3
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D


def _set_readonly(model):
    """Make a model weights readonly."""
    for layer in model.layers:
        layer.trainable = False
    return model


def _classification(model, output_shape):
    """Add classification layers."""
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(output_shape, activation='softmax')(x)

    return Model(inputs=model.input, outputs=x)


def get_model(input_shape, output_shape):
    """Get the model."""
    # load the Inception_V3 model
    model = inception_v3.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # make them readonly
    model = _set_readonly(model)
    # add new classification layers
    model = _classification(model, output_shape)

    return model
