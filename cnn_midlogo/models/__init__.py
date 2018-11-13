
"""Models."""

from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D


def _set_readonly(model, until=None):
    """Make a model weights readonly."""
    for layer in model.layers[:until]:
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
