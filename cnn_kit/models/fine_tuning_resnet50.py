
""" Keras Model."""

from keras.applications import resnet50

from . import _set_readonly, _classification


def get_model(input_shape, output_shape, readonly_until):
    """Get the model."""
    # load the model
    model = resnet50.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # make them readonly
    model = _set_readonly(model, readonly_until)
    # add new classification layers
    model = _classification(model, output_shape)

    return model
