
"""Console interface."""

from __future__ import absolute_import

from keras.preprocessing.image import ImageDataGenerator

from .utils import callbacks, get_output_shape, load_fun


def run(config):
    """Run training."""
    img_width, img_height, img_depth = config['main']['img_shape']
    output_shape = get_output_shape(config['train']['flow']['directory'])

    get_model = load_fun(config['main']['model'])

    model = get_model(config['main']['img_shape'], output_shape=output_shape)

    model.compile(**config['compile'])

    gen_train = ImageDataGenerator(**config['train']['data_gen'])
    gen_valid = ImageDataGenerator(**config['validate']['data_gen'])

    flow_train = gen_train.flow_from_directory(**config['train']['flow'])
    flow_valid = gen_valid.flow_from_directory(**config['validate']['flow'])

    cbs = callbacks(config['train']['callbacks'])

    return model.fit_generator(
        flow_train,
        validation_data=flow_valid,
        callbacks=cbs,
        **config['train']['fit']
    )
