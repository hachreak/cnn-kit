
"""Train."""

from __future__ import absolute_import

from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator

from .utils import callbacks, get_output_shape, load_fun, get_phase_cfg


def _flow_cfg(config, name):
    img_width, img_height, _ = deepcopy(config['main']['img_shape'])
    cfg = deepcopy(config[name]['flow'])
    cfg['target_size'] = (img_height, img_width)
    return cfg


def run(config):
    """Run training."""
    #  img_width, img_height, img_depth = config['main']['img_shape']
    output_shape = get_output_shape(config['train']['flow']['directory'])

    get_model = load_fun(config['main']['model']['name'])

    model = get_model(
        input_shape=config['main']['img_shape'],
        output_shape=output_shape,
        **config['main']['model'].get('args')
    )

    model.compile(**config['compile'])

    gen_train = ImageDataGenerator(**config['train']['data_gen'])
    gen_valid = ImageDataGenerator(**config['validate']['data_gen'])

    flow_train = gen_train.flow_from_directory(
            **get_phase_cfg(config, 'train'))
    flow_valid = gen_valid.flow_from_directory(
            **get_phase_cfg(config, 'validate'))

    cbs = callbacks(config['train']['callbacks'])

    return model.fit_generator(
        flow_train,
        validation_data=flow_valid,
        callbacks=cbs,
        validation_steps=flow_valid.n // flow_valid.batch_size,
        **config['train']['fit']
    )
