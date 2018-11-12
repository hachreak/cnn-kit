
"""Predict."""

import numpy as np

from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator


def _most_probable(model_result):
    """Get most probable class from prediction result."""
    return np.argmax(model_result)


def predict(model, imgs, classes):
    """Make a prediction."""
    return classes[_most_probable(model.predict(imgs))]


def _flow_cfg(config, name):
    img_width, img_height, _ = deepcopy(config['main']['img_shape'])
    cfg = deepcopy(config[name]['flow'])
    cfg['shuffle'] = False
    cfg['target_size'] = (img_height, img_width)
    return cfg


def predict_on_the_fly(model, config):
    """Run prediction on the fly on a directory."""
    classes = config['test']['classes']
    gen = ImageDataGenerator()
    flow = gen.flow_from_directory(**_flow_cfg(config, 'test'))
    pred = model.predict_generator(flow)
    return [
        (flow.filenames[i], classes[_most_probable(p)])
        for i, p in enumerate(pred)
    ]
