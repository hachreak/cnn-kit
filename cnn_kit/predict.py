
"""Predict."""

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from .utils import get_phase_cfg
from .exc import NoPrediction


def most_probable(model_result):
    """Get most probable class from prediction result."""
    return np.argmax(model_result)


def over_threshold(threshold):
    """Get only prediction over threshold."""
    over = np.vectorize(lambda x: 1 if x > threshold else 0)

    def f(model_result):
        res = over(model_result)
        if res.sum() != 1:
            raise NoPrediction()
        return most_probable(res)
    return f


def predict(model, imgs, classes):
    """Make a prediction."""
    return classes[most_probable(model.predict(imgs))]


def _flow_cfg(config, name):
    cfg = get_phase_cfg(config, name)
    cfg['shuffle'] = False
    return cfg


def predict_on_the_fly(model, config, choose=None):
    """Run prediction on the fly on a directory."""
    choose = choose or most_probable
    classes = config['test']['classes']
    gen = ImageDataGenerator(**config['test']['data_gen'])
    flow = gen.flow_from_directory(**_flow_cfg(config, 'test'))
    pred = model.predict_generator(
        flow,
        steps=flow.n // flow.batch_size,
    )
    for i, p in enumerate(pred):
        try:
            yield (flow.filenames[i], classes[flow.classes[i]],
                   classes[choose(p)])
        except NoPrediction:
            pass
