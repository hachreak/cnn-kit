
"""Utils."""

from __future__ import absolute_import

import os
import importlib

from copy import deepcopy


def load_fun(name):
    """Load a function from name."""
    module, fun_name = name.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, fun_name)


def callbacks(config):
    """Dinamically load callbacks."""
    return [load_fun(key)(**value) for key, value in config.items()]


def get_output_shape(directory):
    """Get output shape checking dataset structure."""
    return len(os.listdir(directory))


def get_phase_cfg(config, name):
    img_width, img_height, _ = deepcopy(config['main']['img_shape'])
    cfg = deepcopy(config.get(name, {}).get('flow', {}))
    cfg['target_size'] = img_width, img_height
    return cfg
