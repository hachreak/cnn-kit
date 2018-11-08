
"""Utils."""

from __future__ import absolute_import

import os
import importlib


def load_fun(name):
    """Load a function from name."""
    module, fun_name = name.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, fun_name)


def callbacks(config):
    """Dinamically load callbacks."""
    return [load_fun(key)(**value) for key, value in config.items()]


def get_output_shape(directory):
    """Get output shape."""
    return len(os.listdir(directory))
