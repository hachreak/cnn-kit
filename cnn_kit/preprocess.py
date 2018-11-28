
"""Preprocess images."""

import dlib
import cv2
import numpy as np
import os


def batch(block, size):
    """Split in batches the input."""
    for block in np.array_split(block, size):
        yield block


def stream(fun, stream):
    """Transform a function into a stream."""
    for value in stream:
        yield fun(value)


def get_files(directory, types=None):
    """Get list of images reading recursively."""
    types = types or ['.jpg']
    for root, dirnames, files in os.walk(directory):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in types:
                yield os.path.join(root, name)


def load_img(filename):
    """Load a image from file."""
    try:
        return dlib.load_rgb_image(filename)
    except RuntimeError:
        return cv2.imread(filename)


def rgb_to_bn(matrix):
    """Adapt rgb image to input for the CNN as b/n image."""
    return cv2.cvtColor(matrix, cv2.COLOR_RGB2GRAY)


def matrix_to_bn(batch_x):
    """Adapt matrix to input for the CNN as b/n image."""
    (img_x, img_y) = batch_x[0].shape
    return batch_x.reshape(batch_x.shape[0], img_x, img_y, 1)


def astype(name):
    """Convert matrix to this new type."""
    def f(matrix):
        return matrix.astype(name)
    return f


def normalize(max_):
    """Normalize matrix."""
    def f(matrix):
        matrix = matrix.astype('float32')
        matrix /= max_
        return matrix
    return f
