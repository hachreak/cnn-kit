
"""Preprocess images."""

import dlib
import cv2
import numpy as np
import os

from PIL import Image, ImageChops


def batch(block, size):
    """Split in batches the input."""
    for block in np.array_split(block, size):
        yield block


def stream(fun, stream):
    """Transform a function into a stream."""
    for value in stream:
        yield fun(value)


def remove_basepath(filenames, basepath):
    """Remove basepath from filename list."""
    length = len(basepath)
    if not basepath.endswith('/'):
        length += 1
    return [f[length:] for f in filenames]


def split_files_per_class(filenames):
    classes = {}
    for f in filenames:
        key, value = f.split('/', 1)
        try:
            classes[key].append(value)
        except KeyError:
            classes[key] = []
    return classes


def get_files(directory, types=None, get_all=False):
    """Get list of images reading recursively."""
    types = types or ['.jpg']
    for root, dirnames, files in os.walk(directory):
        for name in files:
            _, ext = os.path.splitext(name)
            if get_all or ext.lower() in types:
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


def get_class_weigth(train_dir, file_types):
    files = split_files_per_class(remove_basepath(
        get_files(train_dir, types=file_types),
        train_dir
    ))
    sizes = np.array([float(len(files[key])) for key in sorted(files.keys())])
    biggest = max(sizes)
    return biggest / sizes


def crop_white(filename):
    """Crop white from the image."""
    img = Image.open(filename)
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img
