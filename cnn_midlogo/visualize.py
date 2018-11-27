
"""Visualize utilities."""


import scipy.ndimage as ndimage
import numpy as np

from sklearn.metrics import classification_report as cr, \
    confusion_matrix as cm
from functools import partial
from keras import activations
from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array
from matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency

from .utils import get_phase_cfg


def plot_saliency(model, x, y_category, y_name, cmap=None, smooth=True):
    """Visualize saliency."""
    # color map
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # Swap softmax with linear
    layer_idx = len(model.layers) - 1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_saliency(
        model, layer_idx, filter_indices=y_category,
        seed_input=x,
        backprop_modifier='guided'
    )

    # Plot with 'jet' colormap to visualize as a heatmap.
    plt.imshow(grads, cmap='jet')
    plt.colorbar()
    plt.suptitle(y_name)

    if smooth:
        smoothe = ndimage.gaussian_filter(grads, sigma=5)
        plt.imshow(smoothe, alpha=.7)

    return plt


def plot_saliency_on_the_fly(model, img_path, config):
    """Plot saliency on the fly."""
    cfg = get_phase_cfg(config, 'saliency')
    imgs = np.array([img_to_array(load_img(img_path, **cfg))])
    flow = next(ImageDataGenerator().flow(imgs))
    return partial(
        plot_saliency, model=model, x=flow[0],
        y_category=range(0, len(config['test']['classes'])),
        y_name=config['test']['classes']
    )


def classification_report(predictions, cfg):
    """Show report."""
    predictions = list(predictions)
    y_true = [p[1] for p in predictions]
    y_pred = [p[2] for p in predictions]
    return cr(y_true, y_pred, labels=cfg['test']['classes'],
              target_names=cfg['test']['classes'])


def confusion_matrix(predictions, cfg):
    """Show confusion matrix."""
    predictions = list(predictions)
    labels = cfg['test']['classes']
    y_true = [p[1] for p in predictions]
    y_pred = [p[2] for p in predictions]
    return cm(y_true, y_pred, labels=cfg['test']['classes']), labels


def print_matrix(matrix, labels):
    """Print a matrix."""
    for i, row in enumerate(matrix):
        to_print = ''.join(['{:8}'.format(round(item, 2)) for item in row])
        print("{0:<15} {1}".format(labels[i], to_print))
