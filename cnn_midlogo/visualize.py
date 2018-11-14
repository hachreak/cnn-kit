
"""Visualize utilities."""

import scipy.ndimage as ndimage

from keras import activations
from matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency


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
        model, layer_idx, filter_indices=[y_category],
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
