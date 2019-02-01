
"""Lambda layers."""

from keras import backend as K


def abs_diff(tensors):
    """Make abs(A - B)."""
    return K.abs(tensors[0] - tensors[1])


def euclidean_distance(tensors):
    """Make euclidean distance."""
    return K.sqrt(
        (K.square(tensors[0] - tensors[1])).sum(axis=1, keepdims=True)
    )
