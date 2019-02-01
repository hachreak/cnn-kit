
"""Losses."""

from keras import backend as K


def contrastive_loss(margin=1):
    """Contrastive loss from Hadsell-et-al.'06

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def f(y, d):
        return K.mean(
            y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0))
        )
    return f
