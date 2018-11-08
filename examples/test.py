
"""Test."""

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from cnn_midlogo import train


cfg = {
    'main': {
        'img_shape': (150, 150, 3),
        'model': 'cnn_midlogo.models.fine_tuning_incv3.get_model',
    },
    'compile': {
        'loss': categorical_crossentropy,
        'optimizer': Adam(lr=0.001),
        'metrics': ['accuracy'],
    },
    'train': {
        'flow': {
            'batch_size': 1,
            'directory': '/tmp/fuu',
            'class_mode': 'categorical',
            'target_size': (150, 150),
        },
        'callbacks': {},
        'data_gen': {
            'shear_range': 0.5,
            'zoom_range': 0.4,
            'rotation_range': 120,
            'width_shift_range': 0.3,
            'height_shift_range': 0.3,
            'channel_shift_range': 150,
            'fill_mode': 'nearest',
            'horizontal_flip': False,
        },
        'fit': {
            'epochs': 1,
            'verbose': 1,
        },
    },
    'validate': {
        'flow': {
            'batch_size': 1,
            'directory': '/tmp/fuu',
            'class_mode': 'categorical',
            'target_size': (150, 150),
        },
        'data_gen': {},
    },
}

res = train.run(cfg)
res.model.summary()
