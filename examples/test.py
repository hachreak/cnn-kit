
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
            'shuffle': True,
        },
        'callbacks': {
            'keras.callbacks.EarlyStopping': {
                'monitor': 'val_loss',
                'min_delta': 0.1,
                'patience': 10,
                'verbose': 1,
                'mode': 'auto',
                'restore_best_weights': True,
            },
        },
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
            'step_per_epochs': 5,
            'validation_steps': 3,
            'verbose': 1,
        },
    },
    'validate': {
        'flow': {
            'batch_size': 1,
            'directory': '/tmp/fuu',
            'class_mode': 'categorical',
        },
        'data_gen': {},
    },
}

res = train.run(cfg)
res.model.summary()
