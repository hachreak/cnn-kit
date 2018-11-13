
"""Test."""

import sys

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import load_model

from cnn_midlogo import train, predict


cfg = {
    'main': {
        'img_shape': (500, 500, 3),
        'model': {
            'name': 'cnn_midlogo.models.fine_tuning_vgg19.get_model',
            'args': {
                'readonly_until': 18,
            },
        },
    },
    'compile': {
        'loss': categorical_crossentropy,
        'optimizer': Adam(lr=0.001),
        'metrics': ['accuracy'],
    },
    'train': {
        'flow': {
            'batch_size': 8,
            'directory': '/tmp/dataset/train',
            'class_mode': 'categorical',
            'shuffle': True,
        },
        'callbacks': {
            'keras.callbacks.EarlyStopping': {
                'monitor': 'val_loss',
                'min_delta': 0,
                'patience': 10,
                'verbose': 1,
                'mode': 'auto',
                'restore_best_weights': True,
            },
            'keras.callbacks.ModelCheckpoint': {
                'filepath': 'model-{epoch:02d}-{val_acc:.2f}.hdf5',
                'monitor': 'val_acc',
                'save_best_only': True,
                'save_weights_only': False,
                'mode': 'max',
                'verbose': 1,
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
            'epochs': 50,
            'steps_per_epoch': 80,
            'validation_steps': 10,
            'verbose': 1,
        },
    },
    'validate': {
        'flow': {
            'batch_size': 8,
            'directory': '/tmp/dataset/validate',
            'class_mode': 'categorical',
        },
        'data_gen': {},
    },
    'test': {
        'classes': ['medium', 'no'],
        'flow': {
            'batch_size': 1,
            'directory': '/tmp/dataset/validate',
        },
    }
}


def print_menu(args):
    """Print menu."""
    print("{0} train".format(args[0]))
    print("{0} predict [model_name]".format(args[0]))


if len(sys.argv) < 2:
    print_menu(sys.argv)
    sys.exit(1)

if sys.argv[1] == 'train':
    train.run(cfg)
else:
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]
    model = load_model(name)
    for filename, prediction in predict.predict_on_the_fly(model, cfg):
        print("{0} -> {1}".format(filename, prediction))
