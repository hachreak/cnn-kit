
"""Test."""

import sys

from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.models import load_model

from cnn_midlogo import train, predict, visualize


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
        'optimizer': SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True),
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
            'shear_range': 0,
            'zoom_range': (0.8, 1),
            'rotation_range': 1200,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'channel_shift_range': 70,
            'fill_mode': 'nearest',
            'horizontal_flip': True,
            'vertical_flip': True,
        },
        'fit': {
            'epochs': 60,
            'steps_per_epoch': 300,
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
    },
}


def do_predict(cfg):
    """Do predict."""
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]

    model = load_model(name)
    for filename, y_true, y_pred in predict.predict_on_the_fly(model, cfg):
        print("{0} -> {1}".format(filename, y_pred))


def do_saliency(cfg):
    """Do saliency."""
    if len(sys.argv) < 4:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]
    img_path = sys.argv[3]

    model = load_model(name)
    plt = visualize.plot_saliency_on_the_fly(model, img_path, cfg)
    plt().show()


def do_report(cfg):
    """Do report."""
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]

    model = load_model(name)
    print(visualize.classification_report(
        predict.predict_on_the_fly(model, cfg), cfg
    ))


def print_menu(args):
    """Print menu."""
    print("{0} train".format(args[0]))
    print("{0} predict [model_name]".format(args[0]))
    print("{0} saliency [model_name] [img_path]".format(args[0]))
    print("{0} report [model_name]".format(args[0]))


if len(sys.argv) < 2:
    print_menu(sys.argv)
    sys.exit(1)

main_arg = sys.argv[1]

if main_arg == 'train':
    train.run(cfg)
elif main_arg == 'predict':
    do_predict(cfg)
elif main_arg == 'saliency':
    do_saliency(cfg)
else:
    do_report(cfg)
