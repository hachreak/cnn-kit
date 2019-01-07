
"""Test."""

import sys
import numpy as np

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import load_model

from cnn_kit import train, predict, visualize, preprocess as pr


cfg = {
    'main': {
        'img_shape': (500, 500, 3),
        'model': {
            'name': 'cnn_kit.models.fine_tuning_vgg19.get_model',
            #  'name': 'cnn_kit.models.fine_tuning_incv3.get_model',
            #  'name': 'cnn_kit.models.fine_tuning_nasnetlarge.get_model',
            'args': {
                #  'readonly_until': 200,
                'readonly_until': 18,
            },
        },
    },
    'compile': {
        'loss': categorical_crossentropy,
        'optimizer': Adam(0.0001),
        'metrics': ['accuracy'],
    },
    'train': {
        'flow': {
            'batch_size': 8,
            'directory': '/tmp/dataset/train',
            'class_mode': 'categorical',
            'shuffle': True,
            #  'save_to_dir': '/tmp/generated',
            #  'save_format': 'jpeg',
        },
        'callbacks': {
            #  'keras.callbacks.EarlyStopping': {
            #      'monitor': 'val_loss',
            #      'min_delta': 0,
            #      'patience': 10,
            #      'verbose': 1,
            #      'mode': 'auto',
            #      'restore_best_weights': True,
            #  },
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
            #  'zoom_range': (0.8, 1),
            #  'rotation_range': 100,
            #  'width_shift_range': 0.05,
            #  'height_shift_range': 0.05,
            'channel_shift_range': 0.1,
            'fill_mode': 'nearest',
            'horizontal_flip': True,
            'vertical_flip': True,
            'rescale': 1./255,
        },
        'fit': {
            'epochs': 60,
            'steps_per_epoch': 400,
            'verbose': 1,
            'class_weight': [],
        },
    },
    'validate': {
        'flow': {
            'batch_size': 8,
            'directory': '/tmp/dataset/test',
            'class_mode': 'categorical',
        },
        'data_gen': {
            'rescale': 1./255,
        },
    },
    'test': {
        'data_gen': {
            'rescale': 1./255,
        },
        'classes': ['medium', 'no', 'regular'],
        'flow': {
            'batch_size': 1,
            'directory': '/tmp/tocheck',
        },
        'normalize_cm': True,
    },
}


def do_predict(cfg, threshold):
    """Do predict."""
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]

    model = load_model(name)
    for filename, y_true, y_pred in predict.predict_on_the_fly(
            model, cfg, choose=predict.over_threshold(threshold)):
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


def do_confusion_matrix(cfg):
    """Do confusion matrix."""
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]

    get_matrix = (lambda x: x) if not cfg['test']['normalize_cm'] \
        else visualize.normalize_cm
    model = load_model(name)
    print("Confusion matrix:\n")
    cm, labels = visualize.confusion_matrix(
        predict.predict_on_the_fly(model, cfg), cfg
    )
    visualize.print_matrix(get_matrix(cm), labels)


def do_wrong(cfg, threshold):
    """Show wrong crassified."""
    if len(sys.argv) < 3:
        print_menu(sys.argv)
        sys.exit(1)

    name = sys.argv[2]

    model = load_model(name)
    visualize.print_wrong(predict.predict_on_the_fly(
        model, cfg, choose=predict.over_threshold(threshold)
    ))


def print_menu(args):
    """Print menu."""
    print("{0} train".format(args[0]))
    print("{0} predict [model_name]".format(args[0]))
    print("{0} saliency [model_name] [img_path]".format(args[0]))
    print("{0} report [model_name]".format(args[0]))
    print("{0} cm [model_name]".format(args[0]))
    print("{0} wrong [model_name]".format(args[0]))


if len(sys.argv) < 2:
    print_menu(sys.argv)
    sys.exit(1)

main_arg = sys.argv[1]

cfg['train']['fit']['class_weight'] = pr.get_class_weigth(
    cfg['train']['flow']['directory'], ['.tif', '.jpg']
)

if main_arg == 'train':
    train.run(cfg)
elif main_arg == 'predict':
    do_predict(cfg, 0.8)
elif main_arg == 'saliency':
    do_saliency(cfg)
elif main_arg == 'report':
    do_report(cfg)
elif main_arg == 'cm':
    do_confusion_matrix(cfg)
elif main_arg == 'wrong':
    do_wrong(cfg, 0.8)
else:
    print_menu(sys.argv)
