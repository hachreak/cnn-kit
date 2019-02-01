
"""Neural network CLI."""

import click

from .validators import validate_json, validate_model
from .. import preprocess as pr, utils, train as tr, predict as pred, visualize


@click.group()
def nn():
    pass


@nn.command()
@click.argument('cfg', callback=validate_json)
def train(cfg):
    """Train neural network."""
    cfg['train']['fit']['class_weight'] = pr.get_class_weigth(
        cfg['train']['flow']['directory'], cfg['main'].get('img_types')
    )
    cfg['compile']['loss'] = utils.load_fun(cfg['compile']['loss'])
    cfg['compile']['optimizer'] = utils.load_fun(
        cfg['compile']['optimizer']['name'])(
            **cfg['compile']['optimizer']['kwargs']
    )

    rescale_factor = cfg['main'].get('rescale_factor', 1)
    cfg['train']['data_gen']['rescale'] = 1. / rescale_factor

    tr.run(cfg)


@nn.command()
@click.argument('cfg', callback=validate_json)
@click.argument('model', callback=validate_model)
@click.option('--threshold', '-t', type=float, default=0.5)
def predict(cfg, model, threshold):
    """Predict."""
    rescale_factor = cfg['main'].get('rescale_factor', 1)
    cfg['test']['data_gen']['rescale'] = 1. / rescale_factor

    for filename, y_true, y_pred in pred.predict_on_the_fly(
            model, cfg, choose=pred.over_threshold(threshold)):
        print("{0}, {1}".format(y_pred, filename))


@nn.command()
@click.argument('cfg', callback=validate_json)
@click.argument('model', callback=validate_model)
@click.option('--threshold', '-t', type=float, default=0.5)
def cm(cfg, model, threshold):
    """Confusion matrix."""
    rescale_factor = cfg['main'].get('rescale_factor', 1)
    cfg['test']['data_gen']['rescale'] = 1. / rescale_factor

    get_matrix = (lambda x: x) if not cfg['test']['normalize_cm'] \
        else visualize.normalize_cm

    print("Confusion matrix:\n")
    cm, labels = visualize.confusion_matrix(
        pred.predict_on_the_fly(model, cfg), cfg
    )
    visualize.print_matrix(get_matrix(cm), labels)
