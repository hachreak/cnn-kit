
"""Neural network CLI."""

import click

from .validators import validate_json
from .. import preprocess as pr, utils, train as tr


@click.group()
def nn():
    pass


@nn.command()
#  @click.argument('model', callback=validate_model)
#  @click.option('--types', '-t', multiple=True)
@click.argument('cfg', callback=validate_json)
def train(cfg):
    """Predict."""
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
