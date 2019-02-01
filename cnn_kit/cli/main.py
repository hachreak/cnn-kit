
"""CLI."""

import click

from .dataset import dataset
from .neuralnet import nn


@click.group()
def cli():
    pass


cli.add_command(dataset)
cli.add_command(nn)
