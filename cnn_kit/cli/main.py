
"""CLI."""

import os
import click

from .dataset import dataset


@click.group()
def cli():
    pass


cli.add_command(dataset)
