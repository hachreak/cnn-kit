
"""CLI."""

import click

from .validators import validate_csv_file, validate_directory
from ..datasets.csv import find_files


@click.group()
def cli():
    pass


@cli.group()
def dataset():
    pass


@dataset.group()
def load():
    pass


@load.command()
@click.argument('csv_file', callback=validate_csv_file, type=click.File('rb'))
@click.argument('src_dir', callback=validate_directory)
@click.argument('dst_dir', callback=validate_directory)
def csv(csv_file, src_dir, dst_dir):
    csv_file = list(csv_file)
    files = find_files(csv_file, src_dir)
    for i in files:
        print(i)
