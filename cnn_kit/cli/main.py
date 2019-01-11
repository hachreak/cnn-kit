
"""CLI."""

import click
import csv as _csv

from .validators import validate_csv_file, validate_directory
from ..datasets import csv as c, get_duplicates


@click.group()
def cli():
    pass


@cli.group()
def dataset():
    pass


@dataset.group()
def csv():
    pass


@csv.command()
@click.argument('csv_file', callback=validate_csv_file, type=click.File('rb'))
@click.argument('src_dir', callback=validate_directory)
@click.argument('dst_dir')
@click.option('-s', '--symlinks', is_flag=True, default=False)
def build(csv_file, src_dir, dst_dir, symlinks):
    csv_file = list(csv_file)
    files = c.find_files(c.get_column(0, csv_file), src_dir)
    c.copy_files(
        c.build_dataset(csv_file, files, dst_dir), is_symlinks=symlinks
    )


@csv.command()
@click.argument('csv_file', callback=validate_csv_file, type=click.File('rb'))
@click.argument('src_dir', callback=validate_directory)
def check(csv_file, src_dir):
    csv_file = list(csv_file)
    csv_filenames = c.get_column(0, csv_file)
    found_filenames = c.basenames(c.find_files(csv_filenames, src_dir))
    not_found = set(csv_filenames) - set(found_filenames)
    for f in not_found:
        print(f)


@csv.command()
@click.argument('csv_file', type=click.File('wb'))
@click.argument('src_dir', callback=validate_directory)
@click.option('--types', '-t', multiple=True)
def save(csv_file, src_dir, types):
    types = types or None
    writer = _csv.writer(csv_file, delimiter=',', quotechar='"')
    c.create_csv(writer, src_dir, types)


@dataset.group()
def files():
    pass


@files.command()
@click.argument('src_dir', callback=validate_directory)
@click.option('--types', '-t', multiple=True)
def dups(src_dir, types):
    types = types or None
    for f in get_duplicates(src_dir, types):
        print(f)
