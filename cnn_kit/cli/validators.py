
"""CLI validators."""

import click
import csv
import os


def validate_csv_file(ctx, param, value):
    if not value.name.endswith('.csv'):
        raise click.BadParameter('Bad csv file')
    reader = csv.reader(value)
    return reader


def validate_directory(ctx, param, value):
    if not os.path.exists(value):
        raise click.BadParameter("Directory {0} doesn't exists".format(value))
    return value
