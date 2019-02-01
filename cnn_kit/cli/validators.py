
"""CLI validators."""

import click
import csv
import json
import os


def validate_csv_file(ctx, param, value):
    """Check if exists and load reader."""
    if not value.name.endswith('.csv'):
        raise click.BadParameter('Bad csv file')
    reader = csv.reader(value)
    return reader


def validate_directory(ctx, param, value):
    """Check directory exists."""
    if not os.path.exists(value):
        raise click.BadParameter("Directory {0} doesn't exist".format(value))
    return value


def validate_model(ctx, param, value):
    """Check model exists and load it."""
    from keras.models import load_model
    if not os.path.isfile(value):
        raise click.BadParameter("Model {0} doesn't exist".format(value))
    return load_model(value)


def validate_json(ctx, param, value):
    """Check json and load it."""
    if not os.path.isfile(value):
        raise click.BadParameter("Model {0} doesn't exist".format(value))
    with open(value) as json_file:
        content = json.load(json_file)
    return content
