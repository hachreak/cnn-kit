
"""Datasets."""

import os

from collections import defaultdict

from .. import preprocess as pr


def get_duplicates(directory, types=None):
    """Check if some files and in train and test at same time."""
    files = [os.path.basename(n) for n in pr.get_files(directory, types=types)]
    counter = defaultdict(int)
    for f in files:
        counter[f] += 1
    return [k for k, v in counter.items() if v > 1]
