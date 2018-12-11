
"""Load and save dataset from/to csv."""

import os

from .. import preprocess as pr


def find_files(csv_file, src_dir):
    """Find dataset files from csv file."""
    fcsv = csv_file[1:]
    filenames = [name for name, _, _ in fcsv]
    exts = get_files_ext(filenames)
    return filter(
        lambda x: os.path.basename(x) in filenames,
        pr.get_files(src_dir, types=exts)
    )


def get_files_ext(filenames):
    """Get filenames extentions."""
    return set([os.path.splitext(filename)[1] for filename in filenames])
