
"""Load and save dataset from/to csv."""

import errno
import os

from shutil import copyfile

from .. import preprocess as pr


def create_csv(csvfile, directory, types):
    """Create a csv file from a dataset."""
    filenames = pr.split_files_per_class(
        pr.remove_basepath(pr.get_files(directory, types=types), directory)
    )
    # write headers
    csvfile.writerow(['name', 'set', 'class'])
    # write files
    for s, flist in filenames.items():
        for c, filenames in pr.split_files_per_class(flist).items():
            for f in filenames:
                csvfile.writerow([f, s, c])


def build_dataset(csv_file, filepaths, dst_dir):
    """Build dataset from csv, using images found."""
    for name, set_, class_ in csv_file[1:]:
        # get full file path from only the name
        paths = filter(
            lambda x: x.endswith("/{0}".format(name)), filepaths
        )
        # only if file is found
        if len(paths) > 0:
            fname = paths[0]
            yield fname, os.path.join(dst_dir, set_, class_, name)


def find_files(filenames, src_dir):
    """Find dataset files from csv file."""
    exts = _get_files_ext(filenames)
    return filter(
        lambda x: os.path.basename(x) in filenames,
        pr.get_files(src_dir, types=exts)
    )


def _get_files_ext(filenames):
    """Get filenames extentions."""
    return set([os.path.splitext(filename)[1] for filename in filenames])


def copy_files(srcs_dsts, is_symlinks=True):
    """create file symlinks."""
    create = _create_symlink if is_symlinks else _copy_file
    for src, dst in srcs_dsts:
        directory, _ = os.path.split(dst)
        _create_dir(directory)
        create(src, dst, True)


def _copy_file(src, dst, force=True):
    """Copy file from src to dst."""
    copyfile(src, dst)


def _create_symlink(src, dst, force=True):
    print("{0} -> {1}".format(src, dst))
    try:
        os.symlink(src, dst)
    except OSError as exc:
        if exc.errno == errno.EEXIST and force:
            print("remove {0}".format(dst))
            os.remove(dst)
            _create_symlink(src, dst, force)


def _create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_column(column, csv_file):
    fcsv = csv_file[1:]
    return [fields[column] for fields in fcsv]


def basenames(filenames):
    """Get base names from full path."""
    return map(lambda x: os.path.basename(x), filenames)
