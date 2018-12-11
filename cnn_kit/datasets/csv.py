
"""Load and save dataset from/to csv."""

import errno
import os

from .. import preprocess as pr


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


def find_files(csv_file, src_dir):
    """Find dataset files from csv file."""
    fcsv = csv_file[1:]
    filenames = [name for name, _, _ in fcsv]
    exts = _get_files_ext(filenames)
    return filter(
        lambda x: os.path.basename(x) in filenames,
        pr.get_files(src_dir, types=exts)
    )


def _get_files_ext(filenames):
    """Get filenames extentions."""
    return set([os.path.splitext(filename)[1] for filename in filenames])


def create_symlinks(srcs_dsts):
    """create file symlinks."""
    for src, dst in srcs_dsts:
        directory, _ = os.path.split(dst)
        _create_dir(directory)
        _create_symlink(src, dst, True)


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

