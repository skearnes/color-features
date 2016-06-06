"""
HDF5 utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import h5py
import numpy as np


def load(filename, load_all=True):
    """
    Load data from HDF5.

    Parameters
    ----------
    filename : str
        Filename.
    load_all : bool, optional (default True)
        Load all datasets into a dictionary.
    """
    if load_all:
        with h5py.File(filename) as f:
            data = {key: np.asarray(f[key]) for key in f.keys()}
        return data
    else:
        return h5py.File(filename)


def dump(data, filename, attrs=None, options=None):
    """
    Dump data to HDF5.

    Parameters
    ----------
    data : dict
        Datasets to serialize.
    filename : str
        Output filename.
    attrs : dict, optional
        Attributes to add to the file.
    options : dict, optional
        Keyword arguments to create_dataset. If None, a standard set of options
        is used.
    """
    if options is None:
        options = {'chunks': True, 'fletcher32': True, 'shuffle': True,
                   'compression': 'gzip'}
    with h5py.File(filename, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value, **options)
        if attrs is not None:
            for key, value in attrs.items():
                f.attrs[key] = value
