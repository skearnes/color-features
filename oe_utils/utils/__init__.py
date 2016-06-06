"""
Miscellaneous utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import cPickle
import gzip


def read_pickle(filename):
    """
    Read pickle from a (possibly gzipped) file.

    Parameters
    ----------
    filename : str
        Filename.
    """
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename)
    data = cPickle.load(f)
    f.close()
    return data


def write_pickle(data, filename, protocol=cPickle.HIGHEST_PROTOCOL):
    """
    Write pickle to a (possibly gzipped) file.

    Parameters
    ----------
    data : object
        Object to pickle.
    filename : str
        Output filename.
    protocol : int, optional (default cPickle.HIGHEST_PROTOCOL)
        Pickle protocol version.
    """
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'wb')
    else:
        f = open(filename, 'wb')
    cPickle.dump(data, f, protocol)
    f.close()
