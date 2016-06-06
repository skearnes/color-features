"""
Utilities for analyzing ROCS output.

Needs to handle:
* Multiple queries per output file (multiple output files)
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import pandas as pd


class ROCS(object):
    """
    Analyze ROCS output.

    Parameters
    ----------
    filename : str
        Name of ROCS output file.
    """
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        """
        Parse ROCS output file into a pandas dataframe.
        """
        if self.filename.endswith('.gz'):
            compression = 'gzip'
        elif self.filename.endswith('.bz2'):
            compression = 'bz2'
        else:
            compression = None
        df = pd.read_table(self.filename, compression=compression)

        # drop empty column from extra tab
        df.dropna(axis=1, how='all', inplace=True)
        return df
