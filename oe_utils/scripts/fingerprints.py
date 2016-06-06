#!/usr/bin/env python
"""
Calculate molecule fingerprints.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import argparse
from functools import partial
import numpy as np

from oe_utils.chem import MolReader
from oe_utils.scripts import get_map
from oe_utils.utils import h5_utils


def get_args(input_args=None):
    """
    Get command line arguments.

    Parameters
    ----------
    input_args : list, optional
        Input arguments. If None, defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=1,
                        help='Input molecules.')
    parser.add_argument('-o', '--output',
                        help='Output filename.')
    parser.add_argument('--cluster-id',
                        help='IPython.parallel cluster ID.')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size.')
    parser.add_argument('-t', '--type', default='circular',
                        choices=['maccs', 'circular', 'path', 'tree'],
                        help='Fingerprint type.')
    return parser.parse_args(input_args)


def main(input_filename, out_filename, fp_type='circular', cluster_id=None,
         batch_size=1000):
    """
    Get ROCS overlays.

    Parameters
    ----------
    input_filename : str
        Input molecule filename.
    out_filename : str
        Output filename.
    fp_type : str, optional (default 'circular')
        Fingerprint type.
    cluster_id : str, optional
        IPython.parallel cluster ID.
    batch_size : int, optional (default 1000)
        Batch size.
    """
    reader = MolReader(input_filename)
    call = get_map(cluster_id)
    f = partial(worker, fp_type=fp_type)
    results = call(f, reader.get_batches(batch_size))
    h5_utils.dump({'fingerprints': np.vstack(results)}, out_filename,
                  attrs={'type': args.type})


def worker(mols, **kwargs):
    """
    Worker.

    Parameters
    ----------
    mols : iterable
        Molecules.
    kwargs : dict, optional
        Keyword arguments for GraphSimilarity engine.
    """
    from oe_utils.graphsim import GraphSimilarity

    engine = GraphSimilarity(**kwargs)
    return [engine.get_bits(engine.get_fingerprint(mol)) for mol in mols]

if __name__ == '__main__':
    args = get_args()
    main(args.input, args.output, args.type, args.cluster_id, args.batch_size)
