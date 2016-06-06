#!/usr/bin/env python
"""
Calculate graph similarity.
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
    parser.add_argument('-r', '--ref', required=1,
                        help='Reference molecules.')
    parser.add_argument('-f', '--fit', required=1,
                        help='Fit molecules.')
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


def main(ref_filename, fit_filename, out_filename, fp_type='circular',
         cluster_id=None, batch_size=1000):
    """
    Get ROCS overlays.

    Parameters
    ----------
    ref_filename : str
        Reference molecule filename.
    fit_filename : str
        Fit molecule filename.
    out_filename : str
        Output filename.
    fp_type : str, optional (default 'circular')
        Fingerprint type.
    cluster_id : str, optional
        IPython.parallel cluster ID.
    batch_size : int, optional (default 1000)
        Batch size.
    """
    results = []
    ref_reader = MolReader(ref_filename)
    call = get_map(cluster_id)
    for ref_mol in ref_reader.get_mols():
        fit_reader = MolReader(fit_filename)  # new generator each time
        f = partial(worker, ref_mol=ref_mol, fp_type=fp_type)
        ref_results = call(f, fit_reader.get_batches(batch_size))
        results.append(np.concatenate(ref_results))
    # transpose to get fit mols on first axis
    results = np.asarray(results, dtype=float).T
    h5_utils.dump({'tanimotos': results}, out_filename)


def worker(fit_mols, ref_mol, **kwargs):
    """
    Worker.

    Parameters
    ----------
    fit_mols : iterable
        Fit molecules.
    ref_mol : OEMol
        Reference molecule.
    kwargs : dict, optional
        Keyword arguments for GraphSimilarity engine.
    """
    from oe_utils.graphsim import GraphSimilarity

    engine = GraphSimilarity(**kwargs)
    engine.set_ref_mol(ref_mol)
    return map(engine.get_tanimoto, fit_mols)

if __name__ == '__main__':
    args = get_args()
    main(args.ref, args.fit, args.output, args.type, args.cluster_id,
         args.batch_size)
