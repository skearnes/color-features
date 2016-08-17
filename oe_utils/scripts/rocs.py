#!/usr/bin/env python
"""
Get ROCS overlays.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import argparse
import collections
from functools import partial
import numpy as np

from oe_utils.chem import MolReader
from oe_utils.scripts import get_map
from oe_utils.shape import ROCS
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
    parser.add_argument('-o', '--out', required=1,
                        help='Output filename.')
    parser.add_argument('--cluster-id',
                        help='IPython.parallel cluster ID.')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size.')
    return parser.parse_args(input_args)


def main(ref_filename, fit_filename, out_filename, cluster_id=None,
         batch_size=1000):
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
    cluster_id : str, optional
        IPython.parallel cluster ID.
    batch_size : int, optional (default 1000)
        Batch size.
    """
    rocs_kwargs = {}  # TODO: get from command line
    results = collections.defaultdict(list)
    ref_reader = MolReader(ref_filename)
    call = get_map(cluster_id)
    for ref_mol in ref_reader.get_mols():
        fit_reader = MolReader(fit_filename)  # new generator each time
        f = partial(worker, ref_mol=ref_mol, **rocs_kwargs)
        worker_results = call(f, fit_reader.get_batches(batch_size))
        ref_results = collections.defaultdict(list)
        for worker_result in worker_results:
            for fit_result in worker_result:
                ref_results['overlaps'].append(fit_result['overlap'])
                ref_results['fit_titles'].append(fit_result['fit_title'])
        for key, value in ref_results.iteritems():
            results[key].append(value)
        results['ref_titles'].append(ref_mol.GetTitle())
    # transpose to get fit mols on first axis
    results['overlaps'] = np.asarray(results['overlaps']).T
    data = ROCS.group_results(results['overlaps'])
    # check that fit titles are consistent
    for fit_titles in results['fit_titles'][1:]:
        assert np.array_equal(fit_titles, results['fit_titles'][0])
    results['fit_titles'] = results['fit_titles'][0]
    # add titles to output dict
    for key in ['ref_titles', 'fit_titles']:
        data[key] = results[key]
    h5_utils.dump(data, out_filename)


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
        Keyword arguments for ROCS engine.
    """
    from oe_utils.shape import ROCS

    engine = ROCS(**kwargs)
    engine.SetRefMol(ref_mol)
    ref_results = []
    for fit_mol in fit_mols:
        ref_results.append({
            'overlap': engine.get_best_overlay(fit_mol),
            'fit_title': fit_mol.GetTitle()})
    return ref_results

if __name__ == '__main__':
    args = get_args()
    main(args.ref, args.fit, args.out, args.cluster_id, args.batch_size)
