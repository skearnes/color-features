#!/usr/bin/env python
"""
Get separate scores for each reference color atom overlap.
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
from oe_utils.shape.overlap import ColorOverlap
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
    Get separate scores for each reference color atom overlap.

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
    color_overlap_kwargs, rocs_kwargs = {}, {}  # TODO: get from command line
    results = collections.defaultdict(list)
    ref_reader = MolReader(ref_filename)
    call = get_map(cluster_id)
    ref_size, fit_size = 0, 0
    for ref_mol in ref_reader.get_mols():
        fit_reader = MolReader(fit_filename)  # new generator each time
        f = partial(worker, ref_mol=ref_mol, rocs_kwargs=rocs_kwargs,
                    color_overlap_kwargs=color_overlap_kwargs)
        worker_results = call(f, fit_reader.get_batches(batch_size))
        # store each result as a separate array
        ref_results = collections.defaultdict(list)
        for worker_result in worker_results:
            for fit_result in worker_result:
                ref_results['overlaps'].append(fit_result['overlaps'])
                ref_results['fit_titles'].append(fit_result['fit_title'])
                check_color_consistency(ref_results, fit_result)
        for key, value in ref_results.iteritems():
            results[key].append(value)
        results['ref_titles'].append(ref_mol.GetTitle())
        ref_size += 1
        fit_size = len(ref_results['overlaps'])
    # ensure that results end up 2D (not 3D)
    # Since each reference molecule can have a different number of color atoms,
    # we store the arrays as objects and use a mask in the grouping function.
    cao_data = np.zeros((ref_size, fit_size), dtype=object)
    for i, ref_results in enumerate(results['overlaps']):
        for j, fit_results in enumerate(ref_results):
            cao_data[i, j] = fit_results
    # transpose to get fit mols on first axis
    cao_data = np.asarray(cao_data).transpose((1, 0))
    cao_data = ColorOverlap.group_ref_color_atom_overlaps(cao_data)
    grouped_data = process_metadata(results, ref_size, cao_data.shape[-1])
    # check that fit titles are consistent
    for fit_titles in results['fit_titles'][1:]:
        assert np.array_equal(fit_titles, results['fit_titles'][0])
    results['fit_titles'] = results['fit_titles'][0]
    # construct output dict
    h5_utils.dump(
        {'color_atom_overlaps': cao_data.filled(np.nan),
         'mask': cao_data.mask,
         'ref_color_coords': grouped_data['ref_color_coords'].filled(np.nan),
         'ref_color_types': grouped_data['ref_color_types'].filled(np.nan),
         'ref_color_type_names': grouped_data[
             'ref_color_type_names'].filled(''),
         'ref_color_coords_mask': grouped_data['ref_color_coords'].mask,
         'ref_color_types_mask': grouped_data['ref_color_types'].mask,
         'ref_color_type_names_mask': grouped_data['ref_color_type_names'].mask,
         'ref_titles': results['ref_titles'],
         'fit_titles': results['fit_titles']},
        out_filename)


def worker(fit_mols, ref_mol, rocs_kwargs=None, color_overlap_kwargs=None):
    """
    Worker.

    Parameters
    ----------
    fit_mols : iterable
            Fit molecules.
    ref_mol : OEMol
        Reference molecule.
    rocs_kwargs : dict, optional
        Keyword arguments for ROCS engine.
    color_overlap_kwargs : dict, optional
        Keyword arguments for ColorOverlap engine.

    Returns
    -------
    List of dicts (one for each fit_mol) containing overlap results.
    """
    import numpy as np

    from oe_utils.shape import ROCS
    from oe_utils.shape.overlap import ColorOverlap

    if rocs_kwargs is None:
        rocs_kwargs = {}
    if color_overlap_kwargs is None:
        color_overlap_kwargs = {}
    rocs_engine = ROCS(**rocs_kwargs)
    overlap_engine = ColorOverlap(**color_overlap_kwargs)
    rocs_engine.SetRefMol(ref_mol)
    ref_results = []
    for fit_mol in fit_mols:
        rocs_result = rocs_engine.get_best_overlay(fit_mol)
        ref_conf, fit_conf = rocs_engine.get_aligned_confs(
            ref_mol, fit_mol, rocs_result)
        overlap_engine.SetRefMol(ref_conf)
        fit_results = overlap_engine.get_ref_color_atom_overlaps(fit_conf)
        # Extract color overlap scores from ColorOverlapResults.
        fit_results['overlaps'] = np.asarray(
            [result.color_overlap for result in fit_results['overlaps']],
            dtype=float)
        fit_results['fit_title'] = fit_mol.GetTitle()
        ref_results.append(fit_results)
    return ref_results


def check_color_consistency(primary, secondary):
    """Check that reference molecule color atoms are consistent."""
    keys = ['ref_color_coords', 'ref_color_types', 'ref_color_type_names']
    for key in keys:
        if key not in primary:
            primary[key] = secondary[key]
        else:
            assert np.array_equal(primary[key], secondary[key])


def process_metadata(results, ref_size, max_size):
    """Process the reference molecule color atom metadata."""
    # Maximum length of color type names.
    max_length = max([max([len(type_name)
                           for type_name in type_names])
                      for type_names in results['ref_color_type_names']])
    grouped_data = {}
    for key in ['ref_color_coords', 'ref_color_types', 'ref_color_type_names']:
        shape = (ref_size, max_size)
        if key == 'ref_color_coords':
            shape += (3,)
        if key == 'ref_color_type_names':
            dtype = 'S' + str(max_length)
        else:
            dtype = float
        data = np.ma.masked_all(shape, dtype=dtype)
        for i, ref_values in enumerate(results[key]):
            data[i, :len(ref_values)] = ref_values
        grouped_data[key] = data
    return grouped_data

if __name__ == '__main__':
    args = get_args()
    main(args.ref, args.fit, args.out, args.cluster_id, args.batch_size)
