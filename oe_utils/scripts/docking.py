#!/usr/bin/env python
"""
Docking.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import argparse
import cPickle
import gzip
import h5py
from joblib import delayed, Parallel
import numpy as np
import os

from openeye.oechem import *

from oe_utils.chem import MolReader
from oe_utils.docking import read_receptor, Docker, HybridDocker


def get_args(input_args=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    input_args : list, optional
        Input arguments. If not provided, defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--receptor', required=1)
    parser.add_argument('-d', '--dbase', required=1)
    parser.add_argument('-p', '--prefix', required=1)
    parser.add_argument('-np', '--n_jobs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(input_args)


def main(receptor_filename, dbase_filename, prefix, n_jobs=1, batch_size=1000,
         hybrid=False, verbose=False, **kwargs):
    """
    Dock molecules to a receptor.

    Parameters
    ----------
    receptor_filename : str
        Receptor filename.
    dbase_filename : str
        Molecule database filename.
    prefix : str
        Prefix for output files.
    n_jobs : int, optional (default 1)
        Number of parallel jobs.
    batch_size : int, optional (default 1000)
        Parallel batch size.
    hybrid : bool, optional (default False)
        Whether to perform hybrid docking.
    verbose : bool, optional (default False)
        Whether to provide verbose output.
    kwargs : dict, optional
        Keyword arguments for docking engine.
    """
    receptor = read_receptor(receptor_filename)

    # check for conflicting output files
    if os.path.exists(prefix + '-docked.oeb.gz'):
        os.remove(prefix + '-docked.oeb.gz')
    if os.path.exists(prefix + '-scores.h5'):
        os.remove(prefix + '-scores.h5')
    if os.path.exists(prefix + '-failed.oeb.gz'):
        os.remove(prefix + '-failed.oeb.gz')

    # perform docking in batches
    reader = MolReader()
    reader.open(dbase_filename)
    for i, batch in enumerate(reader.get_batches(batch_size)):
        if verbose:
            print 'Docking batch {}...'.format(i)

        # dock this batch
        results = Parallel(n_jobs=n_jobs, verbose=5 * verbose)(
            delayed(dock)(receptor, mols, hybrid, **kwargs)
            for mols in np.array_split(batch, n_jobs))

        # save results for this batch
        poses = []
        scores = []
        failed = []
        for this_poses, this_scores in results:
            for pose, score in zip(this_poses, this_scores):
                if score is None:  # failed molecules
                    failed.append(pose)
                else:
                    poses.append(pose)
                    scores.append(score)
        poses = [cPickle.loads(state) for state in poses]
        scores = np.asarray(scores, dtype=float)
        failed = [cPickle.loads(state) for state in failed]
        assert len(poses) == len(scores)
        assert len(poses) + len(failed) == len(batch)
        write_results(poses, scores, failed, prefix)
    reader.close()


def dock(receptor, mols, hybrid, **kwargs):
    """
    Dock molecules to a receptor.

    Parameters
    ----------
    receptor : OEMol
        Receptor.
    mols : array_like
        Molecules to dock to receptor.
    hybrid : bool
        Whether to perform hybrid docking.
    kwargs : dict, optional
        Keyword arguments for docking engine.
    """
    if hybrid:
        engine = HybridDocker(receptor, **kwargs)
    else:
        engine = Docker(receptor, **kwargs)
    poses = []
    scores = []
    for mol in mols:
        this_poses, this_scores = engine.dock(mol)
        poses.append(cPickle.dumps(this_poses))  # pickle manually
        scores.append(this_scores)
    return poses, scores


def write_results(poses, scores, failed, prefix):
    """
    Write docked poses and scores to files.

    Parameters
    ----------
    poses : array_like
        Docked poses.
    scores : array_like
        Docking scores.
    failed : array_like
        Failed molecules.
    prefix : str
        Output file prefix.
    """

    # poses
    append_mols(poses, prefix + '-docked.oeb.gz')

    # scores
    names = [mol.GetTitle() for mol in poses]
    options = {'chunks': True, 'fletcher32': True, 'shuffle': True,
               'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(prefix + '-scores.h5', 'a') as f:
        if 'scores' in f:
            scores = np.vstack((f['scores'][:], scores))
            names = np.concatenate((f['names'][:], names))
            del f['scores'], f['names']
        f.create_dataset('scores', data=scores, **options)
        f.create_dataset('names', data=names, **options)

    # failed
    if len(failed):
        append_mols(failed, prefix + '-failed.oeb.gz')


def append_mols(mols, filename):
    """
    Append molecules to an oeb(.gz) file.

    Parameters
    ----------
    mols : iterable
        Molecules.
    filename : str
        Filename.
    """
    ofs = oemolostream()
    ofs.SetFormat(OEFormat_OEB)
    ofs.openstring()
    for mol in mols:
        OEWriteMolecule(ofs, mol)
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'ab')
    else:
        f = open(filename, 'ab')
    f.write(ofs.GetString())
    f.close()
    ofs.close()

if __name__ == '__main__':
    args = get_args()
    main(args.receptor, args.dbase, args.prefix, args.n_jobs, args.batch_size,
         args.hybrid, args.verbose)
