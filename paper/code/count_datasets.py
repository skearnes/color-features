"""Get dataset size.

Note that these values may differ from published accounts due to OMEGA failures.
"""

import cPickle as pickle
import gflags as flags
import gzip
import logging
from oe_utils import chem
import os
import pandas as pd
import sys

flags.DEFINE_string('datasets', None, 'Dataset filename.')
flags.DEFINE_string('actives', None, 'Path to actives.')
flags.DEFINE_string('decoys', None, 'Path to decoys.')
flags.DEFINE_string('output', None, 'Output filename.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)


def count_mols(filename):
    reader = chem.MolReader(filename)
    count = 0
    max_confs = 0
    for mol in reader.get_mols():
        count += 1
        if mol.NumConfs() > max_confs:
            max_confs = mol.NumConfs()
    logging.info('%s:\t%d\t%d', filename, count, max_confs)
    return count


def main():
    with open(FLAGS.datasets) as f:
        datasets = [line.strip() for line in f]
    rows = []
    table = ''
    for dataset in datasets:
        a_filename = os.path.join(FLAGS.actives, '%s-actives.oeb.gz' % dataset)
        d_filename = os.path.join(FLAGS.decoys, '%s-decoys.oeb.gz' % dataset)
        actives = count_mols(a_filename)
        decoys = count_mols(d_filename)

        # Print a nice table.
        table += '%s & %d & %d & %d \\\\\n' % (
            dataset, actives, decoys, actives + decoys)
        rows.append({'dataset': dataset, 'actives': actives, 'decoys': decoys})
    df = pd.DataFrame(rows)
    with gzip.open(FLAGS.output, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    print table

if __name__ == '__main__':
    flags.MarkFlagAsRequired('datasets')
    flags.MarkFlagAsRequired('actives')
    flags.MarkFlagAsRequired('decoys')
    flags.MarkFlagAsRequired('output')
    FLAGS(sys.argv)
    main()


