"""Print a dataset table."""

import cPickle as pickle
import gflags as flags
import gzip
import sys

flags.DEFINE_string('input', None, 'Input dataset pickle.')
FLAGS = flags.FLAGS


def main():
    with gzip.open(FLAGS.input) as f:
        df = pickle.load(f)
    table = ''
    for _, row in df.iterrows():
        actives = row['actives']
        decoys = row['decoys']
        f_active = 100 * float(actives) / (actives + decoys)
        table += '%s & %d & %d & %.0f \\\\\n' % (
            row['dataset'], actives, decoys, f_active)
    print table

if __name__ == '__main__':
    flags.MarkFlagAsRequired('input')
    FLAGS(sys.argv)
    main()
