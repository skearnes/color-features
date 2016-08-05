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
        table += '%s & %d & %d & %d \\\\\n' % (
            row['dataset'], row['actives'], row['decoys'],
            row['actives'] + row['decoys'])
    print table

if __name__ == '__main__':
    flags.MarkFlagAsRequired('input')
    FLAGS(sys.argv)
    main()
