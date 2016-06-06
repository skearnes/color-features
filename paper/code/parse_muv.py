"""Parse MUV datasets.

MUV datasets are provided as supporting data to
http://jcheminf.springeropen.com/articles/10.1186/1758-2946-5-26

This script can also be used for the other datasets in the supporting data.
"""

import gflags as flags
import glob
import gzip
import logging
import os
import re
import sys

flags.DEFINE_string('root', None, 'Root directory containing datasets.')
flags.DEFINE_string('prefix', 'aid', 'Prefix to append to output filenames.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)


def extract_smiles(suffix):
    for filename in glob.glob(os.path.join(FLAGS.root, '*_%s.dat.gz' % suffix)):
        match = re.search('cmp_list_.*?_(.*?)_%s' % suffix, filename)
        name = match.group(1)
        logging.info('%s -> %s', filename, name)
        output_filename = '%s%s-%s.smi' % (FLAGS.prefix, name, suffix)
        with open(output_filename, 'wb') as outfile:
            count = 0
            with gzip.open(filename) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    source_id, data_id, smi = line.split()
                    outfile.write('%s\t%s\n' % (smi, data_id))
                    count += 1
            logging.info('%s: %d', output_filename, count)


def main():
    extract_smiles('actives')
    extract_smiles('decoys')


if __name__ == '__main__':
    flags.MarkFlagAsRequired('root')
    FLAGS(sys.argv)
    main()
