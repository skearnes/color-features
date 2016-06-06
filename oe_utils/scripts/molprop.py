"""
Calculate simple descriptors.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import argparse

from openeye.oechem import *
from openeye.oemolprop import *

from oe_utils.chem import MolReader


def get_args():
    """
    Get command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=1)
    parser.add_argument('-o', '--output', required=1)
    return parser.parse_args()


def main(args):
    """
    Calculate simple descriptors.
    """
    OEThrow.SetLevel(OEErrorLevel_Warning)
    engine = OEFilter()
    ofs = oeofstream()
    if not ofs.open(args.output):
        raise IOError("Cannot open output file '{}'.".format(args.output))
    engine.SetTable(ofs, False)
    for mol in MolReader(args.input).get_mols():
        engine(mol)

if __name__ == '__main__':
    main(get_args())
