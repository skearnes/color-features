"""Visualize learned weights for color atoms.

(1) Load molecule and assign color atoms.
(2) Load trained model and assign learned weights to color atoms.
(3) Save molecule for visualization with VIDA.

The color atom overlap features are loaded as a sanity check for color atom
index order.
"""

import cPickle as pickle
import gflags as flags
import gzip
import logging
import numpy as np
from oe_utils import chem
from oe_utils.utils import h5_utils
from openeye.oeshape import *
import sys

flags.DEFINE_string('model', None, 'Trained model.')
flags.DEFINE_string('query', None, 'Query molecule.')
flags.DEFINE_string('prefix', None, 'Prefix for output files.')
flags.DEFINE_string('color_atom_overlaps', None, 'Color atom overlap features.')
flags.DEFINE_integer('ref_index', 0,
                     'Index for query in color atom overlap features.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)


def check_features(mol, color_ff, weights):
    """Check that the query matches the features."""
    features = h5_utils.load(FLAGS.color_atom_overlaps)
    # Compare molecule titles.
    assert mol.GetTitle() == features['ref_titles'][FLAGS.ref_index]
    # Look at feature mask.
    assert np.array_equal(features['mask'][0][FLAGS.ref_index],
                          np.zeros(OECountColorAtoms(mol), dtype=bool))
    # Look at color atom descriptions.
    keys = ['ref_color_coords', 'ref_color_types', 'ref_color_type_names']
    masked = {}
    for key in keys:
        values = features[key][FLAGS.ref_index]
        mask = features[key + '_mask'][FLAGS.ref_index]
        masked[key] = np.ma.masked_array(values, mask=mask)
    for i, color_atom in enumerate(OEGetColorAtoms(mol)):
        # Color atom coordinates.
        coords = mol.GetCoords(color_atom)
        assert np.array_equal(masked['ref_color_coords'][i], coords)
        # Color atom type.
        color_type = OEGetColorType(color_atom)
        color_type_name = color_ff.GetTypeName(color_type)
        assert masked['ref_color_types'][i] == color_type
        assert masked['ref_color_type_names'][i] == color_type_name
        # Assign weight to this atom.
        color_atom.SetFloatData('feature_weight', weights[i])
        logging.info('%d\t%s\t%d\t%s\t%g', i, str(coords), color_type,
                     color_type_name, weights[i])


def main():
    # Load query molecule.
    reader = chem.MolReader(FLAGS.query)
    mols = list(reader.get_mols())
    assert len(mols) == 1, 'Only one query molecule supported.'
    mol = mols[0]

    # Assign color atoms.
    # NOTE: only supports default ImplicitMillsDean color force field.
    color_ff = OEColorForceField()
    color_ff.Init(OEColorFFType_ImplicitMillsDean)
    OEAddColorAtoms(mol, color_ff)

    # Load trained model.
    # Assign feature weights to color atoms.
    with gzip.open(FLAGS.model) as f:
        model = pickle.load(f)
    assert hasattr(model, 'coef_')
    assert model.coef_.shape[0] == 1
    weights = model.coef_.squeeze()
    # First feature is shape similarity.
    logging.info('Shape weight: %g', weights[0])
    weights = weights[1:]
    assert len(weights) == OECountColorAtoms(mol)
    check_features(mol, color_ff, weights)

    # Save annotated molecule for visualization.
    writer = chem.MolWriter('%s-annotated.oeb.gz' % FLAGS.prefix)
    writer.write([mol])
    writer.close()


if __name__ == '__main__':
    flags.MarkFlagAsRequired('model')
    flags.MarkFlagAsRequired('query')
    flags.MarkFlagAsRequired('prefix')
    flags.MarkFlagAsRequired('color_atom_overlaps')
    FLAGS(sys.argv)
    main()
