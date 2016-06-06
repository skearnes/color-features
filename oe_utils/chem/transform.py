"""
OEChem transform utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from openeye.oechem import *


class MolTransform(object):
    """
    Transform molecules.

    Parameters
    ----------
    rotation : array_like, optional
        Rotation matrix.
    translation : array_like, optional
        Translation.
    """
    def __init__(self, rotation=None, translation=None):
        assert rotation is not None or translation is not None
        self.rotation = rotation
        self.translation = translation

    def transform(self, mol, copy=True):
        """
        Transform a molecule.

        Parameters
        ----------
        mol : OEMol
            Molecule to transform.
        copy : bool, optional (default True)
            Copy molecule before applying transform.
        """
        if copy:
            mol = OEMol(mol)
        if self.rotation is not None:
            rotation = OERotMatrix(OEFloatArray(np.ravel(self.rotation)))
            rotation.Transform(mol)
        if self.translation is not None:
            translation = OETranslation(OEFloatArray(self.translation))
            translation.Transform(mol)
        return mol
