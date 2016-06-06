"""
Graph similarity.
"""
import numpy as np

from openeye.oegraphsim import *


class GraphSimilarity(object):
    """
    Calculate graph similarity.

    Parameters
    ----------
    method : str, optional (default OEFPType_Circular)
        Graph similarity type.
    """
    def __init__(self, fp_type=OEFPType_Circular):
        if isinstance(fp_type, str):
            fp_type = self.resolve_fp_type(fp_type)
        self.fp_type = fp_type
        self.ref_mol = None
        self.ref_fp = None

    def set_ref_mol(self, ref_mol):
        """
        Set reference molecule.

        Parameters
        ----------
        ref_mol : OEMol
            Reference molecule.
        """
        self.ref_mol = ref_mol
        self.ref_fp = self.get_fingerprint(ref_mol)

    def get_tanimoto(self, fit_mol):
        """
        Get tanimoto to reference molecule.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        """
        return OETanimoto(self.ref_fp, self.get_fingerprint(fit_mol))

    def get_fingerprint(self, mol):
        """
        Construct molecule fingerprint.

        Parameters
        ----------
        mol : OEMol
            Molecule.
        """
        mol_fp = OEFingerPrint()
        OEMakeFP(mol_fp, mol, self.fp_type)
        return mol_fp

    @staticmethod
    def get_bits(mol_fp):
        """
        Get a molecule fingerprint as a bit vector.

        Parameters
        ----------
        fp : OEFingerPrint
            Molecule fingerprint.
        """
        bits = np.zeros(mol_fp.GetSize(), dtype=bool)
        for i in xrange(mol_fp.GetSize()):
            bits[i] = mol_fp.IsBitOn(i)
        return bits

    @staticmethod
    def resolve_fp_type(fp_type):
        """
        Resolve fingerprint type.

        Parameters
        ----------
        fp_type : str
            Fingerprint type.
        """
        if fp_type == 'maccs':
            return OEFPType_MACCS166
        elif fp_type == 'circular':
            return OEFPType_Circular
        elif fp_type == 'path':
            return OEFPType_Path
        elif fp_type == 'tree':
            return OEFPType_Tree
        else:
            raise NotImplementedError(fp_type)
