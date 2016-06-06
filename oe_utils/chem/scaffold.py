"""
Scaffold utilities.

Based on example code available at
http://docs.eyesopen.com/toolkits/oechem/python/examplesoechem.html
#extract-molecule-scaffolds, which is Copyright (C) 2009-2014 OpenEye
Scientific Software, Inc.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from openeye.oechem import *


class Scaffold(object):
    """
    Extract the scaffold of a molecule. The default parameters give the
    Murcko scaffold as defined in J. Med. Chem. 1996, 39, 2887-289.

    To be part of the scaffold, an atom must be in a ring or in a linker
    between two rings.

    Parameters
    ----------
    stereo : bool, optional (default False)
        Whether to retain stereochemistry in the scaffold.
    """
    def __init__(self, stereo=False):
        self.stereo = stereo

    def __call__(self, mol):
        """
        Extract the scaffold of a molecule.

        Parameters
        ----------
        mol : OEMol
            Molecule.
        """
        return self.get_scaffold(mol)

    def get_scaffold(self, mol):
        """
        Extract the scaffold of a molecule.

        Parameters
        ----------
        mol : OEMol
            Molecule.
        """
        mol = mol.CreateCopy()
        pred = IsInScaffold()
        OEFindRingAtomsAndBonds(mol)  # prepare molecule
        scaffold = OEMol()
        adjust_h_count = True
        OESubsetMol(scaffold, mol, pred, adjust_h_count)

        # acyclic molecules: return the original molecule
        if not scaffold.IsValid():
            scaffold = mol

        # handle stereochemistry
        if not self.stereo:
            flat = OECreateCanSmiString(scaffold)
            scaffold = OEMol()
            OESmilesToMol(scaffold, flat)
        else:
            OEPerceiveChiral(scaffold)
        return scaffold

    def get_scaffold_smiles(self, mol):
        """
        Extract the scaffold of a molecule as a SMILES string.

        Parameters
        ----------
        mol : OEMol
            Molecule.
        """
        return OECreateIsoSmiString(self.get_scaffold(mol))


class IsInScaffold(OEUnaryAtomPred):
    """
    Test if an atom is part of the Murcko scaffold of a molecule.
    """
    def __call__(self, atom):
        """
        Check whether an atom is in the Murcko scaffold.

        Parameters
        ----------
        atom : OEAtom
            Atom.
        """
        if atom.IsInRing():
            return True
        count = self.depth_first_search_for_ring(atom)
        return count > 1

    def depth_first_search_for_ring(self, atom):
        """
        Perform a depth-first search to count the number of rings that are
        directly or indirectly connected to an atom.

        Parameters
        ----------
        atom : OEAtom
            Atom.
        """
        count = 0
        for nbor in atom.GetAtoms():
            visited = set()
            visited.add(atom)
            if nbor.IsInRing() or self.traverse_for_ring(visited, nbor):
                count += 1
        return count

    def traverse_for_ring(self, visited, atom):
        """
        Test if this atom is directly or indirectly connected to a ring.

        Parameters
        ----------
        visited : set
            Set of previously visited atoms.
        atom : OEAtom
            Atom.
        """
        visited.add(atom)
        for nbor in atom.GetAtoms():
            if nbor not in visited:
                if nbor.IsInRing():
                    return True
                if self.traverse_for_ring(visited, nbor):
                    return True
        return False
