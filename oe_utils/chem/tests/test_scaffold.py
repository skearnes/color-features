"""
Test scaffold utilities.
"""
import unittest

from oe_utils.chem.scaffold import Scaffold
from openeye.oechem import *


class TestScaffold(unittest.TestCase):
    """
    Test Scaffold.
    """
    def setUp(self):
        """
        Set up tests.
        """
        smiles = 'C[C@@]1(C(=O)NC(=O)N(C1=O)C)C2=CCCCC2 (S)-Hexobarbital'
        self.mol = OEMol()
        OESmilesToMol(self.mol, smiles)
        self.engine = Scaffold()

    def test_murcko_scaffold(self):
        """
        Test Murcko scaffold.
        """
        smiles = self.engine.get_scaffold_smiles(self.mol)
        scaffold = OEMol()
        OESmilesToMol(scaffold, 'C1(CNCNC1)C2=CCCCC2')
        assert smiles == OECreateIsoSmiString(scaffold)
