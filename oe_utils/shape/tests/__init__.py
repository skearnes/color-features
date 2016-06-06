"""
Tests for OEShape utilities.
"""
import numpy as np
import unittest

from openeye.oechem import *
from openeye.oeomega import *
from openeye.oeshape import *

from .. import ROCS


class TestShapeBase(unittest.TestCase):
    """
    Base class for shape tests.
    """
    def setUp(self):
        """
        Set up tests.
        """
        ifs = oemolistream()
        ifs.openstring('CC(=O)OC1=CC=CC=C1C(=O)O aspirin\n' +
                       'C[C@@H](C1=CC=C(C=C1)CC(C)C)C(=O)O dexibuprofen\n')
        aspirin = OEMol()
        OEReadMolecule(ifs, aspirin)
        ibuprofen = OEMol()
        OEReadMolecule(ifs, ibuprofen)
        self.mols = [aspirin, ibuprofen]

        # generate 3D coordinates
        omega = OEOmega()
        omega.SetMaxConfs(3)
        for mol in self.mols:
            omega(mol)


class TestROCS(TestShapeBase):
    """
    Test ROCS and ROCSResult.
    """
    def setUp(self):
        """
        Set up tests.
        """
        super(TestROCS, self).setUp()
        self.rocs = ROCS()

    def test_overlay(self):
        """
        Test ROCS.overlay.
        """
        self.rocs.SetRefMol(self.mols[0])
        results = list(self.rocs.overlay(self.mols[1]))
        assert len(results) == np.product([mol.NumConfs()
                                           for mol in self.mols])

    def test_get_best_overlay(self):
        """
        Test ROCS.get_best_overlay.
        """
        self.rocs.SetRefMol(self.mols[0])

        # self overlay
        result = self.rocs.get_best_overlay(self.mols[0])
        assert np.allclose(result.shape_tanimoto, 1, atol=0.01)
        assert np.allclose(result.color_tanimoto, 1, atol=0.01)

        # other overlay
        result = self.rocs.get_best_overlay(self.mols[1])
        assert not np.allclose(result.shape_tanimoto, 1, atol=0.01)
        assert not np.allclose(result.color_tanimoto, 1, atol=0.01)

    def test_get_aligned_confs(self):
        """
        Test ROCS.get_aligned_confs.
        """
        # get best alignment
        self.rocs.SetRefMol(self.mols[0])
        result = self.rocs.get_best_overlay(self.mols[1])
        assert result.shape_tanimoto > 0
        ref_conf, fit_conf = self.rocs.get_aligned_confs(
            self.mols[0], self.mols[1], result)

        # check shape overlap
        engine = OEOverlap()
        engine.SetUseHydrogens(self.rocs.GetUseHydrogens())
        engine.SetMethod(self.rocs.GetMethod())
        engine.SetRadiiApproximation(self.rocs.GetRadiiApproximation())
        engine.SetRefMol(ref_conf)
        aligned_overlap = OEOverlapResults()
        engine.Overlap(fit_conf, aligned_overlap)
        assert np.allclose(
            aligned_overlap.tanimoto, result.shape_tanimoto, atol=0.01)

        # check that original hasn't been modified
        original_overlap = OEOverlapResults()
        engine.Overlap(self.mols[1].GetConf(OEHasConfIdx(result.fit_conf_idx)),
                       original_overlap)
        assert not np.allclose(
            original_overlap.tanimoto, result.shape_tanimoto, atol=0.1)

        # check that original is modified with copy=False
        self.rocs.get_aligned_confs(
            self.mols[0], self.mols[1], result, copy=False)
        modified_overlap = OEOverlapResults()
        engine.Overlap(self.mols[1].GetConf(OEHasConfIdx(result.fit_conf_idx)),
                       modified_overlap)
        assert np.allclose(
            modified_overlap.tanimoto, result.shape_tanimoto, atol=0.01)
