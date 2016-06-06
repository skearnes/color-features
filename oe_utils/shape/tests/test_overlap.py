"""
Tests for OEShape overlap utilities.
"""
import numpy as np

from openeye.oechem import *
from openeye.oeshape import *

from . import TestShapeBase
from .. import ROCS
from ..color import ColorForceField
from ..overlap import ColorOverlap


class TestColorOverlap(TestShapeBase):
    """
    Tests for ColorOverlap and ColorOverlapResult.
    """
    def setUp(self):
        """
        Set up tests.
        """
        super(TestColorOverlap, self).setUp()

        # align with ROCS
        rocs = ROCS()
        rocs.SetRefMol(self.mols[0])
        self.result = rocs.get_best_overlay(self.mols[1])
        self.ref_conf, self.fit_conf = rocs.get_aligned_confs(
            self.mols[0], self.mols[1], self.result)

        self.overlap = ColorOverlap()
        self.overlap.SetRefMol(self.ref_conf)

        # get expected color atoms
        self.colored_ref_conf = OEMol(self.ref_conf)
        self.colored_fit_conf = OEMol(self.fit_conf)
        OEAddColorAtoms(self.colored_ref_conf, self.overlap.color_ff)
        OEAddColorAtoms(self.colored_fit_conf, self.overlap.color_ff)
        self.common_color_types = np.intersect1d(
            [OEGetColorType(color_atom)
             for color_atom in OEGetColorAtoms(self.colored_ref_conf)],
            [OEGetColorType(color_atom)
             for color_atom in OEGetColorAtoms(self.colored_fit_conf)])
        assert np.array_equal(self.common_color_types, [2, 4, 5])

    def test_overlap(self):
        """
        Test ColorOverlap.overlap.
        """
        result = self.overlap.overlap(self.fit_conf)
        assert result.color_tanimoto > 0
        assert np.allclose(result.color_tanimoto, self.result.color_tanimoto,
                           atol=0.01)

    def get_tanimoto(self, results):
        """
        Calculate Tanimoto from component or overlap results.
        """
        ref_self_color, fit_self_color, color_overlap = 0, 0, 0
        for result in results:
            ref_self_color += result.ref_self_color
            fit_self_color += result.fit_self_color
            color_overlap += result.color_overlap
        return np.true_divide(
            color_overlap, ref_self_color + fit_self_color - color_overlap)

    def test_get_color_components(self):
        """
        Test ColorOverlap.get_color_components.
        """
        results = self.overlap.get_color_components(self.fit_conf)

        # test that all interactions are accounted for
        assert len(results) == len(
            ColorForceField(self.overlap.color_ff).get_interactions())

        # look at individual results
        for result, engine in zip(results,
                                  self.overlap.color_component_engines):

            # verify this CFF has just one interaction
            interactions = engine.color_ff.get_interactions()
            assert len(interactions) == 1

            # look for specific interactions
            if interactions[0][0] in self.common_color_types:
                assert result.color_tanimoto > 0
            else:
                assert result.color_tanimoto == 0

        # check that original color scores can be recovered from components
        ref_self_color, fit_self_color, color_overlap = 0, 0, 0
        for result in results:
            ref_self_color += result.ref_self_color
            fit_self_color += result.fit_self_color
            color_overlap += result.color_overlap
        assert np.allclose(ref_self_color, self.result.ref_self_color)
        assert np.allclose(fit_self_color, self.result.fit_self_color)
        assert np.allclose(color_overlap, self.result.color_overlap)

    def test_get_ref_color_atom_overlaps(self):
        """
        Test ColorOverlap.get_ref_color_atom_overlaps.
        """
        results = self.overlap.get_ref_color_atom_overlaps(self.fit_conf)

        # test that all ref color atoms are accounted for
        assert len(results) == 5
        unmatched = 0
        for result, ref_color_atom in zip(
                results, OEGetColorAtoms(self.colored_ref_conf)):
            if OEGetColorType(ref_color_atom) in self.common_color_types:
                try:
                    assert result.color_tanimoto > 0
                except AssertionError:  # one ref acceptor is not matched
                    acceptor_type = self.overlap.color_ff.GetType('acceptor')
                    assert OEGetColorType(ref_color_atom) == acceptor_type
                    assert result.color_tanimoto == 0
                    unmatched += 1
            else:
                assert result.color_tanimoto == 0
        assert unmatched == 1

        # check that original color scores can be recovered from overlaps
        # FIXME: component ref_self_color is wrong for some reason
        # component fit_self_color should be constant for all results
        color_overlap = 0
        for result in results:
            color_overlap += result.color_overlap
            # fit self color should be constant
            assert result.fit_self_color == self.result.fit_self_color
        # assert np.allclose(ref_self_color, self.result.ref_self_color)
        assert np.allclose(color_overlap, self.result.color_overlap)
