"""
Tests for OEShape color utilities.
"""
import numpy as np
import unittest

from openeye.oechem import *
from openeye.oeshape import *

from ..color import ColorForceField


class TestColorForceField(unittest.TestCase):
    """
    Tests for ColorForceField.
    """
    def setUp(self):
        """
        Set up tests.
        """
        self.color_ff = ColorForceField()
        self.color_ff.Init(OEColorFFType_ImplicitMillsDean)

    def test_get_interactions(self):
        """
        Test ColorForceField.get_interactions.
        """
        interactions = self.color_ff.get_interactions()
        assert len(interactions) == 6
        for (a_type, b_type, decay, weight, radius) in interactions:
            assert a_type == b_type
            assert decay == 'gaussian'
            assert weight < 0
            assert radius > 0

    def test_get_string(self):
        """
        Test ColorForceField.get_string.
        """
        ifs = oeisstream(self.color_ff.get_string())
        color_ff = ColorForceField()
        color_ff.Init(ifs)
        for a_interaction, b_interaction in zip(
                color_ff.get_interactions(), self.color_ff.get_interactions()):
            assert np.array_equal(a_interaction, b_interaction)

    def test_isolate_interactions(self):
        """
        Test ColorForceField.isolate_interactions.
        """
        interactions = set()
        for color_ff in self.color_ff.isolate_interactions():
            assert len(color_ff.get_interactions()) == 1
            for interaction in color_ff.get_interactions():
                interactions.add(interaction)
        assert interactions == set(self.color_ff.get_interactions())
