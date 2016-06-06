"""
OEShape color utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from openeye.oechem import *
from openeye.oeshape import *


class ColorForceField(OEColorForceField):
    """
    Color force field wrapper with additional utility methods.
    """
    def get_interactions(self):
        """
        Extract color-color interactions from INTERACTION lines.

        Returns
        -------
        interactions : array_like
            Array containing tuples of (a_type, b_type, decay, weight, radius).

        Notes
        -----
        * Negative weights indicate attractive interactions.
        * Interaction list is sorted.
        """
        interactions = []
        for line in self.get_string().split('\n'):
            fields = line.split()
            if not len(fields) or str.lower(fields[0]) != 'interaction':
                continue
            assert len(fields) == 7

            # get color atom types
            a_type = self.GetType(fields[1])
            b_type = self.GetType(fields[2])
            assert a_type and b_type

            # get decay
            decay = fields[4]

            # get weight and direction (negative weights are attractive)
            weight_name, weight = fields[5].split('=')
            assert str.lower(weight_name) == 'weight'
            weight = float(weight)
            if str.lower(fields[3]) == 'attractive':
                weight *= -1
            else:
                assert str.lower(fields[3]) == 'repulsive'

            # get radius
            radius_name, radius = fields[6].split('=')
            assert str.lower(radius_name) == 'radius'
            radius = float(radius)

            # construct an interaction tuple matching the order required by
            # OEColorForceField.AddInteraction
            interactions.append((a_type, b_type, decay, weight, radius))
        return sorted(interactions)  # sort interaction list

    def get_string(self):
        """
        Return the color force field as a string.
        """
        ostream = oeosstream()
        self.Write(ostream)
        return ostream.str()

    def isolate_interactions(self):
        """
        Yield color force fields that each have only a single interaction.
        """
        for interaction in self.get_interactions():
            color_ff = ColorForceField(self)  # create a copy
            color_ff.ClearInteractions()
            color_ff.AddInteraction(*interaction)
            yield color_ff
