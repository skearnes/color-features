"""
OEShape overlap utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from openeye.oechem import *
from openeye.oeshape import *

from oe_utils.shape.color import ColorForceField


class ColorOverlap(OEColorOverlap):
    """
    Color overlap.

    Parameters
    ----------
    color_ff : int or OEColorForceField, optional (default
        OEColorFFType_ImplicitMillsDean)
        Color force field.
    all_color : bool, optional (default True)
        Calculate full pairwise color atom overlaps.
    """
    def __init__(
            self, color_ff=OEColorFFType_ImplicitMillsDean, all_color=True):
        super(ColorOverlap, self).__init__()
        if isinstance(color_ff, OEColorForceField):
            self.color_ff = color_ff
        else:
            self.color_ff = OEColorForceField()
            self.color_ff.Init(color_ff)
        self.SetColorForceField(self.color_ff)
        self.SetAllColor(all_color)
        self.ref_mol = None
        self.color_component_engines = None

    def SetRefMol(self, ref_mol):
        """
        Set reference molecule.

        Parameters
        ----------
        ref_mol : OEMol
            Reference molecule.
        """
        self.ref_mol = ref_mol
        return super(ColorOverlap, self).SetRefMol(ref_mol)

    def overlap(self, fit_mol):
        """
        Get color overlap results.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        """
        result = OEColorResults()
        self.ColorScore(fit_mol, result)
        return ColorOverlapResult(result)

    def get_color_components(self, fit_mol):
        """
        Get overlap scores for each color type.

        The color overlap is repeated with a series of different color force
        fields that each have a single color type defined.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        """
        if self.color_component_engines is None:
            self.color_component_engines = self.get_color_component_engines()
        results = []
        for engine in self.color_component_engines:
            results.append(engine.overlap(fit_mol))
        return results

    def get_color_component_engines(self):
        """
        Create a separate ColorOverlap engine for each interaction.
        """
        color_component_engines = []
        color_ff = ColorForceField(self.color_ff)
        for this_color_ff in color_ff.isolate_interactions():
            engine = ColorOverlap(
                color_ff=this_color_ff, all_color=self.GetAllColor())
            engine.SetRefMol(self.ref_mol)
            color_component_engines.append(engine)
        return color_component_engines

    @staticmethod
    def group_color_component_results(results):
        """
        Extract scores from each overlay into arrays for each score type.

        Parameters
        ----------
        scores : array_like
            2D array containing color component results.
        """
        results = np.atleast_2d(results)
        shape = results.shape
        keys = [
            'color_tanimoto', 'color_overlap', 'ref_self_color',
            'fit_self_color']
        data = {key: np.zeros(shape, dtype=float) for key in keys}
        for i, this_results in enumerate(results):
            for j, component_results in enumerate(this_results):
                for k, component_result in enumerate(component_results):
                    for key in keys:
                        data[key][i, j, k] = getattr(component_result, key)
        return data

    def get_ref_color_atom_overlaps(self, fit_mol):
        """
        Get overlap scores for each reference molecule color atom.

        Each color atom in the reference molecule is isolated and the color
        overlap with the fit molecule is scored.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        """
        results = []
        # Use OEMol instead of CreateCopy because otherwise color atoms are
        # added to self.ref_mol
        colored_ref_mol = OEMol(self.ref_mol)
        OEAddColorAtoms(colored_ref_mol, self.color_ff)
        assert OECountColorAtoms(self.ref_mol) == 0
        ref_color_coords = []
        ref_color_types = []
        ref_color_type_names = []
        for ref_color_atom in OEGetColorAtoms(colored_ref_mol):
            coords = colored_ref_mol.GetCoords(ref_color_atom)
            ref_color_type = OEGetColorType(ref_color_atom)
            ref_color_type_name = self.color_ff.GetTypeName(ref_color_type)
            ref_color_coords.append(coords)
            ref_color_types.append(ref_color_type)
            ref_color_type_names.append(ref_color_type_name)
            # Use OEMol instead of CreateCopy because otherwise colored_ref_mol
            # color atoms are deleted by OERemoveColorAtoms
            this_ref_mol = OEMol(colored_ref_mol)
            OERemoveColorAtoms(this_ref_mol)
            OEAddColorAtom(this_ref_mol, OEFloatArray(coords), ref_color_type,
                           ref_color_type_name)
            assert OECountColorAtoms(this_ref_mol) == 1
            super(ColorOverlap, self).SetRefMol(this_ref_mol)
            results.append(self.overlap(fit_mol))
        super(ColorOverlap, self).SetRefMol(self.ref_mol)  # reset ref mol
        return {'overlaps': results,
                'ref_color_coords': ref_color_coords,
                'ref_color_types': ref_color_types,
                'ref_color_type_names': ref_color_type_names}

    @staticmethod
    def group_ref_color_atom_overlaps(results):
        """
        Create a 3D masked array containing all overlap scores.

        Parameters
        ----------
        scores : array_like
            2D array containing reference molecule color atom overlap results.
        """
        # get maximum number of ref color atoms
        # don't use `for result in it` because that gives an array of size 1
        max_size = 0
        it = np.nditer(results, flags=['multi_index', 'refs_ok'])
        for _ in it:
            max_size = max(max_size, results[it.multi_index].size)

        # build a masked array containing results
        # don't use data[it.multi_index][:result.size] because that assigns
        # to a view and not to data
        data = np.ma.masked_all((results.shape[:2] + (max_size,)), dtype=float)
        it = np.nditer(results, flags=['multi_index', 'refs_ok'])
        for _ in it:
            i, j = it.multi_index
            result = results[i, j]
            data[i, j, :result.size] = result
        return data


class ColorOverlapResult(object):
    """
    Color overlap result.

    Parameters
    ----------
    result : OEColorResults
        Color overlap result.
    """
    def __init__(self, result):

        # extract overlap scores
        self.color_tanimoto = result.GetTanimoto()
        self.color_overlap = result.colorscore
        self.ref_self_color = result.refSelfColor
        self.fit_self_color = result.fitSelfColor
