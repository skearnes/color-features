"""
OEShape utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from openeye.oechem import *
from openeye.oeshape import *

from oe_utils.chem import transform


class ROCS(OEBestOverlay):
    """
    Rapid overlay of chemical structures.

    Parameters
    ----------
    use_hydrogens : bool, optional (default False)
        Include hydrogens in shape overlap.
    color_ff : int, optional (default OEColorFFType_ImplicitMillsDean)
        Color force field.
    color_opt : bool, optional (default True)
        Include color in overlap optimization.
    all_color : bool, optional (default True)
        Calculate full pairwise color atom overlaps.
    radii_approx : int, optional (default OEOverlapRadii_All)
        Radii approximation method.
    overlap_method : int, optional (default OEOverlapMethod_Analytic)
        Overlap method.
    initial_orientation : int, optional (default OEBOOrientation_Inertial)
        Method for generating initial overlap orientations.
    """
    def __init__(
            self, use_hydrogens=False,
            color_ff=OEColorFFType_ImplicitMillsDean, color_opt=True,
            all_color=True, radii_approx=OEOverlapRadii_All,
            overlap_method=OEOverlapMethod_Analytic,
            initial_orientation=OEBOOrientation_Inertial):
        super(ROCS, self).__init__()
        self.SetUseHydrogens(use_hydrogens)
        self.SetColorForceField(color_ff)
        self.SetColorOptimize(color_opt)
        self.SetAllColor(all_color)
        self.SetRadiiApproximation(radii_approx)
        self.SetMethod(overlap_method)
        self.SetInitialOrientation(initial_orientation)

    def overlay(self, fit_mol, sort=OEHighestTanimotoCombo(), n_best=None):
        """
        Perform ROCS alignment and scoring.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        sort : OEBinaryPredicate, optional (default OEHighestTanimotoCombo())
            Sort predicate.
        n_best : int, optional
            Number of results to return after sorting. If None, all results are
            returned.
        """
        assert n_best is None or n_best > 0
        result = self.Overlay(fit_mol)
        for i, this_result in enumerate(
                self.parse_overlay_results(result, sort)):
            if n_best is not None and i >= n_best:
                break
            yield this_result

    def get_best_overlay(self, fit_mol, sort=OEHighestTanimotoCombo()):
        """
        Perform ROCS alignment and scoring and return the best result.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        sort : OEBinaryPredicate, optional (default OEHighestTanimotoCombo())
            Sort predicate.
        """
        results = list(self.overlay(fit_mol, sort=sort, n_best=1))
        assert len(results) == 1  # sanity check
        return results[0]

    @staticmethod
    def parse_overlay_results(result, sort=OEHighestTanimotoCombo()):
        """
        Parse overlay results.

        Parameters
        ----------
        result : OEBestOverlayResultIter
            Overlay result returned by OEBestOverlay.Overlay.
        sort : OEBinaryPredicate, optional (default OEHighestTanimotoCombo())
            Sort predicate.
        """
        score_iter = OEBestOverlayScoreIter()
        OESortOverlayScores(score_iter, result, sort)
        for score in score_iter:
            yield ROCSResult(score)

    @staticmethod
    def get_aligned_confs(ref_mol, fit_mol, result, copy=True):
        """
        Get aligned conformers matching a given result.

        Parameters
        ----------
        ref_mol : OEMol
            Reference molecule.
        fit_mol : OEMol
            Fit molecule.
        result : ROCSResult
            Overlay result.
        copy : bool, optional (default True)
            Create a copy of the fit conformer before transforming.
        """
        ref_conf = ref_mol.GetConf(OEHasConfIdx(result.ref_conf_idx))
        fit_conf = fit_mol.GetConf(OEHasConfIdx(result.fit_conf_idx))
        fit_conf = result.transform(fit_conf, copy=copy)
        return ref_conf, fit_conf

    @staticmethod
    def group_results(results):
        """
        Extract scores from each overlay into arrays for each score type.

        Parameters
        ----------
        scores : array_like
            2D array containing overlay results.
        """
        results = np.atleast_2d(results)
        shape = results.shape
        scalar_keys = [
            'ref_conf_idx', 'fit_conf_idx', 'shape_tanimoto', 'shape_overlap',
            'ref_self_shape', 'fit_self_shape', 'color_tanimoto',
            'color_overlap', 'ref_self_color', 'fit_self_color']
        data = {key: np.zeros(shape, dtype=float) for key in scalar_keys}
        data['rotation'] = np.zeros(shape + (9,), dtype=float)
        data['translation'] = np.zeros(shape + (3,), dtype=float)
        for i, this_results in enumerate(results):
            for j, result in enumerate(this_results):
                for key in scalar_keys + ['rotation', 'translation']:
                    data[key][i, j] = getattr(result, key)
        return data


class ROCSResult(object):
    """
    ROCS overlay result.

    Parameters
    ----------
    result : OEBestOverlayScore
        Overlay result from OEBestOverlayScoreIter.
    """
    def __init__(self, result):

        # conformer indices
        self.ref_conf_idx = result.refconfidx
        self.fit_conf_idx = result.fitconfidx

        # shape
        self.shape_tanimoto = result.GetShapeTanimoto()
        self.shape_overlap = result.overlap
        self.ref_self_shape = result.refSelfOverlap
        self.fit_self_shape = result.fitSelfOverlap

        # color
        self.color_tanimoto = result.GetColorTanimoto()
        self.color_overlap = result.colorscore
        self.ref_self_color = result.refSelfColor
        self.fit_self_color = result.fitSelfColor

        # overlay
        rotation = OEFloatArray(9)
        result.GetRotMatrix(rotation)
        self.rotation = np.asarray(rotation, dtype=float)
        translation = OEFloatArray(3)
        result.GetTranslation(translation)
        self.translation = np.asarray(translation, dtype=float)
        self.transformer = transform.MolTransform(
            rotation=self.rotation, translation=self.translation)

    def transform(self, fit_mol, copy=True):
        """
        Transform fit_mol into alignment with ref_mol.

        Parameters
        ----------
        fit_mol : OEMol
            Fit molecule.
        copy : bool, optional (default True)
            Create a copy of the fit conformer before transforming.
        """
        return self.transformer.transform(fit_mol, copy=copy)
