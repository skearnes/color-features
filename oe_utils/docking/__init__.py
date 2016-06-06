"""
OEDocking utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from openeye.oechem import *
from openeye.oedocking import *


def read_receptor(filename):
    """
    Read a receptor from a file.

    Parameters
    ----------
    filename : str
        Filename.
    """
    receptor = OEMol()
    OEReadReceptorFile(receptor, filename)
    assert receptor.IsValid()
    return receptor


class Docker(OEDock):
    """
    Docking.

    Parameters
    ----------
    receptor : OEMol
        Receptor.
    scoring : int, optional (default OEDockMethod_Default)
        Docking method.
    resolution : int, optional (default OESearchResolution_Default)
        Docking search resolution.
    n_poses : int, optional (default 1)
        Number of poses to return for each docked molecule.
    component_scores : bool, optional (default True)
        Whether to save individual components of the total score to SD fields.
    annotate : bool, optional (default False)
        Whether to save the contributions of individual atoms to the docking
        score.
    """
    def __init__(self, receptor, scoring=OEDockMethod_Default,
                 resolution=OESearchResolution_Default, n_poses=1,
                 component_scores=True, annotate=False):
        super(Docker, self).__init__(scoring, resolution)
        self.n_poses = n_poses
        self.component_scores = component_scores
        self.annotate = annotate
        if not self.Initialize(receptor):
            raise RuntimeError('Docking engine initialization failed.')
        self.scores = [self.GetName()]  # score names
        if component_scores:
            for score in sorted(self.GetComponentNames()):
                self.scores.append(score)

    def __call__(self, mol):
        """
        Dock a multiconformer molecule.

        Parameters
        ----------
        mol : OEMol
            Molecule to dock.
        """
        return self.dock(mol)

    def dock(self, mol):
        """
        Dock a multiconformer molecule.

        Component scores are saved in SD fields.

        Parameters
        ----------
        mol : OEMol
            Molecule to dock.
        """
        poses = OEMol()
        self.DockMultiConformerMolecule(poses, mol, self.n_poses)
        if poses.IsValid():
            return poses, self.score(poses)
        else:
            return mol, None

    def score(self, poses):
        """
        Get scores and annotations for poses.

        Parameters
        ----------
        poses : OEMol
            Docked poses.
        """
        scores = np.zeros((poses.NumConfs(), len(self.scores)), dtype=float)
        for i, pose in enumerate(poses.GetConfs()):
            scores[i, 0] = self.ScoreLigand(pose)
            OESetSDData(pose, self.GetName(), str(scores[i, 0]))
            if self.component_scores:
                for j, score in enumerate(self.scores[1:]):
                    scores[i, j + 1] = self.ScoreLigandComponent(pose, score)
                    OESetSDData(pose, score, str(scores[i, j + 1]))
        if self.annotate:
            self.AnnotatePose(poses)
        return np.squeeze(scores)


class HybridDocker(Docker):
    """
    Hybrid docking.

    Parameters
    ----------
    receptor : OEMol
        Receptor.
    scoring : int, optional (default OEDockMethod_Hybrid)
        Docking method.
    resolution : int, optional (default OESearchResolution_Default)
        Docking search resolution.
    n_poses : int, optional (default 1)
        Number of poses to return for each docked molecule.
    component_scores : bool, optional (default True)
        Whether to save individual components of the total score.
    annotate : bool, optional (default False)
        Whether to save the contributions of individual atoms to the docking
        score.
    """
    def __init__(self, receptor, scoring=OEDockMethod_Hybrid,
                 resolution=OESearchResolution_Default, n_poses=1,
                 component_scores=True, annotate=False):
        super(HybridDocker, self).__init__(receptor, scoring, resolution,
                                           n_poses, component_scores, annotate)
