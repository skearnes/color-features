"""
OEChem utilities.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from openeye.oechem import *


class MolReader(object):
    """
    Read molecules.

    Parameters
    ----------
    filename : str, optional
        Input filename.
    conf_test : OEConfTestBase, optional
        Test used to recognize multiconformer molecules. Defaults to
        OEOmegaConfTest.
    """
    def __init__(self, filename=None, conf_test=None):
        self.ifs = oemolistream()
        if conf_test is None:
            conf_test = OEOmegaConfTest(True)
        self.ifs.SetConfTest(conf_test)
        if filename is not None:
            self.open(filename)

    def __del__(self):
        """
        Close the input file.
        """
        self.close()

    def open(self, filename):
        """
        Open a file for reading.

        Parameters
        ----------
        filename : str, optional
            Input filename.
        """
        if not self.ifs.open(filename):
            raise IOError("Cannot open file '{}'.".format(filename))

    def close(self):
        """
        Close the input file.
        """
        self.ifs.close()

    def get_mols(self):
        """
        Read molecules from file.
        """
        for mol in self.ifs.GetOEMols():
            mol = OEMol(mol)  # otherwise mols are destroyed
            yield mol

    def get_batches(self, batch_size):
        """
        Read batches of molecules from file.

        Parameters
        ----------
        batch_size : int
            Batch size.
        """
        batch = []
        for mol in self.get_mols():
            batch.append(mol)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch


class MolWriter(object):
    """
    Write molecules.

    Parameters
    ----------
    filename : str, optional
        Output filename for molecules.
    """
    def __init__(self, filename=None):
        self.ofs = oemolostream()
        if filename is not None:
            self.open(filename)

    def __del__(self):
        """
        Close the output file.
        """
        self.close()

    def open(self, filename):
        """
        Open a file for writing.

        Parameters
        ----------
        filename : str
            Output filename for molecules.
        """
        if not self.ofs.open(filename):
            raise IOError("Cannot open file '{}'.".format(filename))

    def close(self):
        """
        Close the output file.
        """
        self.ofs.close()

    def write(self, mols):
        """
        Write molecules to the output file stream.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        for mol in mols:
            OEWriteMolecule(self.ofs, mol)
