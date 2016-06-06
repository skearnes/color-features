"""
Test OEChem utilities.
"""
import shutil
import tempfile
import unittest

from .. import MolReader, MolWriter


class TestChem(unittest.TestCase):
    """
    Test OEChem utilities.
    """
    def setUp(self):
        """
        Set up tests.
        """
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
        self.temp_dir = tempfile.mkdtemp()
        _, self.filename = tempfile.mkstemp(dir=self.temp_dir, suffix='.smi')
        with open(self.filename, 'wb') as f:
            f.write(smiles)
        self.reader = MolReader()
        self.writer = MolWriter()

    def tearDown(self):
        """
        Clean up tests.
        """
        shutil.rmtree(self.temp_dir)

    def test_read(self):
        """
        Test MolReader.
        """
        self.reader.open(self.filename)
        mols = list(self.reader.get_mols())
        assert len(mols) == 1
        for mol in mols:
            assert mol.IsValid()

    def test_write(self):
        """
        Test MolWriter.
        """
        self.reader.open(self.filename)
        mols = list(self.reader.get_mols())
        self.reader.close()

        # write molecules to new file
        _, filename = tempfile.mkstemp(dir=self.temp_dir, suffix='.sdf')
        self.writer.open(filename)
        self.writer.write(mols)

        # read the written file
        self.reader.open(filename)
        mols = list(self.reader.get_mols())
        assert len(mols) == 1
        for mol in mols:
            assert mol.IsValid()
