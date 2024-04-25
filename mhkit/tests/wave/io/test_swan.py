from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from random import seed, randint
import matplotlib.pylab as plt
from datetime import datetime
import xarray.testing as xrt
import mhkit.wave as wave
from io import StringIO
import pandas as pd
import xarray as xr
import numpy as np
import contextlib
import unittest
import netCDF4
import inspect
import pickle
import time
import json
import sys
import os


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir, "..", "..", "..", "..", "examples", "data", "wave"))


class TestSWAN(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        swan_datadir = join(datadir, "swan")
        self.table_file = join(swan_datadir, "SWANOUT.DAT")
        self.swan_block_mat_file = join(swan_datadir, "SWANOUT.MAT")
        self.swan_block_txt_file = join(swan_datadir, "SWANOUTBlock.DAT")
        self.expected_table = pd.read_csv(
            self.table_file,
            sep="\s+",
            comment="%",
            names=["Xp", "Yp", "Hsig", "Dir", "RTpeak", "TDir"],
        )

    @classmethod
    def tearDownClass(self):
        pass

    def test_read_table(self):
        swan_table, swan_meta = wave.io.swan.read_table(self.table_file)
        assert_frame_equal(self.expected_table, swan_table)

    def test_read_block_mat(self):
        swanBlockMat, metaDataMat = wave.io.swan.read_block(self.swan_block_mat_file)
        self.assertEqual(len(swanBlockMat), 4)
        self.assertAlmostEqual(
            self.expected_table["Hsig"].sum(),
            swanBlockMat["Hsig"].sum().sum(),
            places=1,
        )

    def test_read_block_txt(self):
        swanBlockTxt, metaData = wave.io.swan.read_block(self.swan_block_txt_file)
        self.assertEqual(len(swanBlockTxt), 4)
        sumSum = swanBlockTxt["Significant wave height"].sum().sum()
        self.assertAlmostEqual(self.expected_table["Hsig"].sum(), sumSum, places=-2)

    def test_read_block_txt_xarray(self):
        swanBlockTxt, metaData = wave.io.swan.read_block(
            self.swan_block_txt_file, to_pandas=False
        )
        self.assertEqual(len(swanBlockTxt), 4)
        sumSum = swanBlockTxt["Significant wave height"].sum().sum()
        self.assertAlmostEqual(self.expected_table["Hsig"].sum(), sumSum, places=-2)

    def test_block_to_table(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        df = pd.DataFrame(np.random.rand(5, 5), columns=x, index=y)
        dff = wave.io.swan.block_to_table(df)
        self.assertEqual(dff.shape, (len(x) * len(y), 3))
        self.assertTrue(all(dff.x.unique() == np.unique(x)))

    def test_dictionary_of_block_to_table(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        df = pd.DataFrame(np.random.rand(5, 5), columns=x, index=y)
        keys = ["data1", "data2"]
        data = [df, df]
        dict_of_dfs = dict(zip(keys, data))
        dff = wave.io.swan.dictionary_of_block_to_table(dict_of_dfs)
        self.assertEqual(dff.shape, (len(x) * len(y), 2 + len(keys)))
        self.assertTrue(all(dff.x.unique() == np.unique(x)))
        for key in keys:
            self.assertTrue(key in dff.keys())


if __name__ == "__main__":
    unittest.main()
