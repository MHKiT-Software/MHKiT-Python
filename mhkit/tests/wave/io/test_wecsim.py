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


class TestWECSim(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    ### WEC-Sim data, no mooring
    def test_read_wecSim_no_mooring(self):
        ws_output = wave.io.wecsim.read_output(
            join(datadir, "RM3_matlabWorkspace_structure.mat")
        )
        self.assertEqual(ws_output["wave"].elevation.name, "elevation")
        self.assertEqual(ws_output["bodies"]["body1"].name, "float")
        self.assertEqual(ws_output["ptos"].name, "PTO1")
        self.assertEqual(ws_output["constraints"].name, "Constraint1")
        self.assertEqual(len(ws_output["mooring"]), 0)
        self.assertEqual(len(ws_output["moorDyn"]), 0)
        self.assertEqual(len(ws_output["ptosim"]), 0)
        self.assertEqual(len(ws_output["cables"]), 0)

    ### WEC-Sim data, with cable
    def test_read_wecSim_cable(self):
        ws_output = wave.io.wecsim.read_output(
            join(datadir, "Cable_matlabWorkspace_structure.mat"),
            to_pandas=False,
        )
        self.assertEqual(ws_output["wave"]["elevation"].name, "elevation")
        self.assertEqual(
            ws_output["bodies"]["body1"]["position_dof1"].name, "position_dof1"
        )
        self.assertEqual(len(ws_output["mooring"]), 0)
        self.assertEqual(len(ws_output["moorDyn"]), 0)
        self.assertEqual(len(ws_output["ptosim"]), 0)
        self.assertEqual(len(ws_output["ptos"]), 0)

    ### WEC-Sim data, with mooring
    def test_read_wecSim_with_mooring(self):
        ws_output = wave.io.wecsim.read_output(
            join(datadir, "RM3MooringMatrix_matlabWorkspace_structure.mat")
        )
        self.assertEqual(ws_output["wave"].elevation.name, "elevation")
        self.assertEqual(ws_output["bodies"]["body1"].name, "float")
        self.assertEqual(ws_output["ptos"].name, "PTO1")
        self.assertEqual(ws_output["constraints"].name, "Constraint1")
        self.assertEqual(len(ws_output["mooring"]), 40001)
        self.assertEqual(len(ws_output["moorDyn"]), 0)
        self.assertEqual(len(ws_output["ptosim"]), 0)
        self.assertEqual(len(ws_output["cables"]), 0)

    ### WEC-Sim data, with moorDyn
    def test_read_wecSim_with_moorDyn(self):
        ws_output = wave.io.wecsim.read_output(
            join(datadir, "RM3MoorDyn_matlabWorkspace_structure.mat")
        )
        self.assertEqual(ws_output["wave"].elevation.name, "elevation")
        self.assertEqual(ws_output["bodies"]["body1"].name, "float")
        self.assertEqual(ws_output["ptos"].name, "PTO1")
        self.assertEqual(ws_output["constraints"].name, "Constraint1")
        self.assertEqual(len(ws_output["mooring"]), 40001)
        self.assertEqual(len(ws_output["moorDyn"]), 7)
        self.assertEqual(len(ws_output["ptosim"]), 0)
        self.assertEqual(len(ws_output["cables"]), 0)


if __name__ == "__main__":
    unittest.main()
