from os.path import abspath, dirname, join, isfile, normpath, relpath
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
import scipy.interpolate as interp
import matplotlib.pylab as plt
import mhkit.river as river
import pandas as pd
import numpy as np
import unittest
import netCDF4
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "river"))


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.diameter = 1
        self.height = 2
        self.width = 3
        self.diameters = [1, 2, 3, 4]

    @classmethod
    def tearDownClass(self):
        pass

    def test_circular(self):
        eq, ca = river.performance.circular(self.diameter)
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 0.25 * np.pi * self.diameter**2.0)

    def test_ducted(self):
        eq, ca = river.performance.ducted(self.diameter)
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 0.25 * np.pi * self.diameter**2.0)

    def test_rectangular(self):
        eq, ca = river.performance.rectangular(self.height, self.width)
        self.assertAlmostEqual(eq, 2.76, places=2)
        self.assertAlmostEqual(ca, self.height * self.width, places=2)

    def test_multiple_circular(self):
        eq, ca = river.performance.multiple_circular(self.diameters)
        self.assertAlmostEqual(eq, 5.48, places=2)
        self.assertAlmostEqual(ca, 23.56, places=2)

    def test_tip_speed_ratio(self):
        rotor_speed = [15, 16, 17, 18]  # create array of rotor speeds
        rotor_diameter = 77  # diameter of rotor for GE 1.5
        inflow_speed = [13, 13, 13, 13]  # array of wind speeds
        TSR_answer = [4.7, 5.0, 5.3, 5.6]

        TSR = river.performance.tip_speed_ratio(
            np.asarray(rotor_speed) / 60, rotor_diameter, inflow_speed
        )

        for i, j in zip(TSR, TSR_answer):
            self.assertAlmostEqual(i, j, delta=0.05)

    def test_power_coefficient(self):
        # data obtained from power performance report of wind turbine
        inflow_speed = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        power_out = np.asarray([59, 304, 742, 1200, 1400, 1482, 1497, 1497, 1511])
        capture_area = 4656.63
        rho = 1.225
        Cp_answer = [0.320, 0.493, 0.508, 0.421, 0.284, 0.189, 0.128, 0.090, 0.066]

        Cp = river.performance.power_coefficient(
            power_out * 1000, inflow_speed, capture_area, rho
        )

        for i, j in zip(Cp, Cp_answer):
            self.assertAlmostEqual(i, j, places=2)


if __name__ == "__main__":
    unittest.main()
