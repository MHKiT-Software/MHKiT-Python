import unittest
from os.path import abspath, dirname, join, isfile, normpath, relpath
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mhkit.tidal as tidal

testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/tidal")))


class TestResource(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        file_name = join(datadir, "s08010.json")
        self.data, self.metadata = tidal.io.noaa.read_noaa_json(file_name)
        self.data.s = self.data.s / 100.0  # convert to m/s
        self.flood = 171.5
        self.ebb = 354.5

    @classmethod
    def tearDownClass(self):
        pass

    def test_exceedance_probability(self):
        df = pd.DataFrame.from_records({"vals": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])})
        df["exceedance_probability"] = tidal.resource.exceedance_probability(df.vals)
        self.assertEqual(df["exceedance_probability"].min(), 10)
        self.assertEqual(df["exceedance_probability"].max(), 90)

    def test_principal_flow_directions(self):
        width_direction = 10
        direction1, direction2 = tidal.resource.principal_flow_directions(
            self.data.d, width_direction
        )
        self.assertEqual(direction1, 172.0)
        self.assertEqual(round(direction2, 1), round(352.3, 1))

    def test_plot_current_timeseries(self):
        filename = abspath(join(plotdir, "tidal_plot_current_timeseries.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        tidal.graphics.plot_current_timeseries(self.data.d, self.data.s, 172)
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_joint_probability_distribution(self):
        filename = abspath(
            join(plotdir, "tidal_plot_joint_probability_distribution.png")
        )
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        tidal.graphics.plot_joint_probability_distribution(
            self.data.d, self.data.s, 1, 0.1
        )
        plt.savefig(f"{filename}")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_rose(self):
        filename = abspath(join(plotdir, "tidal_plot_rose.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        tidal.graphics.plot_rose(self.data.d, self.data.s, 1, 0.1)
        plt.savefig(f"{filename}")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_tidal_phase_probability(self):
        filename = abspath(join(plotdir, "tidal_plot_tidal_phase_probability.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        tidal.graphics.tidal_phase_probability(
            self.data.d, self.data.s, self.flood, self.ebb
        )
        plt.savefig(f"{filename}")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_tidal_phase_exceedance(self):
        filename = abspath(join(plotdir, "tidal_plot_tidal_phase_exceedance.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        tidal.graphics.tidal_phase_exceedance(
            self.data.d, self.data.s, self.flood, self.ebb
        )
        plt.savefig(f"{filename}")
        plt.close()

        self.assertTrue(isfile(filename))


if __name__ == "__main__":
    unittest.main()
