from os.path import abspath, dirname, join, isfile, normpath, relpath
import matplotlib.pylab as plt
import xarray.testing as xrt
import mhkit.wave as wave
import pandas as pd
import numpy as np
import unittest
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/wave")))


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(123)
        Hm0 = np.random.rayleigh(4, 100000)
        Te = np.random.normal(4.5, 0.8, 100000)
        P = np.random.normal(200, 40, 100000)
        J = np.random.normal(300, 10, 100000)
        ndbc_data_file = join(datadir, "data.txt")
        [raw_ndbc_data, meta] = wave.io.ndbc.read_file(ndbc_data_file)
        self.S = raw_ndbc_data.T

        self.data = pd.DataFrame({"Hm0": Hm0, "Te": Te, "P": P, "J": J})
        self.Hm0_bins = np.arange(0, 19, 0.5)
        self.Te_bins = np.arange(0, 9, 1)
        self.expected_stats = [
            "mean",
            "std",
            "median",
            "count",
            "sum",
            "min",
            "max",
            "freq",
        ]

    @classmethod
    def tearDownClass(self):
        pass

    def test_capture_length(self):
        L = wave.performance.capture_length(self.data["P"], self.data["J"])
        L_stats = wave.performance.statistics(L)

        self.assertAlmostEqual(L_stats["mean"], 0.6676, 3)

    def test_capture_length_matrix(self):
        L = wave.performance.capture_length(self.data["P"], self.data["J"])
        LM = wave.performance.capture_length_matrix(
            self.data["Hm0"], self.data["Te"], L, "std", self.Hm0_bins, self.Te_bins
        )

        self.assertEqual(LM.shape, (38, 9))
        self.assertEqual(LM.isna().sum().sum(), 131)

    def test_wave_energy_flux_matrix(self):
        JM = wave.performance.wave_energy_flux_matrix(
            self.data["Hm0"],
            self.data["Te"],
            self.data["J"],
            "mean",
            self.Hm0_bins,
            self.Te_bins,
        )

        self.assertEqual(JM.shape, (38, 9))
        self.assertEqual(JM.isna().sum().sum(), 131)

    def test_power_matrix(self):
        L = wave.performance.capture_length(self.data["P"], self.data["J"])
        LM = wave.performance.capture_length_matrix(
            self.data["Hm0"], self.data["Te"], L, "mean", self.Hm0_bins, self.Te_bins
        )
        JM = wave.performance.wave_energy_flux_matrix(
            self.data["Hm0"],
            self.data["Te"],
            self.data["J"],
            "mean",
            self.Hm0_bins,
            self.Te_bins,
        )
        PM = wave.performance.power_matrix(LM, JM)

        self.assertEqual(PM.shape, (38, 9))
        self.assertEqual(PM.isna().sum().sum(), 131)

    def test_mean_annual_energy_production(self):
        L = wave.performance.capture_length(self.data["P"], self.data["J"])
        maep = wave.performance.mean_annual_energy_production_timeseries(
            L, self.data["J"]
        )

        self.assertAlmostEqual(maep, 1754020.077, 2)

    def test_plot_matrix(self):
        filename = abspath(join(plotdir, "wave_plot_matrix.png"))
        if isfile(filename):
            os.remove(filename)

        M = wave.performance.wave_energy_flux_matrix(
            self.data["Hm0"],
            self.data["Te"],
            self.data["J"],
            "mean",
            self.Hm0_bins,
            self.Te_bins,
        )

        plt.figure()
        wave.graphics.plot_matrix(M)
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_powerperformance_workflow(self):
        filename = abspath(join(plotdir, "Capture Length Matrix mean.png"))
        if isfile(filename):
            os.remove(filename)
        P = pd.Series(np.random.normal(200, 40, 743), index=self.S.columns)
        statistic = ["mean"]
        savepath = plotdir
        show_values = True
        h = 60
        expected = 401239.4822345051
        CM, MAEP = wave.performance.power_performance_workflow(
            self.S, h, P, statistic, savepath=savepath, show_values=show_values
        )

        self.assertTrue(isfile(filename))
        self.assertEqual(list(CM.data_vars), self.expected_stats)

        error = (expected - MAEP) / expected  # SSE

        self.assertLess(error, 1e-6)


if __name__ == "__main__":
    unittest.main()
