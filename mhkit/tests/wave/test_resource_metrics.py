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
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/wave")))


class TestResourceMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        omega = np.arange(0.1, 3.5, 0.01)
        self.f = omega / (2 * np.pi)
        self.Hs = 2.5
        self.Tp = 8

        file_name = join(datadir, "ValData1.json")
        with open(file_name, "r") as read_file:
            self.valdata1 = pd.DataFrame(json.load(read_file))

        self.valdata2 = {}

        file_name = join(datadir, "ValData2_MC.json")
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2["MC"] = data
        for i in data.keys():
            # Calculate elevation spectra
            elevation = pd.DataFrame(data[i]["elevation"])
            elevation.index = elevation.index.astype(float)
            elevation.sort_index(inplace=True)
            sample_rate = data[i]["sample_rate"]
            NFFT = data[i]["NFFT"]
            self.valdata2["MC"][i]["S"] = wave.resource.elevation_spectrum(
                elevation, sample_rate, NFFT
            )

        file_name = join(datadir, "ValData2_AH.json")
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2["AH"] = data
        for i in data.keys():
            # Calculate elevation spectra
            elevation = pd.DataFrame(data[i]["elevation"])
            elevation.index = elevation.index.astype(float)
            elevation.sort_index(inplace=True)
            sample_rate = data[i]["sample_rate"]
            NFFT = data[i]["NFFT"]
            self.valdata2["AH"][i]["S"] = wave.resource.elevation_spectrum(
                elevation, sample_rate, NFFT
            )

        file_name = join(datadir, "ValData2_CDiP.json")
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2["CDiP"] = data
        for i in data.keys():
            temp = pd.Series(data[i]["S"]).to_frame("S")
            temp.index = temp.index.astype(float)
            self.valdata2["CDiP"][i]["S"] = temp

    @classmethod
    def tearDownClass(self):
        pass

    def test_kfromw(self):
        for i in self.valdata1.columns:
            f = np.array(self.valdata1[i]["w"]) / (2 * np.pi)
            h = self.valdata1[i]["h"]
            rho = self.valdata1[i]["rho"]

            expected = self.valdata1[i]["k"]
            k = wave.resource.wave_number(f, h, rho)
            calculated = k.loc[:, "k"].values
            error = ((expected - calculated) ** 2).sum()  # SSE

            self.assertLess(error, 1e-6)

    def test_kfromw_one_freq(self):
        g = 9.81
        f = 0.1
        h = 1e9
        w = np.pi * 2 * f  # deep water dispersion
        expected = w**2 / g
        calculated = wave.resource.wave_number(f=f, h=h, g=g).values[0][0]
        error = np.abs(expected - calculated)
        self.assertLess(error, 1e-6)

    def test_wave_length(self):
        k_array = np.asarray([1.0, 2.0, 10.0, 3.0])

        k_int = int(k_array[0])
        k_float = k_array[0]
        k_df = pd.DataFrame(k_array, index=[1, 2, 3, 4])
        k_series = k_df[0]

        for l in [k_array, k_int, k_float, k_df, k_series]:
            l_calculated = wave.resource.wave_length(l)
            self.assertTrue(np.all(2.0 * np.pi / l == l_calculated))

    def test_depth_regime(self):
        h = 10

        # non-array like formats
        l_int = 1
        l_float = 1.0
        expected = True
        for l in [l_int, l_float]:
            calculated = wave.resource.depth_regime(l, h)
            self.assertTrue(np.all(expected == calculated))

        # array-like formats
        l_array = np.array([1, 2, 10, 3])
        l_df = pd.DataFrame(l_array, index=[1, 2, 3, 4])
        l_series = l_df[0]
        l_da = xr.DataArray(l_series)
        l_da.name = "data"
        l_ds = l_da.to_dataset()
        expected = [True, True, False, True]
        for l in [l_array, l_series, l_da, l_ds]:
            calculated = wave.resource.depth_regime(l, h)
            self.assertTrue(np.all(expected == calculated))

        # special formatting for pd.DataFrame
        for l in [l_df]:
            calculated = wave.resource.depth_regime(l, h)
            self.assertTrue(np.all(expected == calculated[0]))

    def test_wave_celerity(self):
        # Depth regime ratio
        dr_ratio = 2

        # small change in f will give similar value cg
        f = np.linspace(20.0001, 20.0005, 5)

        # Choose index to spike at. cg spike is inversly proportional to k
        k_idx = 2
        k_tmp = [1, 1, 0.5, 1, 1]
        k = pd.DataFrame(k_tmp, index=f)

        # all shallow
        cg_shallow1 = wave.resource.wave_celerity(k, h=0.0001, depth_check=True)
        cg_shallow2 = wave.resource.wave_celerity(k, h=0.0001, depth_check=False)
        self.assertTrue(
            all(cg_shallow1.squeeze().values == cg_shallow2.squeeze().values)
        )

        # all deep
        cg = wave.resource.wave_celerity(k, h=1000, depth_check=True)
        self.assertTrue(all(np.pi * f / k.squeeze().values == cg.squeeze().values))

    def test_energy_flux_deep(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        Te = wave.resource.energy_period(S)
        Hm0 = wave.resource.significant_wave_height(S)

        rho = 1025
        g = 9.80665
        coeff = rho * (g**2) / (64 * np.pi)
        J = coeff * (Hm0.squeeze() ** 2) * Te.squeeze()

        h = -1  # not used when deep=True
        J_calc = wave.resource.energy_flux(S, h, deep=True)

        self.assertTrue(J_calc.squeeze() == J)

    def test_energy_flux_shallow(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        Te = wave.resource.energy_period(S)
        Hm0 = wave.resource.significant_wave_height(S)

        rho = 1025
        g = 9.80665
        coeff = rho * (g**2) / (64 * np.pi)
        J = coeff * (Hm0.squeeze() ** 2) * Te.squeeze()

        h = 1000  # effectively deep but without assumptions
        J_calc = wave.resource.energy_flux(S, h, deep=False)
        err = np.abs(J_calc.squeeze() - J)
        self.assertLess(err, 1e-6)

    def test_moments(self):
        for file_i in self.valdata2.keys():  # for each file MC, AH, CDiP
            datasets = self.valdata2[file_i]
            for s in datasets.keys():  # for each set
                data = datasets[s]
                for m in data["m"].keys():
                    expected = data["m"][m]
                    S = data["S"]
                    if s == "CDiP1" or s == "CDiP6":
                        f_bins = pd.Series(data["freqBinWidth"])
                    else:
                        f_bins = None

                    calculated = wave.resource.frequency_moment(
                        S, int(m), frequency_bins=f_bins
                    ).iloc[0, 0]
                    error = np.abs(expected - calculated) / expected

                    self.assertLess(error, 0.01)

    def test_energy_period_to_peak_period(self):
        # This test checks that if we perform the
        # Te to Tp conversion, we create a spectrum
        # (using Tp) that has the provided Te.
        Hs = 2.5
        Te = np.linspace(5, 20, 10)
        gamma = np.linspace(1, 7, 7)

        for g in gamma:
            for T in Te:
                Tp = wave.resource.energy_period_to_peak_period(T, g)

                f = np.linspace(1 / (10 * Tp), 3 / Tp, 100)
                S = wave.resource.jonswap_spectrum(f, Tp, Hs, g)

                Te_calc = wave.resource.energy_period(S).values[0][0]

                error = np.abs(T - Te_calc) / Te_calc
                self.assertLess(error, 0.01)

    def test_metrics(self):
        for file_i in self.valdata2.keys():  # for each file MC, AH, CDiP
            datasets = self.valdata2[file_i]

            for s in datasets.keys():  # for each set
                data = datasets[s]
                S = data["S"]
                if file_i == "CDiP":
                    f_bins = pd.Series(data["freqBinWidth"])
                else:
                    f_bins = None

                # Hm0
                expected = data["metrics"]["Hm0"]
                calculated = wave.resource.significant_wave_height(
                    S, frequency_bins=f_bins
                ).iloc[0, 0]
                error = np.abs(expected - calculated) / expected
                # print('Hm0', expected, calculated, error)
                self.assertLess(error, 0.01)

                # Te
                expected = data["metrics"]["Te"]
                calculated = wave.resource.energy_period(S, frequency_bins=f_bins).iloc[
                    0, 0
                ]
                error = np.abs(expected - calculated) / expected
                # print('Te', expected, calculated, error)
                self.assertLess(error, 0.01)

                # T0
                expected = data["metrics"]["T0"]
                calculated = wave.resource.average_zero_crossing_period(
                    S, frequency_bins=f_bins
                ).iloc[0, 0]
                error = np.abs(expected - calculated) / expected
                # print('T0', expected, calculated, error)
                self.assertLess(error, 0.01)

                # Tc
                expected = data["metrics"]["Tc"]
                calculated = (
                    wave.resource.average_crest_period(
                        S,
                        # Tc = Tavg**2
                        frequency_bins=f_bins,
                    ).iloc[0, 0]
                    ** 2
                )
                error = np.abs(expected - calculated) / expected
                # print('Tc', expected, calculated, error)
                self.assertLess(error, 0.01)

                # Tm
                expected = np.sqrt(data["metrics"]["Tm"])
                calculated = wave.resource.average_wave_period(
                    S, frequency_bins=f_bins
                ).iloc[0, 0]
                error = np.abs(expected - calculated) / expected
                # print('Tm', expected, calculated, error)
                self.assertLess(error, 0.01)

                # Tp
                expected = data["metrics"]["Tp"]
                calculated = wave.resource.peak_period(S).iloc[0, 0]
                error = np.abs(expected - calculated) / expected
                # print('Tp', expected, calculated, error)
                self.assertLess(error, 0.001)

                # e
                expected = data["metrics"]["e"]
                calculated = wave.resource.spectral_bandwidth(
                    S, frequency_bins=f_bins
                ).iloc[0, 0]
                error = np.abs(expected - calculated) / expected
                # print('e', expected, calculated, error)
                self.assertLess(error, 0.001)

                # J
                if file_i != "CDiP":
                    for i, j in zip(data["h"], data["J"]):
                        expected = data["J"][j]
                        calculated = wave.resource.energy_flux(S, i)
                        error = np.abs(expected - calculated.values) / expected
                        self.assertLess(error, 0.1)

                # v
                if file_i == "CDiP":
                    # this should be updated to run on other datasets
                    expected = data["metrics"]["v"]
                    calculated = wave.resource.spectral_width(
                        S, frequency_bins=f_bins
                    ).iloc[0, 0]
                    error = np.abs(expected - calculated) / expected
                    self.assertLess(error, 0.01)

                if file_i == "MC":
                    expected = data["metrics"]["v"]
                    # testing that default uniform frequency bin widths works
                    calculated = wave.resource.spectral_width(S).iloc[0, 0]
                    error = np.abs(expected - calculated) / expected
                    self.assertLess(error, 0.01)

    def test_plot_elevation_timeseries(self):
        filename = abspath(join(plotdir, "wave_plot_elevation_timeseries.png"))
        if isfile(filename):
            os.remove(filename)

        data = self.valdata2["MC"]
        temp = pd.DataFrame(data[list(data.keys())[0]]["elevation"])
        temp.index = temp.index.astype(float)
        temp.sort_index(inplace=True)
        eta = temp.iloc[0:100, :]

        plt.figure()
        wave.graphics.plot_elevation_timeseries(eta)
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))


class TestPlotResouceCharacterizations(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        f_name = "Hm0_Te_46022.json"
        self.Hm0Te = pd.read_json(join(datadir, f_name))

    @classmethod
    def tearDownClass(self):
        pass

    def test_plot_avg_annual_energy_matrix(self):
        filename = abspath(join(plotdir, "avg_annual_scatter_table.png"))
        if isfile(filename):
            os.remove(filename)

        Hm0Te = self.Hm0Te
        Hm0Te.drop(Hm0Te[Hm0Te.Hm0 > 20].index, inplace=True)
        J = np.random.random(len(Hm0Te)) * 100

        plt.figure()
        fig = wave.graphics.plot_avg_annual_energy_matrix(
            Hm0Te.Hm0, Hm0Te.Te, J, Hm0_bin_size=0.5, Te_bin_size=1
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_monthly_cumulative_distribution(self):
        filename = abspath(join(plotdir, "monthly_cumulative_distribution.png"))
        if isfile(filename):
            os.remove(filename)

        a = pd.date_range(start="1/1/2010", periods=10000, freq="h")
        S = pd.Series(np.random.random(len(a)), index=a)
        ax = wave.graphics.monthly_cumulative_distribution(S)
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))


if __name__ == "__main__":
    unittest.main()
