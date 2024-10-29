from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import xarray as xr
import mhkit.wave as wave
import pandas as pd
import numpy as np
import unittest
import pytest
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/wave")))


class TestResourceSpectrum(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        Trep = 600
        df = 1 / Trep
        self.f = np.arange(0, 1, df)
        self.Hs = 2.5
        self.Tp = 8
        self.t = np.arange(0, Trep, 0.05)

    @classmethod
    def tearDownClass(self):
        pass

    def test_pierson_moskowitz_spectrum(self):
        S = wave.resource.pierson_moskowitz_spectrum(self.f, self.Tp, self.Hs)
        Hm0 = wave.resource.significant_wave_height(S).item()
        Tp0 = wave.resource.peak_period(S).item()

        errorHm0 = np.abs(self.Tp - Tp0) / self.Tp
        errorTp0 = np.abs(self.Hs - Hm0) / self.Hs

        self.assertLess(errorHm0, 0.01)
        self.assertLess(errorTp0, 0.01)

    def test_pierson_moskowitz_spectrum_zero_freq(self):
        df = 0.1
        f_zero = np.arange(0, 1, df)
        f_nonzero = np.arange(df, 1, df)

        S_zero = wave.resource.pierson_moskowitz_spectrum(f_zero, self.Tp, self.Hs)
        S_nonzero = wave.resource.pierson_moskowitz_spectrum(
            f_nonzero, self.Tp, self.Hs
        )

        self.assertEqual(S_zero.values.squeeze()[0], 0.0)
        self.assertGreater(S_nonzero.values.squeeze()[0], 0.0)

    def test_jonswap_spectrum(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        Hm0 = wave.resource.significant_wave_height(S).item()
        Tp0 = wave.resource.peak_period(S).item()

        errorHm0 = np.abs(self.Tp - Tp0) / self.Tp
        errorTp0 = np.abs(self.Hs - Hm0) / self.Hs

        self.assertLess(errorHm0, 0.01)
        self.assertLess(errorTp0, 0.01)

    def test_jonswap_spectrum_zero_freq(self):
        df = 0.1
        f_zero = np.arange(0, 1, df)
        f_nonzero = np.arange(df, 1, df)

        S_zero = wave.resource.jonswap_spectrum(f_zero, self.Tp, self.Hs)
        S_nonzero = wave.resource.jonswap_spectrum(f_nonzero, self.Tp, self.Hs)

        self.assertEqual(S_zero.values.squeeze()[0], 0.0)
        self.assertGreater(S_nonzero.values.squeeze()[0], 0.0)

    def test_surface_elevation_phases_xr_and_pd(self):
        S0 = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        S1 = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs * 1.1)
        S = pd.concat([S0, S1], axis=1)

        phases_np = np.random.rand(S.shape[0], S.shape[1]) * 2 * np.pi
        phases_pd = pd.DataFrame(phases_np, index=S.index, columns=S.columns)
        phases_xr = xr.Dataset(phases_pd)

        eta_xr = wave.resource.surface_elevation(S, self.t, phases=phases_xr, seed=1)
        eta_pd = wave.resource.surface_elevation(S, self.t, phases=phases_pd, seed=1)

        assert_frame_equal(eta_xr, eta_pd)

    def test_surface_elevation_frequency_bins_np_and_pd(self):
        S0 = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        S1 = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs * 1.1)
        S = pd.concat([S0, S1], axis=1)

        eta0 = wave.resource.surface_elevation(S, self.t, seed=1)

        f_bins_np = np.array([np.diff(S.index)[0]] * len(S))
        f_bins_pd = pd.DataFrame(f_bins_np, index=S.index, columns=["df"])

        eta_np = wave.resource.surface_elevation(
            S, self.t, frequency_bins=f_bins_np, seed=1
        )
        eta_pd = wave.resource.surface_elevation(
            S, self.t, frequency_bins=f_bins_pd, seed=1
        )

        assert_frame_equal(eta0, eta_np)
        assert_frame_equal(eta_np, eta_pd)

    def test_surface_elevation_moments(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        eta = wave.resource.surface_elevation(S, self.t, seed=1)
        dt = self.t[1] - self.t[0]
        Sn = wave.resource.elevation_spectrum(
            eta, 1 / dt, len(eta.values), detrend=False, window="boxcar", noverlap=0
        )

        m0 = wave.resource.frequency_moment(S, 0).item()
        m0n = wave.resource.frequency_moment(Sn, 0).item()
        errorm0 = np.abs((m0 - m0n) / m0)

        self.assertLess(errorm0, 0.01)

        m1 = wave.resource.frequency_moment(S, 1).item()
        m1n = wave.resource.frequency_moment(Sn, 1).item()
        errorm1 = np.abs((m1 - m1n) / m1)

        self.assertLess(errorm1, 0.01)

    def test_surface_elevation_rmse(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        eta = wave.resource.surface_elevation(S, self.t, seed=1)
        dt = self.t[1] - self.t[0]
        Sn = wave.resource.elevation_spectrum(
            eta, 1 / dt, len(eta), detrend=False, window="boxcar", noverlap=0
        )

        fSn = interp1d(Sn.index.values, Sn.values, axis=0)
        Sn_interp = fSn(S.index.values).squeeze()
        rmse = (S.values.squeeze() - Sn_interp) ** 2
        rmse_sum = (np.sum(rmse) / len(rmse)) ** 0.5

        self.assertLess(rmse_sum, 0.02)

    def test_elevation_spectrum_multiple_variables(self):
        time = np.linspace(0, 100, 1000)
        eta1 = np.sin(2 * np.pi * 0.1 * time)
        eta2 = np.sin(2 * np.pi * 0.2 * time)
        eta3 = np.sin(2 * np.pi * 0.3 * time)

        eta_dataset = xr.Dataset(
            {
                "eta1": (["time"], eta1),
                "eta2": (["time"], eta2),
                "eta3": (["time"], eta3),
            },
            coords={"time": time},
        )

        sample_rate = 10
        nnft = 256

        spectra = wave.resource.elevation_spectrum(eta_dataset, sample_rate, nnft)

        # For each variable, find the frequency at which the spectrum has its maximum value
        for var_name, expected_peak_freq in [
            ("eta1", 0.117),
            ("eta2", 0.2),
            ("eta3", 0.3125),
        ]:
            spec_values = spectra[var_name].values
            peak_index = np.argmax(spec_values)
            peak_freq = spectra.index[peak_index]

            # Assert that the peak frequency is close to the expected frequency
            self.assertAlmostEqual(peak_freq, expected_peak_freq, places=2)

    def test_mhkit_spectrum_without_frequency_index_name_defined(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        S.index.name = None

        eta_ifft_xr = wave.resource.surface_elevation(S, self.t, seed=1, method="ifft")
        eta_sos_xr = wave.resource.surface_elevation(
            S, self.t, seed=1, method="sum_of_sines"
        )

        assert_allclose(eta_ifft_xr, eta_sos_xr)

    def test_user_spectrum_without_frequency_index_name_defined(self):
        spectra = pd.DataFrame({"magnitude": [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        time = [0.0, 1.0, 2.0]

        result = wave.resource.surface_elevation(spectra, time, seed=1)

        expected_magnitude = [-0.983917, 1.274248, -2.129812]

        assert_allclose(result.values[:, 0], expected_magnitude, atol=1e-6)

    def test_ifft_sum_of_sines(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)

        eta_ifft = wave.resource.surface_elevation(S, self.t, seed=1, method="ifft")
        eta_sos = wave.resource.surface_elevation(
            S, self.t, seed=1, method="sum_of_sines"
        )

        assert_allclose(eta_ifft, eta_sos)

    def test_surface_elevation_uses_sum_of_sines_when_input_frequency_index_does_not_have_zero(
        self,
    ):
        f = np.linspace(1 / 30, 1 / 2, 32)
        S = wave.resource.jonswap_spectrum(f, self.Tp, self.Hs)

        eta_default = wave.resource.surface_elevation(S, self.t, seed=1)
        eta_sos = wave.resource.surface_elevation(
            S, self.t, seed=1, method="sum_of_sines"
        )

        assert_allclose(eta_default, eta_sos)

    def test_surface_elevation_warn_user_if_zero_frequency_not_defined_and_using_ifft(
        self,
    ):
        f = np.linspace(1 / 30, 1 / 2, 32)
        S = wave.resource.jonswap_spectrum(f, self.Tp, self.Hs)

        with pytest.warns(UserWarning):
            wave.resource.surface_elevation(S, self.t, seed=1, method="ifft")

    def test_surface_elevation_uses_ifft_when_input_frequency_index_has_zero(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)

        eta_default = wave.resource.surface_elevation(S, self.t, seed=1)
        eta_ifft = wave.resource.surface_elevation(S, self.t, seed=1, method="ifft")

        assert_allclose(eta_default, eta_ifft)

    def test_plot_spectrum(self):
        filename = abspath(join(plotdir, "wave_plot_spectrum.png"))
        if isfile(filename):
            os.remove(filename)

        S = wave.resource.pierson_moskowitz_spectrum(self.f, self.Tp, self.Hs)

        plt.figure()
        wave.graphics.plot_spectrum(S)
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_chakrabarti(self):
        filename = abspath(join(plotdir, "wave_plot_chakrabarti.png"))
        if isfile(filename):
            os.remove(filename)

        D = 5
        H = 10
        lambda_w = 200

        wave.graphics.plot_chakrabarti(H, lambda_w, D)
        plt.savefig(filename)

    def test_plot_chakrabarti_np(self):
        filename = abspath(join(plotdir, "wave_plot_chakrabarti_np.png"))
        if isfile(filename):
            os.remove(filename)

        D = np.linspace(5, 15, 5)
        H = 10 * np.ones_like(D)
        lambda_w = 200 * np.ones_like(D)

        wave.graphics.plot_chakrabarti(H, lambda_w, D)
        plt.savefig(filename)

        self.assertTrue(isfile(filename))

    def test_plot_chakrabarti_pd(self):
        filename = abspath(join(plotdir, "wave_plot_chakrabarti_pd.png"))
        if isfile(filename):
            os.remove(filename)

        D = np.linspace(5, 15, 5)
        H = 10 * np.ones_like(D)
        lambda_w = 200 * np.ones_like(D)
        df = pd.DataFrame(
            [H.flatten(), lambda_w.flatten(), D.flatten()], index=["H", "lambda_w", "D"]
        ).transpose()

        wave.graphics.plot_chakrabarti(df.H, df.lambda_w, df.D)
        plt.savefig(filename)

        self.assertTrue(isfile(filename))


if __name__ == "__main__":
    unittest.main()
