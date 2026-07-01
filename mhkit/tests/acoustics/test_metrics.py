import os
from os.path import abspath, dirname, join, normpath
import numpy as np
import xarray as xr
import unittest

import mhkit.acoustics as acoustics

testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "acoustics"))


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        file_name = join(datadir, "6247.230204150508.wav")
        P = acoustics.io.read_soundtrap(file_name, sensitivity=-177)
        self.spsd = acoustics.sound_pressure_spectral_density(P, P.fs, bin_length=1)
        self.spsd_60s = acoustics.sound_pressure_spectral_density(
            P, P.fs, bin_length=60
        )

    @classmethod
    def tearDownClass(self):
        pass

    def test_spl(self):
        td_spl = acoustics.sound_pressure_level(self.spsd, fmin=10, fmax=100000)

        # Decidecade octave sound pressure level
        td_spl10 = acoustics.decidecade_sound_pressure_level(
            self.spsd, fmin=10, fmax=100000
        )

        # Median third octave sound pressure level
        td_spl3 = acoustics.third_octave_sound_pressure_level(
            self.spsd, fmin=10, fmax=100000
        )

        cc = np.array(
            [
                "2023-02-04T15:05:08.499983310",
                "2023-02-04T15:05:09.499959707",
                "2023-02-04T15:05:10.499936580",
                "2023-02-04T15:05:11.499913454",
                "2023-02-04T15:05:12.499890089",
            ],
            dtype="datetime64[ns]",
        )
        cd_spl_head = np.array([98.38563, 98.20744, 98.40893, 98.760826, 97.21689])
        cd_spl_tail = np.array([97.29747, 97.51031, 98.302605, 98.08264, 97.961624])

        cd_spl10_freq_head = np.array(
            [10.0, 12.589254, 15.848932, 19.952623, 25.118864]
        )
        cd_spl10_head = np.array(
            [
                [63.97053, 73.666466, 72.09614, 74.55805, 82.75655],
                [62.93203, 66.638306, 70.98349, 77.45272, 81.752464],
                [60.817398, 68.31071, 63.029625, 70.95255, 82.63971],
                [62.09812, 64.15272, 69.552284, 68.90529, 80.36065],
                [66.71991, 68.61455, 68.994385, 74.02169, 82.95276],
            ]
        )
        cd_spl10_freq_tail = np.array(
            [15848.931925, 19952.62315, 25118.864315, 31622.776602, 39810.717055]
        )
        cd_spl10_tail = np.array(
            [
                [83.001396, 81.12602, 81.27904, 83.39552, 82.37139],
                [83.49641, 81.45716, 81.33536, 83.458916, 82.2894],
                [83.659706, 82.013145, 81.73298, 83.76325, 82.473526],
                [83.58633, 82.00813, 81.56679, 83.64441, 82.30002],
                [83.10043, 80.908325, 81.40121, 83.37144, 82.33061],
            ]
        )
        cd_spl3_freq_head = np.array([10.079368, 12.699208, 16.0, 20.158737, 25.398417])
        cd_spl3_head = np.array(
            [
                [64.145065, 74.11997, 71.47932, 75.56602, 82.32091],
                [63.103104, 66.557434, 71.07715, 77.72599, 81.811295],
                [61.326202, 68.260925, 62.92696, 72.777756, 83.01741],
                [62.4825, 63.912766, 69.58983, 69.38629, 80.71226],
                [67.14199, 68.34964, 69.5154, 74.39259, 83.70042],
            ]
        )
        cd_spl3_freq_tail = np.array(
            [16384.0, 20642.546481, 26007.978835, 32768.0, 41285.092963]
        )
        cd_spl3_tail = np.array(
            [
                [82.71037, 81.107666, 81.50009, 83.79481, 80.966225],
                [83.16837, 81.403625, 81.59755, 83.72969, 80.989555],
                [83.40933, 81.91403, 81.98976, 84.04357, 81.151474],
                [83.374664, 81.88523, 81.79578, 83.965515, 80.91496],
                [82.746124, 80.797325, 81.69917, 83.70343, 81.01623],
            ]
        )

        np.testing.assert_allclose(
            td_spl.head().values, cd_spl_head, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl.tail().values, cd_spl_tail, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl["time_psd"].head().astype("int64"), cc.astype("int64"), atol=1
        )

        np.testing.assert_allclose(
            td_spl10["freq_bins"].head().values, cd_spl10_freq_head, atol=1e-6
        )
        np.testing.assert_allclose(
            td_spl10.head().values, cd_spl10_head, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl10["freq_bins"].tail().values, cd_spl10_freq_tail, atol=1e-6
        )
        np.testing.assert_allclose(
            td_spl10.tail().values, cd_spl10_tail, atol=1e-6, rtol=1e-6
        )

        np.testing.assert_allclose(
            td_spl3["freq_bins"].head().values, cd_spl3_freq_head, atol=1e-6
        )
        np.testing.assert_allclose(
            td_spl3.head().values, cd_spl3_head, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl3["freq_bins"].tail().values, cd_spl3_freq_tail, atol=1e-6
        )
        np.testing.assert_allclose(
            td_spl3.tail().values, cd_spl3_tail, atol=1e-6, rtol=1e-6
        )

    def test_nmfs_weighting(self):
        freq = self.spsd["freq"]
        slc = slice(20, 25)  # test 20 - 25 Hz

        W_LF, E_LF = acoustics.nmfs_auditory_weighting(freq, group="LF")
        W_HF, E_HF = acoustics.nmfs_auditory_weighting(freq, group="HF")
        W_VHF, E_VHF = acoustics.nmfs_auditory_weighting(freq, group="VHF")
        W_PW, E_PW = acoustics.nmfs_auditory_weighting(freq, group="PW")
        W_OW, E_OW = acoustics.nmfs_auditory_weighting(freq, group="OW")

        cd_W_LF, cd_E_LF = np.array(
            [-18.241247, -17.827854, -17.434275, -17.058767, -16.699821, -16.3561]
        ), np.array([195.36125, 194.94786, 194.55428, 194.17877, 193.81982, 193.4761])
        cd_W_HF, cd_E_HF = np.array(
            [-59.7284, -59.071625, -58.44541, -57.847057, -57.274178, -56.724693]
        ), np.array([241.0484, 240.39163, 239.76541, 239.16705, 238.59418, 238.0447])
        cd_W_VHF, cd_E_VHF = np.array(
            [-109.34241, -108.397385, -107.49632, -106.635315, -105.81097, -105.02029]
        ), np.array([270.2524, 269.30737, 268.4063, 267.54532, 266.72098, 265.9303])
        cd_W_PW, cd_E_PW = np.array(
            [-52.117348, -51.427025, -50.768852, -50.13999, -49.537937, -48.96051]
        ), np.array([227.40735, 226.71703, 226.05885, 225.43, 224.82794, 224.25052])
        cd_W_OW, cd_E_OW = np.array(
            [-65.056496, -64.386955, -63.748577, -63.138584, -62.55456, -61.99438]
        ), np.array([244.4265, 243.75696, 243.11858, 242.50858, 241.92456, 241.36438])

        np.testing.assert_allclose(W_LF.sel(freq=slc).values, cd_W_LF, atol=1e-5)
        np.testing.assert_allclose(W_HF.sel(freq=slc).values, cd_W_HF, atol=1e-5)
        np.testing.assert_allclose(W_VHF.sel(freq=slc).values, cd_W_VHF, atol=1e-5)
        np.testing.assert_allclose(W_PW.sel(freq=slc).values, cd_W_PW, atol=1e-5)
        np.testing.assert_allclose(W_OW.sel(freq=slc).values, cd_W_OW, atol=1e-5)

        np.testing.assert_allclose(E_LF.sel(freq=slc).values, cd_E_LF, atol=1e-5)
        np.testing.assert_allclose(E_HF.sel(freq=slc).values, cd_E_HF, atol=1e-5)
        np.testing.assert_allclose(E_VHF.sel(freq=slc).values, cd_E_VHF, atol=1e-5)
        np.testing.assert_allclose(E_PW.sel(freq=slc).values, cd_E_PW, atol=1e-5)
        np.testing.assert_allclose(E_OW.sel(freq=slc).values, cd_E_OW, atol=1e-5)

    def test_sel(self):
        td_sel = acoustics.sound_exposure_level(self.spsd_60s, fmin=10, fmax=100000)
        td_sel_lf = acoustics.sound_exposure_level(
            self.spsd_60s, group="LF", fmin=10, fmax=100000
        )
        td_sel_hf = acoustics.sound_exposure_level(
            self.spsd_60s, group="HF", fmin=10, fmax=100000
        )
        td_sel_vhf = acoustics.sound_exposure_level(
            self.spsd_60s, group="VHF", fmin=10, fmax=100000
        )
        td_sel_pw = acoustics.sound_exposure_level(
            self.spsd_60s, group="PW", fmin=10, fmax=100000
        )
        td_sel_ow = acoustics.sound_exposure_level(
            self.spsd_60s, group="OW", fmin=10, fmax=100000
        )

        cc = np.array(
            [
                "2023-02-04T15:05:37.999294757",
                "2023-02-04T15:06:07.998594760",
                "2023-02-04T15:06:37.997895240",
                "2023-02-04T15:07:07.997195243",
                "2023-02-04T15:07:37.996495246",
            ],
            dtype="datetime64[ns]",
        )
        cd_sel = np.array([116.081825, 117.38323, 120.408775, 127.583755, 143.19513])
        cd_sel_lf = np.array([112.261665, 114.67835, 118.88702, 126.86423, 142.66327])
        cd_sel_hf = np.array([112.121216, 114.001945, 117.59107, 124.69318, 139.85515])
        cd_sel_vhf = np.array([110.1319, 110.81934, 112.71749, 118.208664, 133.114])
        cd_sel_pw = np.array([112.120964, 114.46142, 118.58291, 126.152115, 141.58862])
        cd_sel_ow = np.array([110.844246, 113.00219, 116.77398, 123.92641, 138.99828])

        np.testing.assert_allclose(td_sel.head().values, cd_sel, atol=1e-5)
        np.testing.assert_allclose(td_sel_lf.head().values, cd_sel_lf, atol=1e-5)
        np.testing.assert_allclose(td_sel_hf.head().values, cd_sel_hf, atol=1e-5)
        np.testing.assert_allclose(td_sel_vhf.head().values, cd_sel_vhf, atol=1e-5)
        np.testing.assert_allclose(td_sel_pw.head().values, cd_sel_pw, atol=1e-5)
        np.testing.assert_allclose(td_sel_ow.head().values, cd_sel_ow, atol=1e-5)
        np.testing.assert_allclose(
            td_sel["time_psd"].head().astype("int64"), cc.astype("int64"), atol=1
        )

    def test_spl_vs_sel(self):
        # SPL should equal SEL over a 1 second interval
        td_spl = acoustics.sound_pressure_level(self.spsd, fmin=10, fmax=100000)
        td_sel = acoustics.sound_exposure_level(self.spsd, fmin=10, fmax=100000)

        np.testing.assert_allclose(td_spl.values, td_sel.values, atol=1e-6)
