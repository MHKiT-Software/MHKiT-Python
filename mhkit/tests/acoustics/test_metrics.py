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
            P, P.fs, bin_length=60, rms=True
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
        cd_spl_head = np.array([98.12284, 98.639824, 97.62718, 97.85709, 96.98539])
        cd_spl_tail = np.array([98.420975, 98.10879, 97.430115, 97.99395, 97.95798])

        cd_spl10_freq_head = np.array(
            [10.0, 12.589254, 15.848932, 19.952623, 25.118864]
        )
        cd_spl10_head = np.array(
            [
                [68.88561, 75.65294, 68.29522, 75.80323, 82.53724],
                [62.806908, 69.76993, 62.64113, 73.26091, 83.27883],
                [71.73166, 68.541534, 68.056076, 75.438034, 84.268715],
                [70.84345, 68.65471, 63.4681, 72.818085, 77.38771],
                [69.23148, 74.04387, 64.49707, 74.146164, 79.52727],
            ]
        )
        cd_spl10_freq_tail = np.array(
            [19952.62315, 25118.864315, 31622.776602, 39810.717055, 50118.723363]
        )
        cd_spl10_tail = np.array(
            [
                [80.50317, 80.87118, 83.18715, 81.44459, 73.96579],
                [81.933586, 81.51899, 83.47768, 81.85002, 74.25242],
                [81.261314, 81.41166, 83.528534, 81.81753, 74.15244],
                [81.70521, 81.42419, 83.45481, 81.4712, 73.85561],
                [80.90549, 81.397545, 83.36795, 81.5738, 74.3497],
            ]
        )
        cd_spl3_freq_head = np.array([10.0, 12.59921, 15.874011, 20.0, 25.198421])
        cd_spl3_head = np.array(
            [
                [68.88561, 75.65294, 68.29522, 75.80323, 82.53724],
                [62.806908, 69.76993, 62.64113, 73.26091, 83.27883],
                [71.73166, 68.541534, 68.056076, 75.438034, 84.268715],
                [70.84345, 68.65471, 63.4681, 72.818085, 77.38771],
                [69.23148, 74.04387, 64.49707, 74.146164, 79.52727],
            ]
        )
        cd_spl3_freq_tail = np.array(
            [20480.0, 25803.183102, 32509.973544, 40960.0, 51606.366204]
        )
        cd_spl3_tail = np.array(
            [
                [80.37833, 81.21788, 83.5725, 80.37073, 72.06452],
                [81.848434, 81.772064, 83.928505, 80.70311, 72.164345],
                [81.13474, 81.67803, 83.96902, 80.6636, 72.07929],
                [81.532005, 81.694954, 83.796875, 80.38368, 71.94872],
                [80.70353, 81.6905, 83.76083, 80.53248, 72.248276],
            ]
        )

        np.testing.assert_allclose(td_spl.head().values, cd_spl_head, atol=1e-6)
        np.testing.assert_allclose(td_spl.tail().values, cd_spl_tail, atol=1e-6)
        np.testing.assert_allclose(
            td_spl10["freq_bins"].head().values, cd_spl10_freq_head, atol=1e-6
        )
        np.testing.assert_allclose(td_spl10.head().values, cd_spl10_head, atol=1e-6)
        np.testing.assert_allclose(
            td_spl10["freq_bins"].tail().values, cd_spl10_freq_tail, atol=1e-6
        )
        np.testing.assert_allclose(td_spl10.tail().values, cd_spl10_tail, atol=1e-6)
        np.testing.assert_allclose(
            td_spl3["freq_bins"].head().values, cd_spl3_freq_head, atol=1e-6
        )
        np.testing.assert_allclose(td_spl3.head().values, cd_spl3_head, atol=1e-6)
        np.testing.assert_allclose(
            td_spl3["freq_bins"].tail().values, cd_spl3_freq_tail, atol=1e-6
        )
        np.testing.assert_allclose(td_spl3.tail().values, cd_spl3_tail, atol=1e-6)
        np.testing.assert_allclose(
            td_spl["time"].head().astype("int64"), cc.astype("int64"), atol=1
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
                "2023-02-04T15:05:37.999295949",
                "2023-02-04T15:06:37.997894048",
                "2023-02-04T15:07:37.996495485",
                "2023-02-04T15:08:37.995094776",
                "2023-02-04T15:09:37.993695497",
            ],
            dtype="datetime64[ns]",
        )
        cd_sel = np.array([116.18274, 121.698654, 143.28117, 147.37479, 127.01828])
        cd_sel_lf = np.array([112.363075, 120.177086, 142.74931, 146.57983, 125.83696])
        cd_sel_hf = np.array([112.22166, 118.88085, 139.94121, 144.33324, 124.328995])
        cd_sel_vhf = np.array([110.23136, 114.00643, 133.20006, 139.13504, 118.88397])
        cd_sel_pw = np.array([112.22191, 119.87286, 141.67467, 145.6534, 125.419975])
        cd_sel_ow = np.array([110.945404, 118.06397, 139.08435, 143.51094, 123.68077])

        np.testing.assert_allclose(td_sel.values, cd_sel, atol=1e-5)
        np.testing.assert_allclose(td_sel_lf.values, cd_sel_lf, atol=1e-5)
        np.testing.assert_allclose(td_sel_hf.values, cd_sel_hf, atol=1e-5)
        np.testing.assert_allclose(td_sel_vhf.values, cd_sel_vhf, atol=1e-5)
        np.testing.assert_allclose(td_sel_pw.values, cd_sel_pw, atol=1e-5)
        np.testing.assert_allclose(td_sel_ow.values, cd_sel_ow, atol=1e-5)
        np.testing.assert_allclose(
            td_sel["time"].astype("int64"), cc.astype("int64"), atol=1
        )

    def test_spl_vs_sel(self):
        # SPL should equal SEL over a 1 second interval
        td_spl = acoustics.sound_pressure_level(self.spsd, fmin=10, fmax=100000)
        td_sel = acoustics.sound_exposure_level(self.spsd, fmin=10, fmax=100000)

        np.testing.assert_allclose(td_spl.values, td_sel.values, atol=1e-6)
