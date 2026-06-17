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
                [63.707756, 73.40367, 71.83326, 74.295204, 82.49378],
                [61.048367, 68.54169, 63.2604, 71.18353, 82.87061],
                [67.130325, 69.02487, 69.4047, 74.431915, 83.363075],
                [66.81117, 68.42907, 65.77348, 71.172745, 75.995766],
                [66.638435, 73.6602, 65.55014, 72.049995, 79.03914],
            ]
        )
        cd_spl10_freq_tail = np.array(
            [15848.931925, 19952.62315, 25118.864315, 31622.776602, 39810.717055]
        )
        cd_spl10_tail = np.array(
            [
                [82.50739, 80.50211, 80.871185, 83.187004, 82.15865],
                [83.363846, 81.93087, 81.5197, 83.476814, 82.54627],
                [83.13403, 81.25866, 81.41169, 83.52817, 82.50404],
                [83.35105, 81.70449, 81.42432, 83.45459, 82.16487],
                [83.0968, 80.904686, 81.39755, 83.36778, 82.327],
            ]
        )
        cd_spl3_freq_head = np.array([10.079368, 12.699208, 16.0, 20.158737, 25.398417])
        cd_spl3_head = np.array(
            [
                [63.88229, 73.85718, 71.21643, 75.30319, 82.05813],
                [61.557175, 68.491905, 63.157738, 73.00873, 83.248314],
                [67.5524, 68.75996, 69.92572, 74.80283, 84.11074],
                [67.03263, 68.281166, 66.243065, 72.29833, 77.29135],
                [67.13636, 73.57136, 65.761345, 73.54653, 79.427895],
            ]
        )
        cd_spl3_freq_tail = np.array(
            [16384.0, 20642.546481, 26007.978835, 32768.0, 41285.092963]
        )
        cd_spl3_tail = np.array(
            [
                [82.13845, 80.43939, 81.25442, 83.51936, 80.834045],
                [83.07329, 81.89704, 81.748245, 83.87166, 81.16063],
                [82.843, 81.24031, 81.632744, 83.92745, 81.09888],
                [83.10067, 81.60538, 81.68111, 83.73491, 80.842834],
                [82.74249, 80.793686, 81.695526, 83.699776, 81.01264],
            ]
        )

        np.testing.assert_allclose(
            td_spl.head().values, cd_spl_head, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl.tail().values, cd_spl_tail, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            td_spl["time"].head().astype("int64"), cc.astype("int64"), atol=1
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
