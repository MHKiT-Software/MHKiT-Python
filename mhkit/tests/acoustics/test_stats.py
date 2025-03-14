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


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        file_name = join(datadir, "6247.230204150508.wav")
        P = acoustics.io.read_soundtrap(file_name, sensitivity=-177)
        self.spsd = acoustics.sound_pressure_spectral_density(P, P.fs, bin_length=1)
        self.spsd_60s = acoustics.sound_pressure_spectral_density(
            P, P.fs, bin_length=60, rms=False
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
            [10.0, 10.717735, 11.486984, 12.311444, 13.195079]
        )
        cd_spl10_head = np.array(
            [
                [64.40389, 64.78221, 63.64469, 67.782845, 73.05421],
                [56.934277, 62.80422, 66.329056, 67.336395, 65.79995],
                [67.16593, 69.089096, 69.5835, 67.69844, 63.657196],
                [66.010025, 67.567635, 68.16686, 66.72678, 64.52246],
                [63.887203, 68.73698, 72.71424, 72.98322, 69.08172],
            ]
        )
        cd_spl10_freq_tail = np.array(
            [38217.031333, 40960.0, 43899.841025, 47050.684621, 50427.675171]
        )
        cd_spl10_tail = np.array(
            [
                [77.38338, 73.43317, 72.7755, 72.53339, np.nan],
                [77.72596, 73.57787, 73.16561, 72.637436, np.nan],
                [77.61121, 73.59171, 73.20265, 72.57601, np.nan],
                [77.3753, 73.35718, 72.89339, 72.386765, np.nan],
                [77.31649, 73.806496, 73.296, 72.73348, np.nan],
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
        np.testing.assert_equal(td_spl["time"].head().values, cc)

    def test_nmfs_weighting(self):
        freq = self.spsd["freq"]
        slc = slice(20, 25)  # test 20 - 25 Hz

        W_LF, E_LF = acoustics.nmfs_auditory_weighting(freq, group="LF")
        W_HF, E_HF = acoustics.nmfs_auditory_weighting(freq, group="HF")
        W_VHF, E_VHF = acoustics.nmfs_auditory_weighting(freq, group="VHF")
        W_PW, E_PW = acoustics.nmfs_auditory_weighting(freq, group="PW")
        W_OW, E_OW = acoustics.nmfs_auditory_weighting(freq, group="OW")

        cd_W_LF, cd_E_LF = np.array(
            [-9.610503, -10.399795, -11.197884, -12.002824, -12.812856, -13.6263685]
        ), np.array([186.7305, 187.51979, 188.31789, 189.12282, 189.93286, 190.74637])
        cd_W_HF, cd_E_HF = np.array(
            [-0.24596292, -0.29349536, -0.3440557, -0.39752078, -0.45378864, -0.5127737]
        ), np.array([181.56596, 181.6135, 181.66406, 181.71751, 181.77379, 181.83278])
        cd_W_VHF, cd_E_VHF = np.array(
            [-0.1556686, -0.1080610, -0.0709422, -0.0428147, -0.0224707, -0.0089151]
        ), np.array([161.06567, 161.01807, 160.98094, 160.95282, 160.93246, 160.91891])
        cd_W_PW, cd_E_PW = np.array(
            [-1.508032, -1.6820252, -1.8632004, -2.0513422, -2.2462401, -2.447689]
        ), np.array([176.79803, 176.97203, 177.1532, 177.34134, 177.53624, 177.73769])
        cd_W_OW, cd_E_OW = np.array(
            [-2.8513184, -3.2219172, -3.60537, -4.000722, -4.4070754, -4.823583]
        ), np.array([182.22131, 182.59192, 182.97537, 183.37073, 183.77707, 184.19359])

        np.testing.assert_allclose(W_LF.sel(freq=slc).values, cd_W_LF, atol=1e-6)
        np.testing.assert_allclose(W_HF.sel(freq=slc).values, cd_W_HF, atol=1e-6)
        np.testing.assert_allclose(W_VHF.sel(freq=slc).values, cd_W_VHF, atol=1e-6)
        np.testing.assert_allclose(W_PW.sel(freq=slc).values, cd_W_PW, atol=1e-6)
        np.testing.assert_allclose(W_OW.sel(freq=slc).values, cd_W_OW, atol=1e-6)

        np.testing.assert_allclose(E_LF.sel(freq=slc).values, cd_E_LF, atol=1e-6)
        np.testing.assert_allclose(E_HF.sel(freq=slc).values, cd_E_HF, atol=1e-6)
        np.testing.assert_allclose(E_VHF.sel(freq=slc).values, cd_E_VHF, atol=1e-6)
        np.testing.assert_allclose(E_PW.sel(freq=slc).values, cd_E_PW, atol=1e-6)
        np.testing.assert_allclose(E_OW.sel(freq=slc).values, cd_E_OW, atol=1e-6)

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
        cd_sel = np.array([98.2998, 102.6271, 125.41366, 125.71297, 104.94424])
        cd_sel_lf = np.array([74.089165, 74.528076, 73.478775, 76.782104, 76.000015])
        cd_sel_hf = np.array([90.085915, 90.23294, 94.21187, 95.70856, 90.99371])
        cd_sel_vhf = np.array([92.27235, 92.464096, 100.08952, 101.10707, 93.23433])
        cd_sel_pw = np.array([85.331696, 85.516174, 86.10397, 87.81647, 86.33542])
        cd_sel_ow = np.array([81.784424, 82.04712, 81.639206, 83.623116, 83.08575])

        np.testing.assert_allclose(td_sel.values, cd_sel, atol=1e-6)
        np.testing.assert_allclose(td_sel_lf.values, cd_sel_lf, atol=1e-6)
        np.testing.assert_allclose(td_sel_hf.values, cd_sel_hf, atol=1e-6)
        np.testing.assert_allclose(td_sel_vhf.values, cd_sel_vhf, atol=1e-6)
        np.testing.assert_allclose(td_sel_pw.values, cd_sel_pw, atol=1e-6)
        np.testing.assert_allclose(td_sel_ow.values, cd_sel_ow, atol=1e-6)
        np.testing.assert_equal(td_sel["time"].values, cc)
