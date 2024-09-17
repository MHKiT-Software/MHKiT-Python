import os
from os.path import abspath, dirname, join, normpath
import numpy as np
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
        self.spsd = acoustics.sound_pressure_spectral_density(P, P.fs, window=1)

    @classmethod
    def tearDownClass(self):
        pass

    def test_spsdl(self):
        td_spsdl = acoustics.sound_pressure_spectral_density_level(self.spsd)

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
        cd_spsdl = np.array(
            [
                [61.72558153, 60.45878138, 61.02543806, 62.10487326, 53.69452342],
                [64.73788935, 63.7154788, 56.60306848, 55.59145693, 65.14298631],
                [54.88840931, 64.81213715, 68.5464288, 66.96210531, 57.26933701],
                [47.83166387, 46.34269439, 55.26689475, 59.97537222, 62.87564412],
                [51.84125861, 58.33037915, 56.42519674, 55.83574275, 55.48694318],
            ]
        )

        np.testing.assert_allclose(td_spsdl.head().values, cd_spsdl, atol=1e-6)
        np.testing.assert_equal(td_spsdl["time"].head().values, cc)

    def test_averaging(self):
        td_spsdl = acoustics.sound_pressure_spectral_density_level(self.spsd)

        # Frequency average into # octave bands
        octave = 3
        td_spsdl_mean = acoustics.band_average(td_spsdl, octave, fmin=50)

        # Time average into 30 s windows
        window = 30
        td_spsdl_50 = acoustics.time_average(td_spsdl_mean, window, method="median")
        td_spsdl_25 = acoustics.time_average(
            td_spsdl_mean, window, method="quantile", method_arg=0.25
        )
        td_spsdl_75 = acoustics.time_average(
            td_spsdl_mean, window, method="quantile", method_arg=0.75
        )

        cc = np.array(
            [
                "2023-02-04T15:05:23.499983310",
                "2023-02-04T15:05:53.499983310",
                "2023-02-04T15:06:23.499983310",
                "2023-02-04T15:06:53.499983310",
                "2023-02-04T15:07:23.499983310",
            ],
            dtype="datetime64[ns]",
        )
        cd_spsdl_50 = np.array(
            [
                [73.71803613, 70.97557445, 69.79906778, 69.04934313, 67.56449352],
                [73.72245955, 71.53327285, 70.55206775, 68.69638127, 67.75243522],
                [73.64022645, 72.24548986, 70.09995522, 69.00394292, 68.22919418],
                [73.1301846, 71.99940268, 70.56372046, 69.01366589, 67.19515351],
                [74.67880072, 71.27235403, 70.23024477, 67.4915765, 66.73024553],
            ]
        )
        cd_spsdl_25 = np.array(
            [
                [72.42136105, 70.37422873, 68.60783404, 67.56108417, 66.4751517],
                [71.95173902, 71.03281659, 69.59019407, 67.79615712, 66.73980611],
                [71.12756436, 70.68228634, 69.53891917, 68.126758, 67.48463198],
                [71.71909635, 70.1849931, 69.22647784, 68.14102709, 66.18740693],
                [72.25521793, 70.18087912, 68.97354823, 66.71295946, 65.35302077],
            ]
        )
        cd_spsdl_75 = np.array(
            [
                [75.29614796, 71.86901413, 71.08418954, 69.6835928, 68.26993291],
                [74.51608597, 72.82376854, 71.31219865, 70.38580566, 69.01731822],
                [75.17013043, 73.45962974, 71.30593827, 71.50687178, 69.49805535],
                [74.38176106, 73.13456376, 72.13861655, 70.45825381, 67.93458589],
                [75.52387419, 72.99604074, 71.26831962, 68.90629303, 67.79114848],
            ]
        )

        np.testing.assert_allclose(td_spsdl_50.head().values, cd_spsdl_50, atol=1e-6)
        np.testing.assert_allclose(td_spsdl_25.head().values, cd_spsdl_25, atol=1e-6)
        np.testing.assert_allclose(td_spsdl_75.head().values, cd_spsdl_75, atol=1e-6)
        np.testing.assert_equal(td_spsdl_50["time_bins"].head().values, cc)

    def test_freq_loss(self):
        # Test min frequency
        fmin = acoustics.minimum_frequency(water_depth=20, c=1500, c_seabed=1700)
        self.assertEqual(fmin, 39.84375)

    def test_spl(self):
        td_spl = acoustics.sound_pressure_level(self.spsd, fmin=50)

        # Decidecade octave sound pressure level
        td_spl10 = acoustics.decidecade_sound_pressure_level(self.spsd, fmin=50)

        # Median third octave sound pressure level
        td_spl3 = acoustics.third_octave_sound_pressure_level(self.spsd, fmin=50)

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
        cd_spl = np.array(
            [97.48727775, 98.21888437, 96.99586637, 97.43571891, 96.60915502]
        )
        cd_spl10 = np.array(
            [
                [82.06503071, 78.20349846, 79.78088446, 75.31281183, 82.1194826],
                [82.66175023, 79.77804574, 82.86005403, 77.57078269, 76.7598224],
                [77.48975416, 82.72580274, 83.88251531, 74.71242694, 74.01377947],
                [79.11312683, 76.56114947, 82.18953494, 75.40888015, 74.80285354],
                [81.26751434, 82.29074565, 80.08831394, 75.75364773, 73.52176641],
            ]
        )
        cd_spl3 = np.array(
            [
                [86.5847236, 84.98068691, 85.61056131, 83.55067796, 84.41810962],
                [87.5449842, 84.48841036, 84.09406069, 85.81895309, 86.71437852],
                [86.37334939, 84.08914125, 86.01614536, 83.36059983, 84.54635288],
                [84.21413445, 84.63996392, 82.52906024, 84.54731095, 83.45652422],
                [86.90033232, 84.8217658, 83.85297355, 82.92231618, 81.39163217],
            ]
        )

        np.testing.assert_allclose(td_spl.head().values, cd_spl, atol=1e-6)
        np.testing.assert_allclose(td_spl10.head().values, cd_spl10, atol=1e-6)
        np.testing.assert_allclose(td_spl3.head().values, cd_spl3, atol=1e-6)
        np.testing.assert_equal(td_spl["time"].head().values, cc)


if __name__ == "__main__":
    unittest.main()
