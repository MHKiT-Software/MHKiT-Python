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
        td_spsdl_mean = acoustics.band_aggregate(td_spsdl, octave, fmin=10, fmax=100000)

        # Time average into 30 s bins
        lbin = 30
        td_spsdl_50 = acoustics.time_aggregate(td_spsdl_mean, lbin, method="median")
        td_spsdl_25 = acoustics.time_aggregate(
            td_spsdl_mean, lbin, method={"quantile": 0.25}
        )
        td_spsdl_75 = acoustics.time_aggregate(
            td_spsdl_mean, lbin, method={"quantile": 0.75}
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
                [63.45507, 64.753525, 65.04905, 67.15576, 73.47938],
                [62.77437, 64.58199, 65.18464, 66.37395, 72.30796],
                [64.76277, 64.950264, 65.80557, 67.88482, 73.24013],
                [63.654488, 62.31394, 65.598816, 67.370674, 71.52472],
                [62.45623, 62.461388, 62.111694, 66.06419, 72.324936],
            ]
        )
        cd_spsdl_25 = np.array(
            [
                [59.33189297, 62.89503765, 61.60455799, 64.80938911, 70.59576607],
                [60.37440872, 60.69928551, 61.9694643, 64.91986465, 70.00148964],
                [61.1297617, 63.02504444, 64.41207123, 66.37802315, 71.38513947],
                [59.52737236, 59.45869541, 62.48176765, 66.0959053, 70.06054497],
                [58.55439758, 59.88098335, 59.66310596, 63.86431885, 70.20335197],
            ]
        )
        cd_spsdl_75 = np.array(
            [
                [66.33672714, 67.13593102, 67.34234238, 68.7525959, 75.30982399],
                [64.58539009, 66.84792709, 67.11526108, 69.7322197, 74.50746346],
                [66.56425095, 67.85562325, 69.30602646, 69.83069992, 74.79984283],
                [67.34252357, 65.65701294, 67.48604202, 70.948246, 73.59340286],
                [66.26214409, 65.43437958, 64.36196518, 67.67719078, 74.33639717],
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

    def test_sound_pressure_spectral_density(self):
        """
        Test sound pressure spectral density calculation.
        """
        # Create some sample pressure data
        time = np.arange(0, 10, 0.1)
        data = np.sin(time)
        pressure = xr.DataArray(
            data, coords=[time], dims=["time"], attrs={"units": "Pa", "fs": 100}
        )

        # Adjust bin size to get multiple segments
        fs = 100
        bin_length = 0.1  # seconds
        win_samples = int(bin_length * fs)

        # Run the spectral density function
        spsd = acoustics.sound_pressure_spectral_density(
            pressure, fs=fs, bin_length=bin_length
        )

        # Assert that output is an xarray DataArray with expected dimensions
        self.assertIsInstance(spsd, xr.DataArray)
        self.assertIn("freq", spsd.dims)
        self.assertIn("time", spsd.dims)
        self.assertEqual(spsd.attrs["units"], "Pa^2/Hz")
        self.assertEqual(spsd.attrs["nbin"], f"{bin_length} s")

        # Calculate expected number of segments
        overlap = 0.0
        step = int(win_samples * (1 - overlap))
        expected_segments = (len(pressure) - win_samples) // step + 1

        # Calculate expected number of segments without overlap
        expected_segments = len(pressure) // win_samples
        self.assertEqual(spsd.shape[0], expected_segments)

    def test_apply_calibration(self):
        """
        Test the application of calibration curves to spectral density.
        """
        # Create a sample SPSD (Spectral Pressure Spectral Density) and calibration curve
        time = np.arange(0, 10, 0.1)
        freq = np.linspace(10, 1000, len(time))
        spsd_data = np.random.random((len(time), len(freq)))
        spsd = xr.DataArray(
            spsd_data,
            coords=[time, freq],
            dims=["time", "freq"],
            attrs={"units": "V^2/Hz"},
        )

        sensitivity_curve = xr.DataArray(
            np.random.random(len(freq)), coords=[freq], dims=["freq"]
        )
        fill_value = 0.0

        # Apply calibration
        calibrated_spsd = acoustics.apply_calibration(
            spsd, sensitivity_curve, fill_value
        )

        # Assert that the calibration returns the correct data format and values
        self.assertIsInstance(calibrated_spsd, xr.DataArray)
        self.assertEqual(calibrated_spsd.attrs["units"], "Pa^2/Hz")
        self.assertEqual(
            calibrated_spsd.shape, spsd.shape
        )  # Ensure shape remains the same
        np.testing.assert_array_less(
            calibrated_spsd.values, spsd.values
        )  # Calibration should reduce values

    def test_fmax_warning(self):
        """
        Test that fmax warning adjusts the maximum frequency if necessary.
        """
        from mhkit.acoustics.analysis import _fmax_warning

        # Test case where fmax is greater than Nyquist frequency
        fn = 1000
        fmax = 1500  # Greater than Nyquist frequency
        adjusted_fmax = _fmax_warning(fn, fmax)
        self.assertEqual(adjusted_fmax, fn)  # Should return the Nyquist frequency

        # Test case where fmax is within limits
        fmax = 500
        adjusted_fmax = _fmax_warning(fn, fmax)
        self.assertEqual(adjusted_fmax, fmax)  # Should return the original fmax

        # Test with incorrect types
        with self.assertRaises(TypeError):
            _fmax_warning("not a number", fmax)
        with self.assertRaises(TypeError):
            _fmax_warning(fn, "not a number")

    def test_validate_method(self):
        """
        Test the validation of the 'method' parameter in band_aggregate or time_aggregate.
        """
        from mhkit.acoustics.analysis import _validate_method

        # Valid method string
        method_name, method_arg = _validate_method("median")
        self.assertEqual(method_name, "median")
        self.assertIsNone(method_arg)

        # Valid method dictionary
        method_name, method_arg = _validate_method({"quantile": 0.25})
        self.assertEqual(method_name, "quantile")
        self.assertEqual(method_arg, 0.25)

        # Invalid method string
        with self.assertRaises(ValueError):
            _validate_method("unsupported_method")

        # Invalid method dictionary
        with self.assertRaises(ValueError):
            _validate_method({"unsupported_method": None})

        # Invalid quantile value in dictionary
        with self.assertRaises(ValueError):
            _validate_method({"quantile": 1.5})  # Out of valid range (0,1)


if __name__ == "__main__":
    unittest.main()
