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

    def test_frequency_bands(self):
        _, third_octaves = acoustics.create_frequency_bands(3, 2, fmin=1, fmax=48000)
        _, decidecades = acoustics.create_frequency_bands(10, 10, fmin=1, fmax=48000)
        millidecades = acoustics.analysis._get_band_table(
            self.spsd["freq"][:460].values,
            base=10,
            bands_per_division=1000,
            use_fft_res_at_bottom=True,
        )[:, 1]

        cd_third_octaves_head = np.array(
            [1.0, 1.25992105, 1.58740105, 2.0, 2.5198421, 3.1748021, 4.0]
        )
        cd_third_octaves_tail = np.array(
            [16384.0, 20642.54648148, 26007.97883545, 32768.0, 41285.09296296]
        )
        cd_decidecades_head = np.array(
            [1.0, 1.25892541, 1.58489319, 1.99526231, 2.51188643, 3.16227766]
        )
        cd_decidecades_tail = np.array(
            [19952.62314969, 25118.8643151, 31622.77660168, 39810.71705535]
        )
        cd_millidecades = np.array(
            [454.0, 455.0, 456.03691595, 457.08818961, 458.14188671, 459.19801284]
        )

        np.testing.assert_allclose(
            third_octaves["center_freq"][:7], cd_third_octaves_head, atol=1e-5
        )
        np.testing.assert_allclose(
            third_octaves["center_freq"][-5:], cd_third_octaves_tail, atol=1e-5
        )
        np.testing.assert_allclose(
            decidecades["center_freq"][:6], cd_decidecades_head, atol=1e-5
        )
        np.testing.assert_allclose(
            decidecades["center_freq"][-4:], cd_decidecades_tail, atol=1e-5
        )
        np.testing.assert_allclose(millidecades[-6:], cd_millidecades, atol=1e-5)

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
        self.assertEqual(spsd.attrs["bin_length"], bin_length)
        self.assertEqual(spsd.attrs["n_fft"], bin_length * fs)

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

    def test_freq_loss(self):
        # Test min frequency
        fmin = acoustics.minimum_frequency(water_depth=20, c=1500, c_seabed=1700)
        self.assertEqual(fmin, 39.84375)

    def test_spsdl(self):
        """
        Test sound pressure spectral density level calculation.
        """
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
        np.testing.assert_allclose(
            td_spsdl["time"].head().astype("int64"), cc.astype("int64"), atol=1
        )

    def test_averaging_deprecated(self):
        td_spsdl = acoustics.sound_pressure_spectral_density_level(self.spsd)

        # Frequency average into # octave bands
        octave = [3, 2]
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

        np.testing.assert_allclose(td_spsdl_50.head().values, cd_spsdl_50, atol=1e-5)
        np.testing.assert_allclose(td_spsdl_25.head().values, cd_spsdl_25, atol=1e-5)
        np.testing.assert_allclose(td_spsdl_75.head().values, cd_spsdl_75, atol=1e-5)
        np.testing.assert_allclose(
            td_spsdl_50["time_bins"].head().astype("int64"), cc.astype("int64"), atol=1
        )

    def test_averaging(self):
        # Third Octaves
        third_octave_spsd = acoustics.convert_to_third_octave(self.spsd)
        td_spsdl_to = acoustics.sound_pressure_spectral_density_level(third_octave_spsd)
        # Decidecades
        decidecade_spsd = acoustics.convert_to_decidecade(self.spsd)
        td_spsdl_dd = acoustics.sound_pressure_spectral_density_level(decidecade_spsd)
        # Millidecades
        millidecade_spsd = acoustics.convert_to_millidecade(self.spsd)
        td_spsdl_md = acoustics.sound_pressure_spectral_density_level(millidecade_spsd)

        # Time average into 30 s bins
        lbin = 30
        td_spsdl_avg = acoustics.time_average(td_spsdl_to, lbin)
        td_spsdl_sum = acoustics.time_summation(td_spsdl_to, lbin)

        cd_spsdl_dd = np.array(
            [
                [65.93822, 62.86854, 60.797012, 62.410725, 60.77913],
                [68.95052, 65.88085, 63.982758, 65.66743, 61.178196],
                [59.101044, 56.03137, 63.76194, 66.764084, 67.23888],
                [52.0443, 48.974625, 46.74791, 48.29464, 53.098785],
                [56.053894, 52.98422, 57.43947, 60.282322, 57.38807],
            ]
        )
        cd_spsdl_md_head = np.array(
            [
                [61.725582, 60.458782, 61.02544, 62.104874, 53.694523],
                [64.73789, 63.71548, 56.60307, 55.591457, 65.14299],
                [54.88841, 64.81213, 68.54643, 66.962105, 57.269337],
                [47.831665, 46.342693, 55.266895, 59.975372, 62.875645],
                [51.84126, 58.33038, 56.425198, 55.835743, 55.486942],
            ]
        )
        cd_spsdl_md_tail = np.array(
            [
                [33.43558, 44.184784, 33.868732, 34.39551, 34.410404],
                [33.27611, 45.362965, 33.23099, 34.436787, 36.571167],
                [33.29176, 45.027763, 34.02678, 35.329945, 36.35523],
                [33.95818, 44.993164, 34.119263, 34.288776, 35.93025],
                [34.140266, 45.39091, 33.197502, 34.189434, 35.96681],
            ]
        )

        cc = np.array(
            [
                "2023-02-04T15:05:23.49998",
                "2023-02-04T15:05:53.49998",
                "2023-02-04T15:06:23.49998",
                "2023-02-04T15:06:53.49998",
                "2023-02-04T15:07:23.49998",
            ],
            dtype="datetime64[ns]",
        )
        cd_spsdl_avg = np.array(
            [
                [62.4551, 59.374382, 58.97122, 61.07321, 59.783104],
                [61.14534, 58.064617, 57.62369, 59.71631, 59.29018],
                [62.43891, 59.358192, 59.07586, 61.207146, 60.29722],
                [61.163094, 58.08237, 59.036198, 61.416603, 60.38328],
                [62.242744, 59.162025, 57.95079, 59.82794, 58.536304],
            ]
        )
        cd_spsdl_sum = np.array(
            [
                [77.22632, 74.14559, 73.74243, 75.84442, 74.55432],
                [75.91655, 72.83583, 72.394905, 74.487526, 74.06139],
                [77.21013, 74.1294, 73.84707, 75.97836, 75.06843],
                [75.9343, 72.853584, 73.80741, 76.18781, 75.154495],
                [77.01396, 73.93324, 72.72201, 74.59916, 73.30752],
            ]
        )

        np.testing.assert_allclose(td_spsdl_dd.head().values, cd_spsdl_dd, atol=1e-5)
        np.testing.assert_allclose(
            td_spsdl_md.head().values, cd_spsdl_md_head, atol=1e-5
        )
        np.testing.assert_allclose(
            td_spsdl_md.tail().values, cd_spsdl_md_tail, atol=1e-5
        )
        np.testing.assert_allclose(td_spsdl_avg.head().values, cd_spsdl_avg, atol=1e-5)
        np.testing.assert_allclose(
            td_spsdl_avg["time_bins"].head().astype("int64"), cc.astype("int64"), atol=1
        )
        np.testing.assert_allclose(td_spsdl_sum.head().values, cd_spsdl_sum, atol=1e-5)
        np.testing.assert_allclose(
            td_spsdl_sum["time_bins"].head().astype("int64"), cc.astype("int64"), atol=1
        )

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
        from mhkit.acoustics.spsdl import _validate_method

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
