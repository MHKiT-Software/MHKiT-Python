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
        self.spsd = acoustics.sound_pressure_spectral_density(
            P, P.fs, bin_length=1, pct_overlap=0.5
        )

    @classmethod
    def tearDownClass(self):
        pass

    def test_frequency_bands(self):
        _, third_octaves = acoustics.create_frequency_bands(3, 2, fmin=1, fmax=48000)
        _, decidecades = acoustics.create_frequency_bands(10, 10, fmin=1, fmax=48000)
        millidecades = acoustics.analysis._get_band_table(
            self.spsd["freq"][:460].values,
            bands_per_division=1000,
            base=10,
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
        self.assertIn("time_psd", spsd.dims)
        self.assertEqual(spsd.attrs["units"], "Pa^2/Hz")
        self.assertEqual(spsd.attrs["bin_length"], bin_length)
        self.assertEqual(spsd.attrs["n_fft"], bin_length * fs)

        # Calculate expected number of segments using the default pct_overlap=0.5
        default_pct_overlap = 0.5
        step = int(win_samples * (1 - default_pct_overlap))
        expected_segments = (len(pressure) - win_samples) // step + 1
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
                "2023-02-04T15:05:08.499983072",
                "2023-02-04T15:05:08.999971389",
                "2023-02-04T15:05:09.499959945",
                "2023-02-04T15:05:09.999948263",
                "2023-02-04T15:05:10.499936580",
            ],
            dtype="datetime64[ns]",
        )
        cd_spsdl = np.array(
            [
                [61.988365, 60.721523, 61.288155, 62.367813, 53.957424],
                [62.66007, 62.149723, 63.124702, 53.72768, 59.11971],
                [64.50701, 63.48456, 56.372044, 55.36059, 64.91209],
                [58.399498, 55.154545, 60.971836, 63.37324, 61.89729],
                [54.477947, 64.40175, 68.13605, 66.5518, 56.858593],
            ]
        )

        np.testing.assert_allclose(td_spsdl.head().values, cd_spsdl, atol=1e-6)
        np.testing.assert_allclose(
            td_spsdl["time_psd"].head().astype("int64"), cc.astype("int64"), atol=1
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
                "2023-02-04T15:05:23.499983072",
                "2023-02-04T15:05:53.499983072",
                "2023-02-04T15:06:23.499983072",
                "2023-02-04T15:06:53.499983072",
                "2023-02-04T15:07:23.499983072",
            ],
            dtype="datetime64[ns]",
        )
        cd_spsdl_50 = np.array(
            [
                [62.650208, 64.61388, 64.1676, 66.67151, 72.549095],
                [62.699905, 64.61284, 65.24815, 66.801186, 72.43181],
                [64.310326, 64.8985, 65.44002, 67.19502, 72.76947],
                [63.27369, 62.90754, 65.678055, 67.20142, 72.70676],
                [62.58354, 63.303997, 62.390503, 65.700806, 72.212715],
            ]
        )
        cd_spsdl_25 = np.array(
            [
                [58.7102747, 62.32302952, 61.36541462, 63.74158669, 70.69015503],
                [58.94279099, 61.4001379, 61.44286156, 64.70808029, 70.42274666],
                [59.65940571, 62.35227299, 63.31204796, 64.50473785, 71.06458282],
                [60.09864521, 59.31335735, 62.70079613, 65.46805, 70.75235939],
                [59.34419155, 59.50090408, 59.94284725, 63.75802994, 70.25077629],
            ]
        )
        cd_spsdl_75 = np.array(
            [
                [65.79078293, 67.23075867, 67.50594711, 68.57876396, 74.55014038],
                [65.39378929, 67.34802818, 67.36330605, 69.40702248, 74.43621254],
                [66.60365868, 67.17274666, 68.90455437, 68.7287178, 74.3719368],
                [66.52753067, 66.5525074, 67.94759941, 70.20380783, 74.38684845],
                [65.72010803, 66.15189743, 65.57708931, 67.85847664, 74.39115524],
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
                [66.201004, 63.131325, 61.059765, 62.673466, 61.041855],
                [66.8727, 63.80303, 62.277332, 64.10167, 62.71208],
                [68.71965, 65.64997, 63.751846, 65.4365, 60.947254],
                [62.612133, 59.542458, 56.17782, 57.10649, 59.21533],
                [58.690582, 55.620907, 63.351555, 66.3537, 66.82849],
            ]
        )
        cd_spsdl_md_head = np.array(
            [
                [61.988365, 60.721523, 61.288155, 62.367813, 53.957424],
                [62.66007, 62.149723, 63.124702, 53.72768, 59.11971],
                [64.50701, 63.48456, 56.372044, 55.36059, 64.91209],
                [58.399498, 55.154545, 60.971836, 63.37324, 61.89729],
                [54.477947, 64.40175, 68.13605, 66.5518, 56.858593],
            ]
        )
        cd_spsdl_md_tail = np.array(
            [
                [33.159134, 44.89513, 33.894135, 35.1973, 36.2197],
                [34.903515, 45.169323, 34.188465, 34.95029, 34.81385],
                [34.266838, 45.30183, 34.427937, 34.59744, 36.227085],
                [33.16512, 45.08286, 35.00004, 33.66843, 35.56141],
                [34.143913, 45.394566, 33.20116, 34.19309, 35.91248],
            ]
        )

        cc = np.array(
            [
                "2023-02-04T15:05:23.499983072",
                "2023-02-04T15:05:53.499983072",
                "2023-02-04T15:06:23.499983072",
                "2023-02-04T15:06:53.499983072",
                "2023-02-04T15:07:23.499983072",
            ],
            dtype="datetime64[ns]",
        )
        cd_spsdl_avg = np.array(
            [
                [62.115547, 59.03482, 58.580532, 60.669807, 59.47111],
                [61.637085, 58.556362, 57.79539, 59.804375, 58.961555],
                [62.613922, 59.5332, 59.79988, 62.052376, 61.0739],
                [61.275757, 58.195038, 58.81557, 61.13693, 59.859146],
                [61.34059, 58.25987, 57.532692, 59.550877, 58.448174],
            ]
        )
        cd_spsdl_sum = np.array(
            [
                [79.89706, 76.81634, 76.362045, 78.451324, 77.252625],
                [79.418594, 76.337875, 75.576904, 77.58589, 76.74307],
                [80.39543, 77.31471, 77.58139, 79.83389, 78.85541],
                [79.057274, 75.976555, 76.59708, 78.91844, 77.640656],
                [79.1221, 76.04138, 75.3142, 77.33239, 76.22969],
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
