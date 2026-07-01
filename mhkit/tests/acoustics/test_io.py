import os
from os.path import abspath, dirname, join, normpath, isfile
import numpy as np
import pandas as pd
import unittest

import mhkit.acoustics as acoustics

testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "acoustics"))


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_read_wav_metadata(self):
        # Test read_wav_metadata function
        file_name = join(datadir, "RBW_6661_20240601_053114.wav")
        with open(file_name, "rb") as f:
            header = acoustics.io._read_wav_metadata(f)
        expected_bits_per_sample = 24
        self.assertEqual(header["bits_per_sample"], expected_bits_per_sample)

    def test_calculate_voltage_and_time(self):
        # Test calculate_voltage_and_time function
        fs = 1
        raw = np.array([0, 32767, -32768, 16384, -16384], dtype=np.int16)
        bits_per_sample = 16
        peak_voltage = 2.5
        start_time = "2024-06-06T00:00:00"

        raw_voltage, time, max_count = acoustics.io._calculate_voltage_and_time(
            fs, raw, bits_per_sample, peak_voltage, start_time
        )

        # Expected max_count
        expected_max_count = 2 ** (bits_per_sample - 1)
        self.assertEqual(max_count, expected_max_count)

        # Expected raw_voltage
        expected_raw_voltage = raw.astype(float) / expected_max_count * peak_voltage
        np.testing.assert_allclose(raw_voltage, expected_raw_voltage, atol=1e-6)

        # Expected time array
        end_time = np.datetime64(start_time) + np.timedelta64(
            raw.size * 1000000000, "ns"
        )
        expected_time = pd.date_range(start_time, end_time, raw.size + 1)
        pd.testing.assert_index_equal(time, expected_time)

    def test_read_iclisten_metadata(self):
        from mhkit.acoustics.io import _read_iclisten_metadata

        file_name = join(datadir, "RBW_6661_20240601_053114.wav")

        with open(file_name, "rb") as f:
            metadata = _read_iclisten_metadata(f)

        expected_metadata = {
            "iart": "icListen HF #6661",
            "iprd": "RB9-ETH R8",
            "icrd": "2024-06-01T05:31:14+00",
            "isft": "icListen HF R40.0",
            "inam": "RBW6661_20240601_053114",
            "peak_voltage": 3.0,
            "stored_sensitivity": -177,
            "humidity": "24.0 % RH",
            "temperature": "8.6 deg C",
            "accelerometer": "Acc(-980,-18,141)",
            "magnetometer": "Mag(3603,3223,-598)",
            "count_at_peak_voltage": "8388608 = Max Count",
            "sequence_num": "2589798400000 = Seq #",
        }

        # Assertions to check if metadata matches expected values
        for key, expected_value in expected_metadata.items():
            self.assertIn(key, metadata)
            if isinstance(expected_value, float):
                self.assertAlmostEqual(metadata[key], expected_value, places=6)
            else:
                self.assertEqual(metadata[key], expected_value)

    def test_read_iclisten(self):
        file_name = join(datadir, "RBW_6661_20240601_053114.wav")
        td_orig = acoustics.io.read_iclisten(file_name)
        td_wrap = acoustics.io.read_hydrophone(
            file_name,
            peak_voltage=3,
            sensitivity=-177,
            start_time="2024-06-01T05:31:14",
        )
        td_volt = acoustics.io.read_iclisten(
            file_name, sensitivity=None, use_metadata=False
        )
        td_ovrrd = acoustics.io.read_iclisten(
            file_name, sensitivity=-180, use_metadata=False
        )
        td_ovrrd2 = acoustics.io.read_iclisten(
            file_name, sensitivity=-180, use_metadata=True
        )

        # Check time coordinate
        cc = np.array(
            [
                "2024-06-01T05:31:14.000000000",
                "2024-06-01T05:31:14.000001953",
                "2024-06-01T05:31:14.000003906",
                "2024-06-01T05:31:14.000005859",
                "2024-06-01T05:31:14.000007812",
            ],
            dtype="datetime64[ns]",
        )
        # Check data
        cd_orig = np.array([0.31546374, 0.30229832, 0.32229963, 0.3159701, 0.30356423])
        cd_volt = np.array([0.0004456, 0.00042701, 0.00045526, 0.00044632, 0.0004288])
        cd_ovrrd = np.array(
            [0.44560438, 0.42700773, 0.45526033, 0.44631963, 0.42879587]
        )

        np.testing.assert_allclose(td_orig.head().values, cd_orig, atol=1e-6)
        np.testing.assert_equal(td_orig["time"].head().values, cc)

        np.testing.assert_allclose(td_wrap.head().values, cd_orig, atol=1e-6)
        np.testing.assert_equal(td_wrap["time"].head().values, cc)

        np.testing.assert_allclose(td_volt.head().values, cd_volt, atol=1e-6)
        np.testing.assert_equal(td_volt["time"].head().values, cc)

        np.testing.assert_allclose(td_ovrrd.head().values, cd_ovrrd, atol=1e-6)
        np.testing.assert_equal(td_ovrrd["time"].head().values, cc)

        np.testing.assert_allclose(
            td_ovrrd.head().values, td_ovrrd2.head().values, atol=1e-6
        )

    def test_read_soundtrap(self):
        file_name = join(datadir, "6247.230204150508.wav")
        td_orig = acoustics.io.read_soundtrap(file_name, sensitivity=-177)
        td_wrap = acoustics.io.read_hydrophone(
            file_name,
            peak_voltage=1,
            sensitivity=-177,
            start_time="2023-02-04T15:05:08",
        )
        td_volt = acoustics.io.read_soundtrap(file_name, sensitivity=None)

        # Check time coordinate
        cc = np.array(
            [
                "2023-02-04T15:05:08.000000000",
                "2023-02-04T15:05:08.000010416",
                "2023-02-04T15:05:08.000020832",
                "2023-02-04T15:05:08.000031249",
                "2023-02-04T15:05:08.000041665",
            ],
            dtype="datetime64[ns]",
        )
        # Check data
        cd_orig = np.array([0.929006, 0.929006, 0.929006, 0.929006, 1.01542517])
        cd_volt = np.array([0.00131226, 0.00131226, 0.00131226, 0.00131226, 0.00143433])

        np.testing.assert_allclose(td_orig.head().values, cd_orig, atol=1e-6)
        np.testing.assert_equal(td_orig["time"].head().values, cc)

        np.testing.assert_allclose(td_wrap.head().values, cd_orig, atol=1e-6)
        np.testing.assert_equal(td_wrap["time"].head().values, cc)

        np.testing.assert_allclose(td_volt.head().values, cd_volt, atol=1e-6)
        np.testing.assert_equal(td_volt["time"].head().values, cc)

    def test_calibration(self):
        file_name = join(datadir, "6247.230204150508.wav")
        td_volt = acoustics.io.read_soundtrap(file_name, sensitivity=None)
        td_spsd = acoustics.sound_pressure_spectral_density(
            td_volt, td_volt.fs, bin_length=1
        )

        # Run calibration
        cal_name = join(datadir, "6247_calibration.csv")
        calibration = pd.read_csv(cal_name, sep=",")
        calibration.index = calibration["Frequency"]
        calibration = calibration.to_xarray()
        fill_Sf = calibration["Analog Sensitivity"][-1].values

        # Apply calibration
        td_spsd = acoustics.apply_calibration(
            td_spsd, calibration["Analog Sensitivity"], fill_value=fill_Sf
        )

        # Check time coordinate
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

        cd_spsd = np.array(
            [
                [
                    7.04428102e-02,
                    4.05186564e-03,
                    9.86163910e-04,
                    4.10977795e-04,
                    2.46398082e-05,
                ],
                [
                    8.22255815e-02,
                    5.62956783e-03,
                    1.50523430e-03,
                    5.62088860e-05,
                    8.08846064e-05,
                ],
                [
                    1.25805956e-01,
                    7.65523631e-03,
                    3.17934716e-04,
                    8.18645956e-05,
                    3.06975506e-04,
                ],
                [
                    3.08283050e-02,
                    1.12449483e-03,
                    9.16890290e-04,
                    5.18037680e-04,
                    1.53328923e-04,
                ],
                [
                    1.24966808e-02,
                    9.45535593e-03,
                    4.77241430e-03,
                    1.07700449e-03,
                    4.80567438e-05,
                ],
            ]
        )

        np.testing.assert_allclose(td_spsd.head().values, cd_spsd, atol=1e-6)
        np.testing.assert_allclose(
            td_spsd["time_psd"].head().astype("int64"), cc.astype("int64"), atol=1
        )

    def test_wispr(self):
        file_name = join(datadir, "WISPR_230825_003936.dat")
        td = acoustics.io.read_wispr(file_name)

        # Check time coordinate
        cc = np.array(
            [
                "2023-08-25T00:39:36.000000000",
                "2023-08-25T00:39:36.000020000",
                "2023-08-25T00:39:36.000040000",
                "2023-08-25T00:39:36.000060000",
                "2023-08-25T00:39:36.000080000",
            ],
            dtype="datetime64[ns]",
        )
        # Check data
        cd = np.array([-0.00167847, -0.00167847, -0.00152588, -0.00183105, -0.00106812])

        np.testing.assert_allclose(td.head().values, cd, atol=1e-6)
        np.testing.assert_equal(td["time"].head().values, cc)

    def test_read_wispr_metadata(self):
        from mhkit.acoustics.io import _read_wispr_metadata

        file_name = join(datadir, "WISPR_230825_003936.dat")

        with open(file_name, "rb") as f:
            metadata = _read_wispr_metadata(f)

        expected_metadata = {
            "version": 1.2,
            "time": "08:25:23:00:39:36",
            "instrument_id": "PERI_1",
            "location_id": "PWSPNE",
            "volts": 15.77,
            "blocks_free": 20.98,
            "file_size": 58575,
            "buffer_size": 16896,
            "samples_per_buffer": 8448,
            "sample_size": 2,
            "sampling_rate": 50000,
            "gain": 0,
            "decimation": 16,
            "adc_vref": 5.0,
            "file_length_sec": 299.904,
        }

        # Assertions to check if metadata matches expected values
        for key, expected_value in expected_metadata.items():
            self.assertIn(key, metadata)
            if isinstance(expected_value, float):
                self.assertAlmostEqual(metadata[key], expected_value, places=6)
            else:
                self.assertEqual(metadata[key], expected_value)

    def test_audio_export(self):
        file_name = join(datadir, "RBW_6661_20240601_053114.wav")
        P = acoustics.io.read_iclisten(file_name)
        acoustics.io.export_audio("sound1", P, gain=1)

        self.assertEqual(isfile("sound1.wav"), True)
        os.remove("sound1.wav")


if __name__ == "__main__":
    unittest.main()
