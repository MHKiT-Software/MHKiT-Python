from os.path import abspath, dirname, join, normpath, relpath
import mhkit.power as power
import pandas as pd
import xarray as xr
import numpy as np
import unittest


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir, relpath("../../../examples/data/power")))


class TestDevice(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.t = 600
        fs = 1000
        self.samples = np.linspace(0, self.t, int(fs * self.t), endpoint=False)
        self.frequency = 60
        self.freq_array = np.ones(len(self.samples)) * 60
        harmonics_int = np.arange(0, 60 * 60, 5)
        self.harmonics_int = harmonics_int
        # since this is an idealized sin wave, the interharmonics should be zero
        self.interharmonic = np.zeros(len(harmonics_int))
        self.harmonics_vals = np.zeros(len(harmonics_int))
        # setting 60th harmonic to amplitude of the signal
        self.harmonics_vals[12] = 1.0

        # harmonic groups should be equal to every 12th harmonic in this idealized example
        self.harmonic_groups = self.harmonics_vals[0::12]
        self.thcd = (
            0.0  # Since this is an idealized sin wave, there should be no distortion
        )

        self.signal = np.sin(2 * np.pi * self.frequency * self.samples)

        self.current_data = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.voltage_data = np.asarray([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])

    @classmethod
    def tearDownClass(self):
        pass

    def test_harmonics_sine_wave_pandas(self):
        current = pd.Series(self.signal, index=self.samples)
        harmonics = power.quality.harmonics(current, 1000, self.frequency)

        for i, j in zip(harmonics["data"].values, self.harmonics_vals):
            self.assertAlmostEqual(i, j, 1)

    def test_harmonics_sine_wave_xarray(self):
        current = xr.DataArray(
            data=self.signal, dims="index", coords={"index": self.samples}
        )
        harmonics = power.quality.harmonics(current, 1000, self.frequency)

        for i, j in zip(harmonics["data"].values, self.harmonics_vals):
            self.assertAlmostEqual(i, j, 1)

    def test_harmonic_subgroup_sine_wave_pandas(self):
        harmonics = pd.DataFrame(self.harmonics_vals, index=self.harmonics_int)
        hsg = power.quality.harmonic_subgroups(harmonics, self.frequency)

        for i, j in zip(hsg.values, self.harmonic_groups):
            self.assertAlmostEqual(i[0], j, 1)

    def test_harmonic_subgroup_sine_wave_xarray(self):
        harmonics = xr.Dataset(
            data_vars={"harmonics": (["index"], self.harmonics_vals)},
            coords={"index": self.harmonics_int},
        )
        hsg = power.quality.harmonic_subgroups(harmonics, self.frequency)

        for i, j in zip(hsg.values, self.harmonic_groups):
            self.assertAlmostEqual(i[0], j, 1)

    def test_TCHD_sine_wave_pandas(self):
        harmonics = pd.DataFrame(self.harmonics_vals, index=self.harmonics_int)
        hsg = power.quality.harmonic_subgroups(harmonics, self.frequency)
        TCHD = power.quality.total_harmonic_current_distortion(hsg)

        self.assertAlmostEqual(TCHD.values[0], self.thcd)

    def test_TCHD_sine_wave_xarray(self):
        harmonics = xr.Dataset(
            data_vars={"harmonics": (["index"], self.harmonics_vals)},
            coords={"index": self.harmonics_int},
        )
        hsg = power.quality.harmonic_subgroups(harmonics, self.frequency)
        TCHD = power.quality.total_harmonic_current_distortion(hsg)

        self.assertAlmostEqual(TCHD.values[0], self.thcd)

    def test_interharmonics_sine_wave_pandas(self):
        harmonics = pd.DataFrame(self.harmonics_vals, index=self.harmonics_int)
        inter_harmonics = power.quality.interharmonics(harmonics, self.frequency)

        for i, j in zip(inter_harmonics.values, self.interharmonic):
            self.assertAlmostEqual(i[0], j, 1)

    def test_interharmonics_sine_wave_xarray(self):
        harmonics = xr.Dataset(
            data_vars={"harmonics": (["index"], self.harmonics_vals)},
            coords={"index": self.harmonics_int},
        )
        inter_harmonics = power.quality.interharmonics(harmonics, self.frequency)

        for i, j in zip(inter_harmonics.values, self.interharmonic):
            self.assertAlmostEqual(i[0], j, 1)

    def test_instfreq_pandas(self):
        um = pd.Series(self.signal, index=self.samples)

        freq = power.characteristics.instantaneous_frequency(um)
        for i in freq.values:
            self.assertAlmostEqual(i[0], self.frequency, 1)

    def test_instfreq_xarray(self):
        um = pd.Series(self.signal, index=self.samples)
        um = um.to_xarray()

        freq = power.characteristics.instantaneous_frequency(um)
        for i in freq.values:
            self.assertAlmostEqual(i[0], self.frequency, 1)

    def test_dc_power_pandas(self):
        current = pd.DataFrame(self.current_data, columns=["A1", "A2", "A3"])
        voltage = pd.DataFrame(self.voltage_data, columns=["V1", "V2", "V3"])

        P = power.characteristics.dc_power(voltage, current)
        P_test = (self.current_data * self.voltage_data).sum()
        self.assertEqual(P.sum()["Gross"], P_test)

        P = power.characteristics.dc_power(voltage["V1"], current["A1"])
        P_test = (self.current_data[:, 0] * self.voltage_data[:, 0]).sum()
        self.assertEqual(P.sum()["Gross"], P_test)

    def test_dc_power_xarray(self):
        current = pd.DataFrame(self.current_data, columns=["A1", "A2", "A3"])
        voltage = pd.DataFrame(self.voltage_data, columns=["V1", "V2", "V3"])
        current = current.to_xarray()
        voltage = voltage.to_xarray()

        P = power.characteristics.dc_power(voltage, current)
        P_test = (self.current_data * self.voltage_data).sum()
        self.assertEqual(P.sum()["Gross"], P_test)

        P = power.characteristics.dc_power(voltage["V1"], current["A1"])
        P_test = (self.current_data[:, 0] * self.voltage_data[:, 0]).sum()
        self.assertEqual(P.sum()["Gross"], P_test)

    def test_ac_power_three_phase_pandas(self):
        current = pd.DataFrame(self.current_data, columns=["A1", "A2", "A3"])
        voltage = pd.DataFrame(self.voltage_data, columns=["V1", "V2", "V3"])

        P1 = power.characteristics.ac_power_three_phase(voltage, current, 1, False)
        P1b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, False)
        P2 = power.characteristics.ac_power_three_phase(voltage, current, 1, True)
        P2b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, True)

        P_test = (self.current_data * self.voltage_data).sum()
        self.assertEqual(P1.sum().iloc[0], P_test)
        self.assertEqual(P1b.sum().iloc[0], P_test / 2)
        self.assertAlmostEqual(P2.sum().iloc[0], P_test * np.sqrt(3), 2)
        self.assertAlmostEqual(P2b.sum().iloc[0], P_test * np.sqrt(3) / 2, 2)

    def test_ac_power_three_phase_xarray(self):
        current = pd.DataFrame(self.current_data, columns=["A1", "A2", "A3"])
        voltage = pd.DataFrame(self.voltage_data, columns=["V1", "V2", "V3"])
        current = current.to_xarray()
        voltage = voltage.to_xarray()

        P1 = power.characteristics.ac_power_three_phase(voltage, current, 1, False)
        P1b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, False)
        P2 = power.characteristics.ac_power_three_phase(voltage, current, 1, True)
        P2b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, True)

        P_test = (self.current_data * self.voltage_data).sum()
        self.assertEqual(P1.sum().iloc[0], P_test)
        self.assertEqual(P1b.sum().iloc[0], P_test / 2)
        self.assertAlmostEqual(P2.sum().iloc[0], P_test * np.sqrt(3), 2)
        self.assertAlmostEqual(P2b.sum().iloc[0], P_test * np.sqrt(3) / 2, 2)


if __name__ == "__main__":
    unittest.main()
