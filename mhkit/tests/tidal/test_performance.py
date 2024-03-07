import unittest
from os.path import abspath, dirname, join, isfile, normpath, relpath
import os
import numpy as np
from numpy.testing import assert_allclose

from mhkit.tidal import resource, graphics, performance
from mhkit.dolfyn import load

testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/tidal")))


class TestResource(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        filename = join(datadir, "adcp.principal.a1.20200815.nc")
        self.ds = load(filename)
        # Emulate power data
        self.power = abs(self.ds["vel"][0, 10] ** 3 * 1e5)

    @classmethod
    def tearDownClass(self):
        pass

    def test_power_curve(self):
        df93_circ = performance.power_curve(
            power=self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            doppler_cell_size=0.5,
            sampling_frequency=1,
            window_avg_time=600,
            turbine_profile="circular",
            diameter=3,
            height=None,
            width=None,
        )
        test_circ = np.array(
            [
                1.26250990e00,
                1.09230978e00,
                1.89122103e05,
                1.03223668e04,
                2.04261423e05,
                1.72095731e05,
            ]
        )

        df93_rect = performance.power_curve(
            power=self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            doppler_cell_size=0.5,
            sampling_frequency=1,
            window_avg_time=600,
            turbine_profile="rectangular",
            diameter=None,
            height=1,
            width=3,
        )
        test_rect = np.array(
            [
                1.15032239e00,
                3.75747621e-01,
                1.73098627e05,
                3.04090212e04,
                2.09073742e05,
                1.27430552e05,
            ]
        )

        assert_allclose(df93_circ.values[-2], test_circ, atol=1e-5)
        assert_allclose(df93_rect.values[-3], test_rect, atol=1e-5)

    def test_power_curve_xarray(self):
        df93_circ = performance.power_curve(
            power=self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            doppler_cell_size=0.5,
            sampling_frequency=1,
            window_avg_time=600,
            turbine_profile="circular",
            diameter=3,
            height=None,
            width=None,
            to_pandas=False,
        )
        test_circ = np.array(
            [
                1.26250990e00,
                1.09230978e00,
                1.89122103e05,
                1.03223668e04,
                2.04261423e05,
                1.72095731e05,
            ]
        )

        df93_rect = performance.power_curve(
            power=self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            doppler_cell_size=0.5,
            sampling_frequency=1,
            window_avg_time=600,
            turbine_profile="rectangular",
            diameter=None,
            height=1,
            width=3,
            to_pandas=False,
        )
        test_rect = np.array(
            [
                1.15032239e00,
                3.75747621e-01,
                1.73098627e05,
                3.04090212e04,
                2.09073742e05,
                1.27430552e05,
            ]
        )

        assert_allclose(df93_circ.isel(U_bins=-2).to_array(), test_circ, atol=1e-5)
        assert_allclose(df93_rect.isel(U_bins=-3).to_array(), test_rect, atol=1e-5)

    def test_velocity_profiles(self):
        df94 = performance.velocity_profiles(
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            water_depth=10,
            sampling_frequency=1,
            window_avg_time=600,
            function="mean",
        )
        df95a = performance.velocity_profiles(
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            water_depth=10,
            sampling_frequency=1,
            window_avg_time=600,
            function="rms",
        )
        df95b = performance.velocity_profiles(
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            water_depth=10,
            sampling_frequency=1,
            window_avg_time=600,
            function="std",
        )

        test_df94 = np.array([0.32782955, 0.69326691, 1.00948623])
        test_df95a = np.array([0.3329345, 0.69936798, 1.01762123])
        test_df95b = np.array([0.05635571, 0.08671777, 0.12735139])

        assert_allclose(df94.values[1], test_df94, atol=1e-5)
        assert_allclose(df95a.values[1], test_df95a, atol=1e-5)
        assert_allclose(df95b.values[1], test_df95b, atol=1e-5)

    def test_velocity_profiles_xarray(self):
        df94 = performance.velocity_profiles(
            velocity=self.ds["vel"].sel(dir="streamwise"),
            hub_height=4.2,
            water_depth=10,
            sampling_frequency=1,
            window_avg_time=600,
            function="mean",
            to_pandas=False,
        )

        test_df94 = np.array([0.32782955, 0.69326691, 1.00948623])

        assert_allclose(df94[1], test_df94, atol=1e-5)

    def test_power_efficiency(self):
        df97 = performance.device_efficiency(
            self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            water_density=self.ds["water_density"],
            capture_area=np.pi * 1.5**2,
            hub_height=4.2,
            sampling_frequency=1,
            window_avg_time=600,
        )

        test_df97 = np.array(24.79197)
        assert_allclose(df97.values[-1, -1], test_df97, atol=1e-5)

    def test_power_efficiency_xarray(self):
        df97 = performance.device_efficiency(
            self.power,
            velocity=self.ds["vel"].sel(dir="streamwise"),
            water_density=self.ds["water_density"],
            capture_area=np.pi * 1.5**2,
            hub_height=4.2,
            sampling_frequency=1,
            window_avg_time=600,
            to_pandas=False,
        )

        test_df97 = np.array(24.79197)
        assert_allclose(df97["Efficiency"][-1], test_df97, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
