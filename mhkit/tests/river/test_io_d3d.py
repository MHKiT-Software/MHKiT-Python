from os.path import abspath, dirname, join, normpath
from numpy.testing import assert_array_almost_equal
import scipy.interpolate as interp
import mhkit.river as river
import mhkit.tidal as tidal
import pandas as pd
import xarray as xr
import numpy as np
import unittest
import netCDF4
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "river"))


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        d3ddatadir = normpath(join(datadir, "d3d"))

        filename = "turbineTest_map.nc"
        self.d3d_flume_data = netCDF4.Dataset(join(d3ddatadir, filename))

    @classmethod
    def tearDownClass(self):
        pass

    def test_get_all_time(self):
        data = self.d3d_flume_data
        seconds_run = river.io.d3d.get_all_time(data)
        seconds_run_expected = np.ndarray(
            shape=(5,), buffer=np.array([0, 60, 120, 180, 240]), dtype=int
        )
        np.testing.assert_array_equal(seconds_run, seconds_run_expected)

    def test_convert_time(self):
        data = self.d3d_flume_data
        time_index = 2
        seconds_run = river.io.d3d.index_to_seconds(data, time_index=time_index)
        seconds_run_expected = 120
        self.assertEqual(seconds_run, seconds_run_expected)
        seconds_run = 60
        time_index = river.io.d3d.seconds_to_index(data, seconds_run=seconds_run)
        time_index_expected = 1
        self.assertEqual(time_index, time_index_expected)
        seconds_run = 62
        time_index = river.io.d3d.seconds_to_index(data, seconds_run=seconds_run)
        time_index_expected = 1
        output_expected = f"ERROR: invalid seconds_run. Closest seconds_run found {time_index_expected}"
        self.assertWarns(UserWarning)

    def test_convert_time_from_tidal(self):
        """
        Test the conversion of time from using tidal import of d3d
        """
        data = self.d3d_flume_data
        time_index = 2
        seconds_run = tidal.io.d3d.index_to_seconds(data, time_index=time_index)
        seconds_run_expected = 120
        self.assertEqual(seconds_run, seconds_run_expected)

    def test_layer_data(self):
        data = self.d3d_flume_data
        variable = ["ucx", "s1"]
        for var in variable:
            layer = 2
            time_index = 3
            layer_data = river.io.d3d.get_layer_data(data, var, layer, time_index)
            layer_compare = 2
            time_index_compare = 4
            layer_data_expected = river.io.d3d.get_layer_data(
                data, var, layer_compare, time_index_compare
            )

            assert_array_almost_equal(layer_data.x, layer_data_expected.x, decimal=2)
            assert_array_almost_equal(layer_data.y, layer_data_expected.y, decimal=2)
            assert_array_almost_equal(layer_data.v, layer_data_expected.v, decimal=2)

    def test_create_points_three_points(self):
        """
        Test the scenario where all three inputs (x, y, z) are points.
        """
        x, y, z = 1, 2, 3

        expected = pd.DataFrame([[x, y, z]], columns=["x", "y", "waterdepth"])

        points = river.io.d3d.create_points(x, y, z)
        assert_array_almost_equal(points.values, expected.values, decimal=2)

    def test_create_points_invalid_input(self):
        """
        Test scenarios where invalid inputs are provided to the function.
        """
        with self.assertRaises(TypeError):
            river.io.d3d.create_points("invalid", 2, 3)

    def test_create_points_two_arrays_one_point(self):
        """
        Test with two arrays and one point.
        """
        result = river.io.d3d.create_points(np.array([1, 2]), np.array([3]), 4)
        expected = pd.DataFrame({"x": [1, 2], "y": [3, 3], "waterdepth": [4, 4]})
        pd.testing.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
            check_names=False,
            check_index_type=False,
        )

    def test_create_points_user_made_two_arrays_one_point(self):
        """
        Test the scenario where all three inputs (x, y, z) are created from
        points.
        """
        x, y, z = np.linspace(1, 3, num=3), np.linspace(1, 3, num=3), 1

        # Adjust the order of the expected values
        expected_data = [
            [i, j, 1] for j in y for i in x
        ]  # Notice the swapped loop order
        expected = pd.DataFrame(expected_data, columns=["x", "y", "waterdepth"])

        points = river.io.d3d.create_points(x, y, z)
        assert_array_almost_equal(points.values, expected.values, decimal=2)

    def test_create_points_mismatched_array_lengths(self):
        """
        Test the scenario where x and y are arrays of different lengths.
        """
        with self.assertRaises(ValueError):
            river.io.d3d.create_points(
                np.array([1, 2, 3]), np.array([1, 2]), np.array([3, 4])
            )

    def test_create_pointsempty_arrays(self):
        """
        Test the scenario where provided arrays are empty.
        """
        with self.assertRaises(ValueError):
            river.io.d3d.create_points([], [], [])

    def test_create_points_mixed_data_types(self):
        """
        Test a combination of np.ndarray, pd.Series, and xr.DataArray.
        """
        x = np.array([1, 2])
        y = pd.Series([3, 4])
        z = xr.DataArray([5, 6])
        result = river.io.d3d.create_points(x, y, z)
        expected = pd.DataFrame(
            {"x": [1, 2, 1, 2], "y": [3, 4, 3, 4], "waterdepth": [5, 5, 6, 6]}
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
            check_names=False,
            check_index_type=False,
        )

    def test_create_points_array_like_inputs(self):
        """
        Test array-like inputs such as lists.
        """
        result = river.io.d3d.create_points([1, 2], [3, 4], [5, 6])
        expected = pd.DataFrame(
            {"x": [1, 2, 1, 2], "y": [3, 4, 3, 4], "waterdepth": [5, 5, 6, 6]}
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
            check_names=False,
            check_index_type=False,
        )

    def test_variable_interpolation(self):
        data = self.d3d_flume_data
        variables = ["ucx", "turkin1"]
        transformes_data = river.io.d3d.variable_interpolation(
            data, variables, points="faces", edges="nearest"
        )
        self.assertEqual(
            np.size(transformes_data["ucx"]), np.size(transformes_data["turkin1"])
        )
        transformes_data = river.io.d3d.variable_interpolation(
            data, variables, points="cells", edges="nearest"
        )
        self.assertEqual(
            np.size(transformes_data["ucx"]), np.size(transformes_data["turkin1"])
        )
        x = np.linspace(1, 3, num=3)
        y = np.linspace(1, 3, num=3)
        waterdepth = 1
        points = river.io.d3d.create_points(x, y, waterdepth)
        transformes_data = river.io.d3d.variable_interpolation(
            data, variables, points=points
        )
        self.assertEqual(
            np.size(transformes_data["ucx"]), np.size(transformes_data["turkin1"])
        )

    def test_get_all_data_points(self):
        data = self.d3d_flume_data
        variable = "ucx"
        time_step = 3
        output = river.io.d3d.get_all_data_points(data, variable, time_step)
        size_output = np.size(output)
        time_step_compair = 4
        output_expected = river.io.d3d.get_all_data_points(
            data, variable, time_step_compair
        )
        size_output_expected = np.size(output_expected)
        self.assertEqual(size_output, size_output_expected)

    def test_unorm(self):
        x = np.linspace(1, 3, num=3)
        y = np.linspace(1, 3, num=3)
        z = np.linspace(1, 3, num=3)
        unorm = river.io.d3d.unorm(x, y, z)
        unorm_expected = [
            np.sqrt(1**2 + 1**2 + 1**2),
            np.sqrt(2**2 + 2**2 + 2**2),
            np.sqrt(3**2 + 3**2 + 3**2),
        ]
        assert_array_almost_equal(unorm, unorm_expected, decimal=2)

    def test_turbulent_intensity(self):
        data = self.d3d_flume_data
        time_index = -1
        x_test = np.linspace(1, 17, num=10)
        y_test = np.linspace(3, 3, num=10)
        waterdepth_test = np.linspace(1, 1, num=10)

        test_points = np.array(
            [
                [x, y, waterdepth]
                for x, y, waterdepth in zip(x_test, y_test, waterdepth_test)
            ]
        )
        points = pd.DataFrame(test_points, columns=["x", "y", "waterdepth"])

        TI = river.io.d3d.turbulent_intensity(data, points, time_index)

        TI_vars = ["turkin1", "ucx", "ucy", "ucz"]
        TI_data_raw = {}
        for var in TI_vars:
            # get all data
            var_data_df = river.io.d3d.get_all_data_points(data, var, time_index)
            TI_data_raw[var] = var_data_df
            TI_data = points.copy(deep=True)

        for var in TI_vars:
            TI_data[var] = interp.griddata(
                TI_data_raw[var][["x", "y", "waterdepth"]],
                TI_data_raw[var][var],
                points[["x", "y", "waterdepth"]],
            )
            idx = np.where(np.isnan(TI_data[var]))

            if len(idx[0]):
                for i in idx[0]:
                    TI_data[var][i] = interp.griddata(
                        TI_data_raw[var][["x", "y", "waterdepth"]],
                        TI_data_raw[var][var],
                        [points["x"][i], points["y"][i], points["waterdepth"][i]],
                        method="nearest",
                    )

        u_mag = river.io.d3d.unorm(TI_data["ucx"], TI_data["ucy"], TI_data["ucz"])
        turbulent_intensity_expected = (
            np.sqrt(2 / 3 * TI_data["turkin1"]) / u_mag
        ) * 100

        assert_array_almost_equal(
            TI.turbulent_intensity, turbulent_intensity_expected, decimal=2
        )

        TI = river.io.d3d.turbulent_intensity(data, points="faces")
        TI_size = np.size(TI["turbulent_intensity"])
        turkin1 = river.io.d3d.get_all_data_points(data, "turkin1", time_index)
        turkin1_size = np.size(turkin1["turkin1"])
        self.assertEqual(TI_size, turkin1_size)

        TI = river.io.d3d.turbulent_intensity(data, points="cells")
        TI_size = np.size(TI["turbulent_intensity"])
        ucx = river.io.d3d.get_all_data_points(data, "ucx", time_index)
        ucx_size = np.size(ucx["ucx"])
        self.assertEqual(TI_size, ucx_size)


if __name__ == "__main__":
    unittest.main()
