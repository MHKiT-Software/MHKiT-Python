import unittest
from os.path import abspath, dirname, join, normpath
from matplotlib.animation import FuncAnimation
import xarray as xr
import mhkit.mooring as mooring
import pytest
import numpy as np

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "mooring"))


class TestMooring(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        fpath = join(datadir, "line1_test.nc")
        self.ds = xr.open_dataset(fpath)
        self.dsani = self.ds.sel(Time=slice(0, 10))

    def test_moordyn_out(self):
        fpath = join(datadir, "Test.MD.out")
        inputpath = join(datadir, "TestInput.MD.dat")
        ds = mooring.io.read_moordyn(fpath, input_file=inputpath)
        self.assertIsInstance(ds, xr.Dataset)

    def test_lay_length(self):
        fpath = join(datadir, "line1_test.nc")
        ds = xr.open_dataset(fpath)
        laylengths = mooring.lay_length(ds, depth=-56, tolerance=0.25)
        laylength = laylengths.mean().values
        self.assertAlmostEqual(laylength, 45.0, 1)

    def test_animate_3d(self):
        dsani = self.ds.sel(Time=slice(0, 10))
        ani = mooring.graphics.animate(
            dsani,
            dimension="3d",
            interval=10,
            repeat=True,
            xlabel="X-axis",
            ylabel="Y-axis",
            zlabel="Depth [m]",
            title="Mooring Line Example",
        )
        self.assertIsInstance(ani, FuncAnimation)

    def test_animate_2d(self):
        dsani = self.ds.sel(Time=slice(0, 10))
        ani2d = mooring.graphics.animate(
            dsani,
            dimension="2d",
            xaxis="x",
            yaxis="z",
            repeat=True,
            xlabel="X-axis",
            ylabel="Depth [m]",
            title="Mooring Line Example",
        )
        self.assertIsInstance(ani2d, FuncAnimation)

    def test_animate_2d_update(self):
        ani2d = mooring.graphics.animate(
            self.ds,
            dimension="2d",
            xaxis="x",
            yaxis="z",
            repeat=True,
            xlabel="X-axis",
            ylabel="Depth [m]",
            title="Mooring Line Example",
        )

        # Extract the figure and axes
        fig = ani2d._fig
        ax = fig.axes[0]
        (line,) = ax.lines

        # Simulate the update for a specific frame
        frame = 5

        # Extracting data from the list of nodes
        nodes_x, nodes_y, _ = mooring.graphics._get_axis_nodes(
            self.dsani, "x", "z", "y"
        )
        x_data = self.dsani[nodes_x[0]].isel(Time=frame).values
        y_data = self.dsani[nodes_y[0]].isel(Time=frame).values

        # Manually set the data for the line object
        line.set_data([x_data], [y_data])

        # Extract updated data from the line object
        updated_x, updated_y = line.get_data()

        # Assert that the updated data matches the dataset
        np.testing.assert_array_equal(updated_x, x_data)
        np.testing.assert_array_equal(updated_y, y_data)

    def test_animate_3d_update(self):
        ani3d = mooring.graphics.animate(
            self.ds,
            dimension="3d",
            xaxis="x",
            yaxis="z",
            zaxis="y",
            repeat=True,
            xlabel="X-axis",
            ylabel="Depth [m]",
            zlabel="Y-axis",
            title="Mooring Line Example",
        )

        # Extract the figure and axes
        fig = ani3d._fig
        ax = fig.axes[0]
        (line,) = ax.lines

        # Simulate the update for a specific frame
        frame = 5

        # Extracting data for the specified frame
        nodes_x, nodes_y, nodes_z = mooring.graphics._get_axis_nodes(
            self.dsani, "x", "z", "y"
        )
        x_data = self.dsani[nodes_x[0]].isel(Time=frame).values
        y_data = self.dsani[nodes_y[0]].isel(Time=frame).values
        z_data = self.dsani[nodes_z[0]].isel(Time=frame).values

        # Manually set the data for the line object
        line.set_data([x_data], [y_data])
        line.set_3d_properties(z_data)

        # Extract updated data from the line object
        updated_x, updated_y, updated_z = line._verts3d

        # Assert that the updated data matches the dataset
        np.testing.assert_array_equal(updated_x, x_data)
        np.testing.assert_array_equal(updated_y, y_data)
        np.testing.assert_array_equal(updated_z, z_data)

    # Test for xaxis, yaxis, zaxis type handling
    def test_animate_xaxis_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, xaxis=123)

    def test_animate_yaxis_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, yaxis=123)

    def test_animate_zaxis_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, zaxis=123)

    # Test for zlim and zlabel in 3D mode
    def test_animate_zlim_type_handling_3d(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, dimension="3d", zlim="invalid")

    def test_animate_zlabel_type_handling_3d(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, dimension="3d", zlabel=123)

    # Test for xlim, ylim, interval, repeat, xlabel, ylabel, title
    def test_animate_xlim_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, xlim="invalid")

    def test_animate_ylim_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, ylim="invalid")

    def test_animate_interval_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, interval="invalid")

    def test_animate_repeat_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, repeat="invalid")

    def test_animate_xlabel_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, xlabel=123)

    def test_animate_ylabel_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, ylabel=123)

    def test_animate_title_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, title=123)

    def test_animate_dsani_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate("not_a_dataset")

    def test_animate_xlim_type_handling_none(self):
        try:
            mooring.graphics.animate(self.dsani, xlim=None)
        except TypeError:
            pytest.fail("Unexpected TypeError with xlim=None")

    def test_animate_ylim_type_handling_none(self):
        try:
            mooring.graphics.animate(self.dsani, ylim=None)
        except TypeError:
            pytest.fail("Unexpected TypeError with ylim=None")

    def test_animate_interval_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, interval="not_an_int")

    def test_animate_repeat_type_handling(self):
        with pytest.raises(TypeError):
            mooring.graphics.animate(self.dsani, repeat="not_a_bool")

    def test_animate_xlabel_type_handling_none(self):
        try:
            mooring.graphics.animate(self.dsani, xlabel=None)
        except TypeError:
            pytest.fail("Unexpected TypeError with xlabel=None")

    def test_animate_ylabel_type_handling_none(self):
        try:
            mooring.graphics.animate(self.dsani, ylabel=None)
        except TypeError:
            pytest.fail("Unexpected TypeError with ylabel=None")

    def test_animate_title_type_handling_none(self):
        try:
            mooring.graphics.animate(self.dsani, title=None)
        except TypeError:
            pytest.fail("Unexpected TypeError with title=None")

    def test_animate_dimension_type_handling(self):
        with pytest.raises(ValueError):
            mooring.graphics.animate(self.dsani, dimension="not_2d_or_3d")


if __name__ == "__main__":
    unittest.main()
