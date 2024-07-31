"""
This module contains unit tests for the WPTO hindcast data retrieval 
functions in the mhkit.wave package. The tests are designed to verify
the correct functioning of the following functionalities:

1. Retrieval of multiple years of data for a single data type,
    latitude-longitude pair, and parameter.
2. Retrieval of multiple parameters for a single data type, year, 
    and latitude-longitude pair.
3. Retrieval of data for multiple locations for point data and
    directional spectrum at a single data type, year, and parameter.

The tests use the unittest framework and compare the output of the 
hindcast retrieval functions with expected output data. The expected 
data is read from CSV files located in the examples/data/wave directory.

Functions tested:
- wave.io.hindcast.hindcast.request_wpto_point_data
- wave.io.hindcast.hindcast.request_wpto_directional_spectrum

Usage:
Run the script directly as a standalone program, or import the 
TestWPTOhindcast class in another test suite.
"""

import unittest
from os.path import abspath, dirname, join, normpath
from pandas.testing import assert_frame_equal
import xarray.testing as xrt
import pandas as pd
import mhkit.wave as wave
import xarray as xr

testdir = dirname(abspath(__file__))
datadir = normpath(
    join(testdir, "..", "..", "..", "..", "..", "examples", "data", "wave")
)


class TestWPTOhindcast(unittest.TestCase):
    """
    A test call designed to check the WPTO hindcast retrival
    """

    @classmethod
    def setUpClass(cls):
        """
        Intitialize the WPTO hindcast test with expected data
        """
        cls.sy_swh = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast.csv"),
            index_col="time_index",
            names=["time_index", "significant_wave_height_0"],
            header=0,
            dtype={"significant_wave_height_0": "float32"},
        )
        cls.sy_swh.index = pd.to_datetime(cls.sy_swh.index)

        cls.sy_meta = pd.read_csv(
            join(datadir, "hindcast/single_year_meta.csv"),
            names=[
                "water_depth",
                "latitude",
                "longitude",
                "distance_to_shore",
                "timezone",
                "jurisdiction",
                "gid",
            ],
            header=0,
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

        cls.my_odwp = pd.read_csv(
            join(datadir, "hindcast/multi_year_hindcast.csv"),
            index_col="time_index",
            names=["time_index", "omni-directional_wave_power_0"],
            header=0,
            dtype={"omni-directional_wave_power_0": "float32"},
        )
        cls.my_odwp.index = pd.to_datetime(cls.my_odwp.index)

        cls.my_odwp_meta = pd.read_csv(
            join(datadir, "hindcast/multi_year_hindcast_meta.csv"),
            names=[
                "water_depth",
                "latitude",
                "longitude",
                "distance_to_shore",
                "timezone",
                "jurisdiction",
                "gid",
            ],
            header=0,
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

        cls.ml = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast_multiloc.csv"),
            index_col="time_index",
            names=["time_index", "energy_period_0", "energy_period_1"],
            header=0,
            dtype={
                "energy_period_0": "float32",
                "energy_period_1": "float32",
            },
        )
        cls.ml.index = pd.to_datetime(cls.ml.index)

        cls.ml_meta = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast_multiloc_meta.csv"),
            index_col=0,
            names=[
                None,
                "water_depth",
                "latitude",
                "longitude",
                "distance_to_shore",
                "timezone",
                "jurisdiction",
                "gid",
            ],
            header=0,
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

        cls.mp = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast_multiparam.csv"),
            index_col="time_index",
            names=[
                "time_index",
                "significant_wave_height_0",
                "peak_period_0",
                "mean_wave_direction_0",
            ],
            header=0,
            dtype={
                "significant_wave_height_0": "float32",
                "peak_period_0": "float32",
                "mean_wave_direction_0": "float32",
            },
        )
        cls.mp.index = pd.to_datetime(cls.mp.index)

        cls.mp_meta = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast_multiparam_meta.csv"),
            index_col=0,
            names=[
                None,
                "water_depth",
                "latitude",
                "longitude",
                "distance_to_shore",
                "timezone",
                "jurisdiction",
                "gid",
            ],
            header=0,
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

        cls.dir_spectra_meta = pd.read_csv(
            join(datadir, "hindcast/dir_spectra_meta.csv"),
            index_col=0,
            names=[
                None,
                "water_depth",
                "latitude",
                "longitude",
                "distance_to_shore",
                "timezone",
                "jurisdiction",
                "gid",
            ],
            header=0,
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

        cls.dir_spectra = xr.open_dataset(join(datadir, "hindcast/dir_spectra.nc"))

    def test_point_data(self):
        """
        Test request data on a single data_type, lat_lon, and parameter
        """
        data_type = "3-hour"
        years = [1995]
        lat_lon = (44.624076, -124.280097)
        parameter = "significant_wave_height"

        Hs, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameter, lat_lon, years
        )

        assert_frame_equal(self.sy_swh, Hs)
        assert_frame_equal(self.sy_meta, meta)

    def test_multi_loc(self):
        """
        Test mutiple locations on point data and directional spectrum at a
        single data_type, year, and parameter.
        """
        data_type = "3-hour"
        years = [1995]
        lat_lon = ((44.624076, -124.280097), (43.489171, -125.152137))
        parameters = "energy_period"
        wave_multiloc, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameters, lat_lon, years
        )
        assert_frame_equal(self.ml, wave_multiloc)
        assert_frame_equal(self.ml_meta, meta)

    def test_multi_year(self):
        """
        Test multiple years on a single data_type, lat_lon, and parameter
        """
        data_type = "3-hour"
        years = [1995, 1996]
        lat_lon = (44.624076, -124.280097)
        parameters = "omni-directional_wave_power"

        wave_multiyear, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type,
            parameters,
            lat_lon,
            years,
        )

        assert_frame_equal(self.my_odwp, wave_multiyear)
        assert_frame_equal(self.my_odwp_meta, meta)

    def test_multi_parm(self):
        """
        Test multiple parameters on a single data_type, year, and lat_lon
        """
        data_type = "1-hour"
        years = [1995]
        lat_lon = (44.624076, -124.280097)
        parameters = ["significant_wave_height", "peak_period", "mean_wave_direction"]
        wave_multiparm, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameters, lat_lon, years
        )

        assert_frame_equal(self.mp, wave_multiparm)
        assert_frame_equal(self.mp_meta, meta)

    def test_request_directional_spectrum(self):
        """
        Test `request_wpto_directional_spectrum`. The spectra data will be
        returned as an xarray while the metadata will be returned as a
        Pandas DataFrame.
        """
        year = "1993"  # only one year can be passed at a time as a string
        lat_lon = (43.489171, -125.152137)
        dir_spectra, meta = wave.io.hindcast.hindcast.request_wpto_directional_spectrum(
            lat_lon, year
        )

        # Down sample
        dir_spectra.isel(
            time_index=slice(0, 100),  # Select the first 100 time steps
            frequency=slice(0, 10),  # Select the first 10 frequencies
            direction=slice(0, 8),  # Select the first 8 directions
        )

        assert dir_spectra.equals(
            dir_spectra
        ), "The directional spectrum datasets are not equal"
        assert_frame_equal(self.dir_spectra_meta, meta)


if __name__ == "__main__":
    unittest.main()
