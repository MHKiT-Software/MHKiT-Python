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
        cls.my_swh = pd.read_csv(
            join(datadir, "hindcast/multi_year_hindcast.csv"),
            index_col="time_index",
            names=["time_index", "significant_wave_height_0"],
            header=0,
            dtype={"significant_wave_height_0": "float32"},
        )
        cls.my_swh.index = pd.to_datetime(cls.my_swh.index)

        cls.ml = pd.read_csv(
            join(datadir, "hindcast/single_year_hindcast_multiloc.csv"),
            index_col="time_index",
            names=["time_index", "mean_absolute_period_0", "mean_absolute_period_1"],
            header=0,
            dtype={
                "mean_absolute_period_0": "float32",
                "mean_absolute_period_1": "float32",
            },
        )
        cls.ml.index = pd.to_datetime(cls.ml.index)

        cls.mp = pd.read_csv(
            join(datadir, "hindcast/multiparm.csv"),
            index_col="time_index",
            names=["time_index", "energy_period_87", "mean_zero-crossing_period_87"],
            header=0,
            dtype={
                "energy_period_87": "float32",
                "mean_zero-crossing_period_87": "float32",
            },
        )
        cls.mp.index = pd.to_datetime(cls.mp.index)

        cls.ml_meta = pd.read_csv(
            join(datadir, "hindcast/multiloc_meta.csv"),
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

        cls.my_meta = pd.read_csv(
            join(datadir, "hindcast/multi_year_meta.csv"),
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

        cls.mp_meta = pd.read_csv(
            join(datadir, "hindcast/multiparm_meta.csv"),
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

        cls.multi_year_dir_spectra = xr.open_dataset(
            join(datadir, "hindcast/multi_year_dir_spectra.nc")
        )

        cls.multi_year_dir_spectra_meta = pd.read_csv(
            join(datadir, "hindcast/multi_year_dir_spectra_meta.csv"),
            dtype={
                "water_depth": "float32",
                "latitude": "float32",
                "longitude": "float32",
                "distance_to_shore": "float32",
                "timezone": "int16",
                "gid": "int64",
            },
        )

    def test_multi_year(self):
        """
        Test multiple years on a single data_type, lat_lon, and parameter
        """
        data_type = "3-hour"
        years = [1990, 1992]
        lat_lon = (44.624076, -124.280097)
        parameters = "significant_wave_height"

        wave_multiyear, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameters, lat_lon, years, to_pandas=False
        )
        wave_multiyear_df = (
            wave_multiyear["significant_wave_height_0"]
            .to_dataframe()
            .tz_localize("UTC")
        )

        assert_frame_equal(self.my_swh, wave_multiyear_df)
        assert_frame_equal(self.my_meta, meta)

    def test_multi_parm(self):
        """
        Test multiple parameters on a single data_type, year, and lat_lon
        """
        data_type = "1-hour"
        years = [1996]
        lat_lon = (44.624076, -124.280097)
        parameters = ["energy_period", "mean_zero-crossing_period"]
        wave_multiparm, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameters, lat_lon, years
        )
        assert_frame_equal(self.mp, wave_multiparm)
        assert_frame_equal(self.mp_meta, meta)

    def test_multi_loc(self):
        """
        Test mutiple locations on point data and directional spectrum at a
        single data_type, year, and parameter.
        """
        data_type = "3-hour"
        years = [1995]
        lat_lon = ((44.624076, -124.280097), (43.489171, -125.152137))
        parameters = "mean_absolute_period"
        wave_multiloc, meta = wave.io.hindcast.hindcast.request_wpto_point_data(
            data_type, parameters, lat_lon, years
        )
        (
            dir_multiyear,
            meta_dir,
        ) = wave.io.hindcast.hindcast.request_wpto_directional_spectrum(
            lat_lon, year=str(years[0])
        )

        dir_multiyear = dir_multiyear.sel(
            time_index=slice(dir_multiyear.time_index[0], dir_multiyear.time_index[99])
        )
        # Convert to effcient range index
        meta_dir.index = pd.RangeIndex(start=0, stop=len(meta_dir.index))

        assert_frame_equal(self.ml, wave_multiloc)
        assert_frame_equal(self.ml_meta, meta)
        xrt.assert_allclose(self.multi_year_dir_spectra, dir_multiyear)
        assert_frame_equal(
            self.multi_year_dir_spectra_meta, meta_dir, check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()
