from os.path import abspath, dirname, join, isfile, normpath
from pandas.testing import assert_frame_equal
import matplotlib.pylab as plt
from datetime import datetime
import mhkit.wave as wave
from io import StringIO
import pandas as pd
import xarray as xr
import numpy as np
import contextlib
import unittest
import os


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir, "..", "..", "..", "..", "examples", "data", "wave"))


class TestIOndbc(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.expected_columns_metRT = [
            "WDIR",
            "WSPD",
            "GST",
            "WVHT",
            "DPD",
            "APD",
            "MWD",
            "PRES",
            "ATMP",
            "WTMP",
            "DEWP",
            "VIS",
            "PTDY",
            "TIDE",
        ]
        self.expected_units_metRT = {
            "WDIR": "degT",
            "WSPD": "m/s",
            "GST": "m/s",
            "WVHT": "m",
            "DPD": "sec",
            "APD": "sec",
            "MWD": "degT",
            "PRES": "hPa",
            "ATMP": "degC",
            "WTMP": "degC",
            "DEWP": "degC",
            "VIS": "nmi",
            "PTDY": "hPa",
            "TIDE": "ft",
        }

        self.expected_columns_metH = [
            "WDIR",
            "WSPD",
            "GST",
            "WVHT",
            "DPD",
            "APD",
            "MWD",
            "PRES",
            "ATMP",
            "WTMP",
            "DEWP",
            "VIS",
            "TIDE",
        ]
        self.expected_units_metH = {
            "WDIR": "degT",
            "WSPD": "m/s",
            "GST": "m/s",
            "WVHT": "m",
            "DPD": "sec",
            "APD": "sec",
            "MWD": "deg",
            "PRES": "hPa",
            "ATMP": "degC",
            "WTMP": "degC",
            "DEWP": "degC",
            "VIS": "nmi",
            "TIDE": "ft",
        }
        self.filenames = ["46042w1996.txt.gz", "46029w1997.txt.gz", "46029w1998.txt.gz"]
        self.swden = pd.read_csv(
            join(datadir, self.filenames[0]), sep=r"\s+", compression="gzip"
        )

        buoy = "42012"
        year = 2021
        date = np.datetime64("2021-02-21T12:40:00")
        directional_data_all = wave.io.ndbc.request_directional_data(buoy, year)
        self.directional_data = directional_data_all.sel(date=date)

    @classmethod
    def tearDownClass(self):
        pass

    # Realtime data
    def test_ndbc_read_realtime_met(self):
        data, units = wave.io.ndbc.read_file(join(datadir, "46097.txt"))
        expected_index0 = datetime(2019, 4, 2, 13, 50)
        self.assertSetEqual(set(data.columns), set(self.expected_columns_metRT))
        self.assertEqual(data.index[0], expected_index0)
        self.assertEqual(data.shape, (6490, 14))
        self.assertEqual(units, self.expected_units_metRT)

    # Historical data
    def test_ndbnc_read_historical_met(self):
        # QC'd monthly data, Aug 2019
        data, units = wave.io.ndbc.read_file(join(datadir, "46097h201908qc.txt"))
        expected_index0 = datetime(2019, 8, 1, 0, 0)
        self.assertSetEqual(set(data.columns), set(self.expected_columns_metH))
        self.assertEqual(data.index[0], expected_index0)
        self.assertEqual(data.shape, (4464, 13))
        self.assertEqual(units, self.expected_units_metH)

    # Spectral data
    def test_ndbc_read_spectral(self):
        data, units = wave.io.ndbc.read_file(join(datadir, "data.txt"), to_pandas=False)
        self.assertEqual(len(data.data_vars), 47)
        self.assertEqual(len(data["dim_0"]), 743)
        self.assertEqual(units, None)

    # Continuous wind data
    def test_ndbc_read_cwind_no_units(self):
        data, units = wave.io.ndbc.read_file(join(datadir, "42a01c2003.txt"))
        self.assertEqual(data.shape, (4320, 5))
        self.assertEqual(units, None)

    def test_ndbc_read_cwind_units(self):
        data, units = wave.io.ndbc.read_file(join(datadir, "46002c2016.txt"))
        self.assertEqual(data.shape, (28468, 5))
        self.assertEqual(units, wave.io.ndbc.parameter_units("cwind"))

    def test_ndbc_available_data(self):
        data = wave.io.ndbc.available_data("swden", buoy_number="46029")
        cols = data.columns.tolist()
        exp_cols = ["id", "year", "filename"]
        self.assertEqual(cols, exp_cols)

        years = [int(year) for year in data.year.tolist()]
        exp_years = [*range(1996, 1996 + len(years))]
        self.assertEqual(years, exp_years)
        self.assertEqual(data.shape, (len(data), 3))

    def test__ndbc_parse_filenames(self):
        filenames = pd.Series(self.filenames)
        buoys = wave.io.ndbc._parse_filenames("swden", filenames)
        years = buoys.year.tolist()
        numbers = buoys.id.tolist()
        fnames = buoys.filename.tolist()

        self.assertEqual(buoys.shape, (len(filenames), 3))
        self.assertListEqual(years, ["1996", "1997", "1998"])
        self.assertListEqual(numbers, ["46042", "46029", "46029"])
        self.assertListEqual(fnames, self.filenames)

    def test_ndbc_request_data(self):
        filenames = pd.Series(self.filenames[0])
        ndbc_data = wave.io.ndbc.request_data("swden", filenames, to_pandas=False)
        self.assertTrue(xr.Dataset(self.swden).equals(ndbc_data["1996"]))

    def test_ndbc_request_data_from_dataframe(self):
        filenames = pd.DataFrame(pd.Series(data=self.filenames[0]))
        ndbc_data = wave.io.ndbc.request_data("swden", filenames)
        assert_frame_equal(self.swden, ndbc_data["1996"])

    def test_ndbc_request_data_filenames_length(self):
        with self.assertRaises(ValueError):
            wave.io.ndbc.request_data("swden", pd.Series(dtype=float))

    def test_ndbc_to_datetime_index(self):
        dt = wave.io.ndbc.to_datetime_index("swden", self.swden)
        self.assertEqual(type(dt.index), pd.DatetimeIndex)
        self.assertFalse({"YY", "MM", "DD", "hh"}.issubset(dt.columns))

    def test_ndbc_request_data_empty_file(self):
        temp_stdout = StringIO()
        # known empty file. If NDBC replaces, this test may fail.
        filename = "42008h1984.txt.gz"
        buoy_id = "42008"
        year = "1984"
        with contextlib.redirect_stdout(temp_stdout):
            wave.io.ndbc.request_data("stdmet", pd.Series(filename))
        output = temp_stdout.getvalue().strip()
        msg = (
            f"The NDBC buoy {buoy_id} for year {year} with "
            f"filename {filename} is empty or missing "
            "data. Please omit this file from your data "
            "request in the future."
        )
        self.assertEqual(output, msg)

    def test_ndbc_request_multiple_files_with_empty_file(self):
        temp_stdout = StringIO()
        # known empty file. If NDBC replaces, this test may fail.
        empty_file = "42008h1984.txt.gz"
        working_file = "46042h1996.txt.gz"
        filenames = pd.Series([empty_file, working_file])

        with contextlib.redirect_stdout(temp_stdout):
            ndbc_data = wave.io.ndbc.request_data("stdmet", filenames)
        self.assertEqual(1, len(ndbc_data))

    def test_ndbc_dates_to_datetime(self):
        dt = wave.io.ndbc.dates_to_datetime(self.swden)
        self.assertEqual(datetime(1996, 1, 1, 1, 0), dt[1])

    def test_ndbc_date_string_to_datetime(self):
        swden = self.swden.copy(deep=True)
        swden["mm"] = np.zeros(len(swden)).astype(int).astype(str)
        year_string = "YY"
        year_fmt = "%y"
        parse_columns = [year_string, "MM", "DD", "hh", "mm"]
        df = wave.io.ndbc._date_string_to_datetime(swden, parse_columns, year_fmt)
        dt = df["date"]
        self.assertEqual(datetime(1996, 1, 1, 1, 0), dt[1])

    def test_ndbc_parameter_units(self):
        parameter = "swden"
        units = wave.io.ndbc.parameter_units(parameter)
        self.assertEqual(units[parameter], "(m*m)/Hz")

    def test_ndbc_request_directional_data(self):
        data = self.directional_data
        # correct 5 parameters
        self.assertEqual(len(data), 5)
        self.assertIn("swden", data)
        self.assertIn("swdir", data)
        self.assertIn("swdir2", data)
        self.assertIn("swr1", data)
        self.assertIn("swr2", data)
        # correct number of data points
        self.assertEqual(len(data.frequency), 47)

    def test_ndbc_create_spread_function(self):
        directions = np.arange(0, 360, 2.0)
        spread = wave.io.ndbc.create_spread_function(self.directional_data, directions)
        self.assertEqual(spread.shape, (47, 180))
        self.assertEqual(spread.units, "1/Hz/deg")

    def test_ndbc_create_directional_spectrum(self):
        directions = np.arange(0, 360, 2.0)
        spectrum = wave.io.ndbc.create_directional_spectrum(
            self.directional_data, directions
        )
        self.assertEqual(spectrum.shape, (47, 180))
        self.assertEqual(spectrum.units, "m^2/Hz/deg")

    def test_plot_directional_spectrum(self):
        directions = np.arange(0, 360, 2.0)
        spectrum = wave.io.ndbc.create_spread_function(
            self.directional_data, directions
        )
        wave.graphics.plot_directional_spectrum(
            spectrum,
            color_level_min=0.0,
            fill=True,
            nlevels=6,
            name="Elevation Variance",
            units="m^2",
        )

        filename = abspath(join(testdir, "wave_plot_directional_spectrum.png"))
        if isfile(filename):
            os.remove(filename)
        plt.savefig(filename)

        self.assertTrue(isfile(filename))
        os.remove(filename)

    def test_get_buoy_metadata(self):
        metadata = wave.io.ndbc.get_buoy_metadata("46042")
        expected_keys = {
            "buoy",
            "provider",
            "type",
            "SCOOP payload",
            "lat",
            "lon",
            "Site elevation",
            "Air temp height",
            "Anemometer height",
            "Barometer elevation",
            "Sea temp depth",
            "Water depth",
            "Watch circle radius",
        }
        self.assertSetEqual(set(metadata.keys()), expected_keys)
        self.assertEqual(
            metadata["provider"], "Owned and maintained by National Data Buoy Center"
        )
        self.assertEqual(metadata["type"], "3-meter foam buoy w/ seal cage")
        self.assertAlmostEqual(float(metadata["lat"]), 36.785)
        self.assertAlmostEqual(float(metadata["lon"]), 122.396)
        self.assertEqual(metadata["Site elevation"], "sea level")

    def test_get_buoy_metadata_invalid_station(self):
        with self.assertRaises(ValueError):
            wave.io.ndbc.get_buoy_metadata("invalid_station")

    def test_get_buoy_metadata_nonexistent_station(self):
        with self.assertRaises(ValueError):
            wave.io.ndbc.get_buoy_metadata("99999")


if __name__ == "__main__":
    unittest.main()
