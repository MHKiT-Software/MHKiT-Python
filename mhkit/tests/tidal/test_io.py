"""
test_io.py

This module contains unit tests for the mhkit.tidal.io.noaa module.
It tests the read_noaa_json and request_noaa_data functions for various
input parameters and expected outcomes.

These tests include:
- Reading data from a JSON file
- Requesting NOAA data with basic parameters
- Requesting NOAA data with the write_json parameter
- Requesting NOAA data with invalid date format
- Requesting NOAA data with the end date before the start date
"""

from os.path import abspath, dirname, join, normpath, relpath
import unittest
import os
import json

import numpy as np
import mhkit.tidal as tidal


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/tidal")))


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_load_noaa_data(self):
        """
        Test that the read_noaa_json function reads data from a
        JSON file and returns a DataFrame and metadata with the
        correct shape and columns.
        """
        file_name = join(datadir, "s08010.json")
        data, metadata = tidal.io.noaa.read_noaa_json(file_name)
        self.assertTrue(np.all(data.columns == ["s", "d", "b"]))
        self.assertEqual(data.shape, (18890, 3))
        self.assertEqual(metadata["id"], "s08010")

    def test_load_noaa_data_xarray(self):
        """
        Test that the read_noaa_json function reads data from a
        JSON file and returns a DataFrame and metadata with the
        correct shape and columns.
        """
        file_name = join(datadir, "s08010.json")
        data = tidal.io.noaa.read_noaa_json(file_name, to_pandas=False)
        self.assertTrue(np.all(list(data.variables) == ["index", "s", "d", "b"]))
        self.assertEqual(len(data["index"]), 18890)
        self.assertEqual(data.attrs["id"], "s08010")

    def test_request_noaa_data_basic(self):
        """
        Test the request_noaa_data function with basic input parameters
        and verify that the returned DataFrame and metadata have the
        correct shape and columns.
        """
        data, metadata = tidal.io.noaa.request_noaa_data(
            station="s08010",
            parameter="currents",
            start_date="20180101",
            end_date="20180102",
            proxy=None,
            write_json=None,
        )
        self.assertTrue(np.all(data.columns == ["s", "d", "b"]))
        self.assertEqual(data.shape, (183, 3))
        self.assertEqual(metadata["id"], "s08010")

    def test_request_noaa_data_basic_xarray(self):
        """
        Test the request_noaa_data function with basic input parameters
        and verify that the returned DataFrame and metadata have the
        correct shape and columns.
        """
        data = tidal.io.noaa.request_noaa_data(
            station="s08010",
            parameter="currents",
            start_date="20180101",
            end_date="20180102",
            proxy=None,
            write_json=None,
            to_pandas=False,
        )
        self.assertTrue(np.all(list(data.variables) == ["index", "s", "d", "b"]))
        self.assertEqual(len(data["index"]), 183)
        self.assertEqual(data.attrs["id"], "s08010")

    def test_request_noaa_data_write_json(self):
        """
        Test the request_noaa_data function with the write_json parameter
        and verify that the returned JSON file has the correct structure
        and can be loaded back into a dictionary.
        """
        test_json_file = "test_noaa_data.json"
        _, _ = tidal.io.noaa.request_noaa_data(
            station="s08010",
            parameter="currents",
            start_date="20180101",
            end_date="20180102",
            proxy=None,
            write_json=test_json_file,
        )
        self.assertTrue(os.path.isfile(test_json_file))

        with open(test_json_file) as f:
            loaded_data = json.load(f)

        os.remove(test_json_file)  # Clean up the test JSON file

        self.assertIn("metadata", loaded_data)
        self.assertIn("s", loaded_data["columns"])
        self.assertIn("d", loaded_data["columns"])
        self.assertIn("b", loaded_data["columns"])

    def test_request_noaa_data_invalid_dates(self):
        """
        Test the request_noaa_data function with an invalid date format
        and verify that it raises a ValueError.
        """
        with self.assertRaises(ValueError):
            tidal.io.noaa.request_noaa_data(
                station="s08010",
                parameter="currents",
                start_date="2018-01-01",  # Invalid date format
                end_date="20180102",
                proxy=None,
                write_json=None,
            )

    def test_request_noaa_data_end_before_start(self):
        """
        Test the request_noaa_data function with the end date before
        the start date and verify that it raises a ValueError.
        """
        with self.assertRaises(ValueError):
            tidal.io.noaa.request_noaa_data(
                station="s08010",
                parameter="currents",
                start_date="20180102",
                end_date="20180101",  # End date before start date
                proxy=None,
                write_json=None,
            )


if __name__ == "__main__":
    unittest.main()
