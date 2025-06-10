from os.path import abspath, dirname, join, isfile, normpath, relpath
import mhkit.river as river
import pandas as pd
import unittest
import os
from unittest.mock import patch, MagicMock
import json
import shutil
from datetime import timezone


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "river"))


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_load_usgs_data_instantaneous(self):
        file_name = join(datadir, "USGS_08313000_Jan2019_instantaneous.json")
        data = river.io.usgs.read_usgs_file(file_name)

        self.assertEqual(data.columns, ["Discharge, cubic feet per second"])
        self.assertEqual(data.shape, (2972, 1))  # 4 data points are missing

    def test_load_usgs_data_daily(self):
        file_name = join(datadir, "USGS_08313000_Jan2019_daily.json")
        data = river.io.usgs.read_usgs_file(file_name)

        expected_index = pd.date_range("2019-01-01", "2019-01-31", freq="D")
        self.assertEqual(data.columns, ["Discharge, cubic feet per second"])
        self.assertEqual((data.index == expected_index.tz_localize("UTC")).all(), True)
        self.assertEqual(data.shape, (31, 1))

    def test_request_usgs_data_daily(self):
        """
        Test request_usgs_data with daily data
        """
        data = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Daily"},
        )
        self.assertEqual(data.columns, ["Discharge, cubic feet per second"])
        self.assertEqual(data.shape, (10, 1))


class TestUSGSInstant(unittest.TestCase):
    def setUp(self):
        # Build the 15-minute interval payload from 2009-08-01 to 2009-08-10
        start = pd.Timestamp("2009-08-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2009-08-10 23:45:00", tz="UTC")
        current = start
        values = []
        while current <= end:
            values.append(
                {
                    "dateTime": current.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "value": "1000",
                    "qualifiers": ["P"],
                }
            )
            current += pd.Timedelta(minutes=15)

        self.mock_payload = {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "variableDescription": "Discharge, cubic feet per second"
                        },
                        "values": [{"value": values}],
                    }
                ]
            }
        }

        # Clear cache so that clear_cache=True actually removes any stored files
        self.cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "mhkit", "usgs"
        )
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_instant(self, mock_get):
        # Prepare the mocked HTTP response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(self.mock_payload)
        mock_get.return_value = mock_resp

        # Call the function under test
        df = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Instantaneous", "clear_cache": True},
        )

        # Verify that we  called requests.get
        mock_get.assert_called_once()
        called_url = mock_get.call_args.kwargs["url"]

        self.assertIn("nwis/iv", called_url)
        self.assertIn("15515500", called_url)
        self.assertIn("00060", called_url)
        self.assertIn("2009-08-01", called_url)
        self.assertIn("2009-08-10", called_url)

        # Column name should match the variableDescription in the JSON
        self.assertListEqual(list(df.columns), ["Discharge, cubic feet per second"])
        # 15-minute intervals over 10 days -> 4 * 24 * 10 = 960 rows
        self.assertEqual(df.shape, (960, 1))
        # Index should be tz-aware UTC
        self.assertTrue(df.index.tz is not None)
        # Check if the timezone is UTC by comparing offset
        self.assertEqual(df.index.tz.utcoffset(None), timezone.utc.utcoffset(None))


if __name__ == "__main__":
    unittest.main()
