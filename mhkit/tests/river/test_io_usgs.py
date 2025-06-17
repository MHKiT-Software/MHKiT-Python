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

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_daily(self, mock_get):
        """
        Test request_usgs_data with daily data
        """
        # Prepare the mocked HTTP response for daily data
        daily_values = []
        start = pd.Timestamp("2009-08-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2009-08-10 23:59:59", tz="UTC")
        current = start
        while current <= end:
            daily_values.append(
                {
                    "dateTime": current.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "value": "1000",
                    "qualifiers": ["P"],
                }
            )
            current += pd.Timedelta(days=1)

        mock_payload = {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "variableDescription": "Discharge, cubic feet per second"
                        },
                        "values": [{"value": daily_values}],
                    }
                ]
            }
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(mock_payload)
        mock_get.return_value = mock_resp

        data = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Daily", "clear_cache": True},
        )

        # Verify that we called requests.get
        mock_get.assert_called_once()

        # Basic functionality checks
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)  # Has data
        self.assertTrue(data.index.tz is not None)  # Timezone aware


class TestUSGSInstant(unittest.TestCase):
    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_instant(self, mock_get):
        mock_payload = {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "variableDescription": "Discharge, cubic feet per second"
                        },
                        "values": [
                            {
                                "value": [
                                    {
                                        "dateTime": "2009-08-01T00:00:00.000Z",
                                        "value": "1000",
                                        "qualifiers": ["P"],
                                    },
                                    {
                                        "dateTime": "2009-08-01T00:15:00.000Z",
                                        "value": "1000",
                                        "qualifiers": ["P"],
                                    },
                                ]
                            }
                        ],
                    }
                ]
            }
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(mock_payload)
        mock_get.return_value = mock_resp

        df = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Instantaneous", "clear_cache": True},
        )

        # Verify that we called requests.get
        mock_get.assert_called_once()

        # Basic functionality checks
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)  # Has data
        self.assertTrue(df.index.tz is not None)  # Timezone aware


if __name__ == "__main__":
    unittest.main()
