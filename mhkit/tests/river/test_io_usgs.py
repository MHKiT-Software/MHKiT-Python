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
        Test request_usgs_data with daily data from the Water Data API
        """
        days = pd.date_range("2009-08-01", "2009-08-10", freq="D")
        mock_get.side_effect = [
            _response(_features([d.strftime("%Y-%m-%d") for d in days])),
            _response(PARAMETER_CODE),
        ]

        data = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Daily", "clear_cache": True},
        )

        first = mock_get.call_args_list[0].kwargs
        self.assertTrue(first["url"].endswith("/collections/daily/items"))
        self.assertEqual(first["params"]["monitoring_location_id"], "USGS-15515500")
        self.assertEqual(first["params"]["statistic_id"], "00003")
        self.assertEqual(first["params"]["time"], "2009-08-01/2009-08-10")

        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(list(data.columns), ["Discharge, cubic feet per second"])
        self.assertEqual(data.shape, (10, 1))
        self.assertTrue(data.index.tz is not None)  # Timezone aware
        self.assertTrue((data.index == days.tz_localize("UTC")).all())


class TestUSGSInstant(unittest.TestCase):
    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_instant(self, mock_get):
        mock_get.side_effect = [
            _response(
                _features(["2009-08-01T00:00:00+00:00", "2009-08-01T00:15:00+00:00"])
            ),
            _response(PARAMETER_CODE),
        ]

        df = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options={"data_type": "Instantaneous", "clear_cache": True},
        )

        first = mock_get.call_args_list[0].kwargs
        self.assertTrue(first["url"].endswith("/collections/continuous/items"))
        self.assertEqual(
            first["params"]["time"], "2009-08-01T00:00:00Z/2009-08-11T00:00:00Z"
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 1))
        self.assertTrue(df.index.tz is not None)  # Timezone aware

    @patch("mhkit.river.io.usgs.requests.get")
    def test_instantaneous_split_by_year_and_pages_followed(self, mock_get):
        """The continuous collection serves at most three years per query, and
        large results are paged through "next" links"""
        next_page = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items?cursor=abc"
        mock_get.side_effect = [
            _response(_features(["2018-06-01T00:00:00+00:00"], next_link=next_page)),
            _response(_features(["2018-06-01T00:15:00+00:00"])),
            _response(_features(["2019-06-01T00:00:00+00:00"])),
            _response(PARAMETER_CODE),
        ]

        df = river.io.usgs.request_usgs_data(
            station="08313000",
            parameter="00060",
            start_date="2018-01-01",
            end_date="2019-12-31",
            options={
                "data_type": "Instantaneous",
                "clear_cache": True,
                "api_key": "KEY",
            },
        )

        calls = [c.kwargs for c in mock_get.call_args_list]
        self.assertEqual(
            calls[0]["params"]["time"], "2018-01-01T00:00:00Z/2019-01-01T00:00:00Z"
        )
        self.assertEqual(calls[1]["url"], next_page + "&api_key=KEY")
        self.assertEqual(
            calls[2]["params"]["time"], "2019-01-01T00:00:00Z/2020-01-01T00:00:00Z"
        )
        self.assertEqual(calls[0]["params"]["api_key"], "KEY")
        self.assertEqual(df.shape, (3, 1))

    def test_read_water_data_geojson(self):
        """GeoJSON from the Water Data API reads like the legacy JSON"""
        text = _features(["2019-01-02", "2019-01-01", "2019-01-01"])
        text["parameter_description"] = "Discharge, cubic feet per second"
        df = river.io.usgs._read_usgs_json(text)
        self.assertEqual(list(df.columns), ["Discharge, cubic feet per second"])
        self.assertEqual(df.shape, (2, 1))  # duplicate time dropped
        self.assertTrue(df.index.is_monotonic_increasing)

        ds = river.io.usgs._read_usgs_json(text, to_pandas=False)
        self.assertIn("Discharge, cubic feet per second", ds.data_vars)


PARAMETER_CODE = {
    "id": "00060",
    "properties": {"parameter_description": "Discharge, cubic feet per second"},
}


def _features(times, next_link=None):
    page = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "parameter_code": "00060",
                    "time": t,
                    "value": "1000",
                    "approval_status": "Provisional",
                    "qualifier": None,
                },
            }
            for t in times
        ],
        "links": [],
    }
    if next_link:
        page["links"].append({"rel": "next", "href": next_link})
    return page


def _response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


if __name__ == "__main__":
    unittest.main()
