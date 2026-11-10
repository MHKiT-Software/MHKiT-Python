from os.path import abspath, dirname, join, isfile, normpath, relpath
import mhkit.river as river
import pandas as pd
import xarray as xr
import unittest
import os
from unittest.mock import patch, MagicMock
import json
import warnings
import requests
import shutil
from datetime import timezone

testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "river"))

PARAMETER_CODE_RESPONSE = {
    "properties": {
        "id": "00060",
        "parameter_description": "Discharge, cubic feet per second",
    }
}


def _mock_response(payload):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(payload)
    return mock_resp


def _feature(time, value="1000"):
    return {
        "type": "Feature",
        "properties": {
            "time_series_id": "abc",
            "monitoring_location_id": "USGS-15515500",
            "parameter_code": "00060",
            "time": time,
            "value": value,
            "unit_of_measure": "ft^3/s",
            "approval_status": "Provisional",
            "qualifier": None,
        },
    }


def _feature_collection(features, next_url=None):
    links = [{"rel": "self", "href": "https://example.com/self"}]
    if next_url:
        links.append({"rel": "next", "href": next_url})
    return {"type": "FeatureCollection", "features": features, "links": links}


def _data_responses(features):
    return [
        _mock_response(_feature_collection(features)),
        _mock_response(PARAMETER_CODE_RESPONSE),
    ]


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

    def test_time_windows(self):
        # 2009-01-01 to 2013-12-31 is 1826 days, which needs two requests
        self.assertEqual(
            river.io.usgs._time_windows("2009-01-01", "2013-12-31", "Instantaneous"),
            [
                ("2009-01-01T00:00:00Z", "2012-01-04T23:59:59Z"),
                ("2012-01-05T00:00:00Z", "2013-12-31T23:59:59Z"),
            ],
        )
        # Local days are converted to UTC, including daylight saving time
        self.assertEqual(
            river.io.usgs._time_windows(
                "2019-01-01", "2019-07-01", "Instantaneous", "America/Denver"
            ),
            [("2019-01-01T07:00:00Z", "2019-07-02T05:59:59Z")],
        )
        self.assertEqual(
            river.io.usgs._time_windows(
                "2019-01-01", "2019-01-31", "Daily", "America/Denver"
            ),
            [("2019-01-01", "2019-01-31")],
        )
        # Midnight does not exist on this day in Santiago
        with self.assertWarns(UserWarning):
            river.io.usgs._time_windows(
                "2019-09-08", "2019-09-08", "Instantaneous", "America/Santiago"
            )


class TestUSGSRequest(unittest.TestCase):
    def setUp(self):
        # Keep mocked responses out of the user's cache
        patcher = patch(
            "mhkit.river.io.usgs.handle_caching", return_value=(None, None, None)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, mock_get, responses, options=None):
        mock_get.side_effect = responses
        return river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            options=options,
        )

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_daily(self, mock_get):
        """
        Test request_usgs_data with daily data
        """
        dates = pd.date_range("2009-08-01", "2009-08-10", freq="D")
        # The API does not return features in time order
        features = [_feature(d.strftime("%Y-%m-%d")) for d in reversed(dates)]

        data = self._request(
            mock_get, _data_responses(features), {"data_type": "Daily"}
        )

        self.assertEqual(mock_get.call_count, 2)
        _, kwargs = mock_get.call_args_list[0]
        self.assertEqual(
            kwargs["url"],
            "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items",
        )
        self.assertEqual(kwargs["params"]["monitoring_location_id"], "USGS-15515500")
        self.assertEqual(kwargs["params"]["statistic_id"], "00003")
        self.assertEqual(kwargs["params"]["time"], "2009-08-01/2009-08-10")

        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(list(data.columns), ["Discharge, cubic feet per second"])
        self.assertEqual(len(data), 10)
        self.assertTrue(data.index.tz is not None)  # Timezone aware
        self.assertTrue(data.index.is_monotonic_increasing)

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_instant(self, mock_get):
        features = [
            _feature("2009-08-01T00:00:00+00:00"),
            _feature("2009-08-01T00:15:00+00:00"),
        ]

        ds = self._request(
            mock_get,
            _data_responses(features),
            {"data_type": "Instantaneous", "to_pandas": False},
        )

        _, kwargs = mock_get.call_args_list[0]
        self.assertEqual(
            kwargs["url"],
            "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items",
        )
        self.assertNotIn("statistic_id", kwargs["params"])
        self.assertEqual(
            kwargs["params"]["time"], "2009-08-01T00:00:00Z/2009-08-10T23:59:59Z"
        )

        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(ds["Discharge, cubic feet per second"].size, 2)

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_pagination(self, mock_get):
        next_url = "https://api.waterdata.usgs.gov/next-page"
        data = self._request(
            mock_get,
            [
                _mock_response(_feature_collection([_feature("2009-08-01")], next_url)),
                _mock_response(_feature_collection([_feature("2009-08-02")])),
                _mock_response(PARAMETER_CODE_RESPONSE),
            ],
        )

        self.assertEqual(mock_get.call_count, 3)
        _, kwargs = mock_get.call_args_list[1]
        self.assertEqual(kwargs["url"], next_url)
        self.assertIsNone(kwargs["params"])
        self.assertEqual(len(data), 2)

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_api_key(self, mock_get):
        features = [_feature("2009-08-01")]

        with patch.dict(os.environ, {"USGS_API_KEY": "env-key"}):
            self._request(mock_get, _data_responses(features))
            for _, kwargs in mock_get.call_args_list:
                self.assertEqual(kwargs["headers"]["X-Api-Key"], "env-key")

            # The api_key option takes precedence and is not sent in the URL
            mock_get.reset_mock()
            self._request(
                mock_get, _data_responses(features), {"api_key": "option-key"}
            )
            for _, kwargs in mock_get.call_args_list:
                self.assertEqual(kwargs["headers"]["X-Api-Key"], "option-key")
                self.assertNotIn("api_key", kwargs["params"])

    @patch("mhkit.river.io.usgs.requests.get")
    def test_request_usgs_data_legacy(self, mock_get):
        payload = {
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
                                        "dateTime": "2009-08-01T00:00:00.000",
                                        "value": "1000",
                                        "qualifiers": ["A"],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        }
        with self.assertWarns(FutureWarning):
            data = self._request(
                mock_get,
                [_mock_response(payload)],
                {"use_legacy_api": True, "data_type": "Instantaneous"},
            )

        _, kwargs = mock_get.call_args_list[-1]
        self.assertEqual(kwargs["url"], "https://waterservices.usgs.gov/nwis/iv/")
        self.assertEqual(kwargs["params"]["sites"], "15515500")
        self.assertEqual(list(data.columns), ["Discharge, cubic feet per second"])
        self.assertEqual(len(data), 1)

    def test_request_usgs_data_legacy_unsupported_options(self):
        for options in ({"include_status": True}, {"timezone": "America/Anchorage"}):
            with self.assertRaises(ValueError):
                river.io.usgs.request_usgs_data(
                    "15515500",
                    "00060",
                    "2009-08-01",
                    "2009-08-10",
                    options={"use_legacy_api": True, **options},
                )


class TestUSGSParity(unittest.TestCase):
    """
    Compares the OGC Water Data APIs against the legacy WaterServices using the
    live services. Skipped if either service is unavailable. Remove with the
    use_legacy_api option.
    """

    def _compare(self, start, end, options, ogc_options=None):
        options = {"clear_cache": True, **options}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                legacy = river.io.usgs.request_usgs_data(
                    "15515500",
                    "00060",
                    start,
                    end,
                    options={
                        "use_legacy_api": True,
                        **options,
                    },
                )
            ogc = river.io.usgs.request_usgs_data(
                "15515500",
                "00060",
                start,
                end,
                options={**options, **(ogc_options or {})},
            )
        except (requests.exceptions.RequestException, ValueError) as e:
            self.skipTest(f"USGS service unavailable: {e}")

        pd.testing.assert_frame_equal(ogc, legacy, check_freq=False)

    def test_parity_daily(self):
        self._compare("2009-08-01", "2009-08-31", {"data_type": "Daily"})

    def test_parity_instantaneous(self):
        # Legacy instantaneous dates are station local days (Alaska)
        self._compare(
            "2009-08-01",
            "2009-08-03",
            {"data_type": "Instantaneous"},
            ogc_options={"timezone": "America/Anchorage"},
        )


if __name__ == "__main__":
    unittest.main()
