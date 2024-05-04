from os.path import abspath, dirname, join, isfile, normpath, relpath
import mhkit.river as river
import pandas as pd
import unittest
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
        data = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            data_type="Daily",
        )
        self.assertEqual(data.columns, ["Discharge, cubic feet per second"])
        self.assertEqual(data.shape, (10, 1))

    def test_request_usgs_data_instant(self):
        data = river.io.usgs.request_usgs_data(
            station="15515500",
            parameter="00060",
            start_date="2009-08-01",
            end_date="2009-08-10",
            data_type="Instantaneous",
        )
        self.assertEqual(data.columns, ["Discharge, cubic feet per second"])
        # Every 15 minutes or 4 times per hour
        self.assertEqual(data.shape, (10 * 24 * 4, 1))


if __name__ == "__main__":
    unittest.main()
