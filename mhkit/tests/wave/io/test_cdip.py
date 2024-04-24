from os.path import abspath, dirname, join, isfile, normpath
import matplotlib.pylab as plt
from datetime import datetime
import mhkit.wave as wave
import unittest
import netCDF4
import pytz
import os


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir, "..", "..", "..", "..", "examples", "data", "wave"))


class TestIOcdip(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        b067_1996 = (
            "http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/"
            + "archive/067p1/067p1_d04.nc"
        )
        self.test_nc = netCDF4.Dataset(b067_1996)

        self.vars2D = [
            "waveEnergyDensity",
            "waveMeanDirection",
            "waveA1Value",
            "waveB1Value",
            "waveA2Value",
            "waveB2Value",
            "waveCheckFactor",
            "waveSpread",
            "waveM2Value",
            "waveN2Value",
        ]

    @classmethod
    def tearDownClass(self):
        pass

    def test_validate_date(self):
        date = "2013-11-12"
        start_date = wave.io.cdip._validate_date(date)
        assert isinstance(start_date, datetime)

        date = "11-12-2012"
        self.assertRaises(ValueError, wave.io.cdip._validate_date, date)

    def test_request_netCDF_historic(self):
        station_number = "067"
        nc = wave.io.cdip.request_netCDF(station_number, "historic")
        isinstance(nc, netCDF4.Dataset)

    def test_request_netCDF_realtime(self):
        station_number = "067"
        nc = wave.io.cdip.request_netCDF(station_number, "realtime")
        isinstance(nc, netCDF4.Dataset)

    def test_start_and_end_of_year(self):
        year = 2020
        start_day, end_day = wave.io.cdip._start_and_end_of_year(year)

        assert isinstance(start_day, datetime)
        assert isinstance(end_day, datetime)

        expected_start = datetime(year, 1, 1)
        expected_end = datetime(year, 12, 31)

        self.assertEqual(start_day, expected_start)
        self.assertEqual(end_day, expected_end)

    def test_dates_to_timestamp(self):
        start_date = datetime(1996, 10, 2, tzinfo=pytz.UTC)
        end_date = datetime(1996, 10, 20, tzinfo=pytz.UTC)

        start_stamp, end_stamp = wave.io.cdip._dates_to_timestamp(
            self.test_nc, start_date=start_date, end_date=end_date
        )

        start_dt = datetime.utcfromtimestamp(start_stamp).replace(tzinfo=pytz.UTC)
        end_dt = datetime.utcfromtimestamp(end_stamp).replace(tzinfo=pytz.UTC)

        self.assertEqual(start_dt, start_date)
        self.assertEqual(end_dt, end_date)

    def test_get_netcdf_variables_all2Dvars(self):
        data = wave.io.cdip.get_netcdf_variables(
            self.test_nc, all_2D_variables=True, to_pandas=False
        )
        returned_keys = [key for key in data["data"]["wave2D"].keys()]
        self.assertTrue(set(returned_keys) == set(self.vars2D))

    def test_get_netcdf_variables_params(self):
        parameters = ["waveHs", "waveTp", "notParam", "waveMeanDirection"]
        data = wave.io.cdip.get_netcdf_variables(self.test_nc, parameters=parameters)

        returned_keys_1D = set([key for key in data["data"]["wave"].keys()])
        returned_keys_2D = [key for key in data["data"]["wave2D"].keys()]
        returned_keys_metadata = [key for key in data["metadata"]["wave"]]

        self.assertTrue(returned_keys_1D == set(["waveHs", "waveTp"]))
        self.assertTrue(returned_keys_2D == ["waveMeanDirection"])
        self.assertTrue(returned_keys_metadata == ["waveFrequency"])

    def test_get_netcdf_variables_time_slice(self):
        start_date = "1996-10-01"
        end_date = "1996-10-31"

        data = wave.io.cdip.get_netcdf_variables(
            self.test_nc, start_date=start_date, end_date=end_date, parameters="waveHs"
        )

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        self.assertTrue(data["data"]["wave"].index[-1] < end_dt)
        self.assertTrue(data["data"]["wave"].index[0] > start_dt)

    def test_request_parse_workflow_multiyear(self):
        station_number = "067"
        year1 = 2011
        year2 = 2013
        years = [year1, year2]
        parameters = ["waveHs", "waveMeanDirection", "waveA1Value"]
        data = wave.io.cdip.request_parse_workflow(
            station_number=station_number, years=years, parameters=parameters
        )

        expected_index0 = datetime(year1, 1, 1)
        expected_index_final = datetime(year2, 12, 31)

        wave1D = data["data"]["wave"]
        self.assertEqual(wave1D.index[0].floor("d").to_pydatetime(), expected_index0)

        self.assertEqual(
            wave1D.index[-1].floor("d").to_pydatetime(), expected_index_final
        )

        for key, wave2D in data["data"]["wave2D"].items():
            self.assertEqual(
                wave2D.index[0].floor("d").to_pydatetime(), expected_index0
            )
            self.assertEqual(
                wave2D.index[-1].floor("d").to_pydatetime(), expected_index_final
            )

    def test_plot_boxplot(self):
        filename = abspath(join(testdir, "wave_plot_boxplot.png"))
        if isfile(filename):
            os.remove(filename)

        station_number = "067"
        year = 2011
        data = wave.io.cdip.request_parse_workflow(
            station_number=station_number,
            years=year,
            parameters=["waveHs"],
            all_2D_variables=False,
        )

        plt.figure()
        wave.graphics.plot_boxplot(data["data"]["wave"]["waveHs"])
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))
        os.remove(filename)

    def test_plot_compendium(self):
        filename = abspath(join(testdir, "wave_plot_boxplot.png"))
        if isfile(filename):
            os.remove(filename)

        station_number = "067"
        year = 2011
        data = wave.io.cdip.request_parse_workflow(
            station_number=station_number,
            years=year,
            parameters=["waveHs", "waveTp", "waveDp"],
            all_2D_variables=False,
        )

        plt.figure()
        wave.graphics.plot_compendium(
            data["data"]["wave"]["waveHs"],
            data["data"]["wave"]["waveTp"],
            data["data"]["wave"]["waveDp"],
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
