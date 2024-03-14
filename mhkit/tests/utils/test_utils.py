from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
import mhkit.utils as utils
import pandas as pd
import numpy as np
import unittest
import json
import xarray as xr


testdir = dirname(abspath(__file__))
loads_datadir = normpath(join(testdir, relpath("../../../examples/data/loads")))


class TestGenUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        loads_data_file = join(loads_datadir, "loads_data_dict.json")
        with open(loads_data_file, "r") as fp:
            data_dict = json.load(fp)
        # convert dictionaries into dataframes
        data = {key: pd.DataFrame(data_dict[key]) for key in data_dict}
        self.data = data

        self.freq = 50  # Hz
        self.period = 600  # seconds

    def test_get_statistics(self):
        # load in file
        df = self.data["loads"]
        df.Timestamp = pd.to_datetime(df.Timestamp)
        df.set_index("Timestamp", inplace=True)
        # run function
        means, maxs, mins, stdevs = utils.get_statistics(
            df,
            self.freq,
            period=self.period,
            vector_channels=["WD_Nacelle", "WD_NacelleMod"],
        )
        # check statistics
        self.assertAlmostEqual(
            means.reset_index().loc[0, "uWind_80m"], 7.773, 2
        )  # mean
        self.assertAlmostEqual(maxs.reset_index().loc[0, "uWind_80m"], 13.271, 2)  # max
        self.assertAlmostEqual(mins.reset_index().loc[0, "uWind_80m"], 3.221, 2)  # min
        self.assertAlmostEqual(
            stdevs.reset_index().loc[0, "uWind_80m"], 1.551, 2
        )  # standard deviation
        self.assertAlmostEqual(
            means.reset_index().loc[0, "WD_Nacelle"], 178.1796, 2
        )  # mean - vector
        self.assertAlmostEqual(
            stdevs.reset_index().loc[0, "WD_Nacelle"], 36.093, 2
        )  # standard devaition - vector
        # check timestamp
        string_time = "2017-03-01 01:28:41"
        time = pd.to_datetime(string_time)
        self.assertTrue(means.index[0] == time)

    def test_vector_statistics(self):
        # load in vector variable
        df = self.data["loads"]
        vector_data = df["WD_Nacelle"]
        vector_avg, vector_std = utils.vector_statistics(vector_data)
        # check answers
        self.assertAlmostEqual(vector_avg, 178.1796, 2)  # mean - vector
        self.assertAlmostEqual(vector_std, 36.093, 2)  # standard devaition - vector

    def test_unwrap_vector(self):
        # create array of test values and corresponding expected answers
        test = [-740, -400, -50, 0, 50, 400, 740]
        correct = [340, 320, 310, 0, 50, 40, 20]
        # get answers from function
        answer = utils.unwrap_vector(test)

        # check if answer is correct
        assert_frame_equal(
            pd.DataFrame(answer, dtype="int32"), pd.DataFrame(correct, dtype="int32")
        )

    def test_matlab_to_datetime(self):
        # store matlab timestamp
        mat_time = 7.367554921296296e05
        # corresponding datetime
        string_time = "2017-03-01 11:48:40"
        time = pd.to_datetime(string_time)
        # test function
        answer = utils.matlab_to_datetime(mat_time)
        answer2 = answer.round("s")  # round to nearest second for comparison

        # check if answer is correct
        self.assertTrue(answer2 == time)

    def test_excel_to_datetime(self):
        # store excel timestamp
        excel_time = 4.279549212962963e04
        # corresponding datetime
        string_time = "2017-03-01 11:48:40"
        time = pd.to_datetime(string_time)
        # test function
        answer = utils.excel_to_datetime(excel_time)
        answer2 = answer.round("s")  # round to nearest second for comparison

        # check if answer is correct
        self.assertTrue(answer2 == time)

    def test_magnitude_phase_2D(self):
        # float
        magnitude = 9
        x = y = np.sqrt(1 / 2 * magnitude**2)
        phase = np.arctan2(y, x)
        mag, theta = utils.magnitude_phase(x, y)

        self.assertAlmostEqual(magnitude, mag)
        self.assertAlmostEqual(phase, theta)

        # list
        xx = [x, x]
        yy = [y, y]
        mag, theta = utils.magnitude_phase(xx, yy)
        self.assertTrue(all(mag == magnitude))
        self.assertTrue(all(theta == phase))

        # series
        xs = pd.Series(xx, index=range(len(xx)))
        ys = pd.Series(yy, index=range(len(yy)))

        mag, theta = utils.magnitude_phase(xs, ys)
        self.assertTrue(all(mag == magnitude))
        self.assertTrue(all(theta == phase))

    def test_magnitude_phase_3D(self):
        # float
        magnitude = 9
        x = y = z = np.sqrt(1 / 3 * magnitude**2)
        phase1 = np.arctan2(y, x)
        phase2 = np.arctan2(np.sqrt(x**2 + y**2), z)
        mag, theta, phi = utils.magnitude_phase(x, y, z)

        self.assertAlmostEqual(magnitude, mag)
        self.assertAlmostEqual(phase1, theta)
        self.assertAlmostEqual(phase2, phi)

        # list
        xx = [x, x]
        yy = [y, y]
        zz = [z, z]
        mag, theta, phi = utils.magnitude_phase(xx, yy, zz)
        self.assertTrue(all(mag == magnitude))
        self.assertTrue(all(theta == phase1))
        self.assertTrue(all(phi == phase2))

        # series
        xs = pd.Series(xx, index=range(len(xx)))
        ys = pd.Series(yy, index=range(len(yy)))
        zs = pd.Series(zz, index=range(len(zz)))

        mag, theta, phi = utils.magnitude_phase(xs, ys, zs)
        self.assertTrue(all(mag == magnitude))
        self.assertTrue(all(theta == phase1))
        self.assertTrue(all(phi == phase2))

    def test_convert_to_dataarray(self):
        # test data
        a = 5
        t = np.arange(0.0, 5.0, 0.5)
        i = np.arange(0.0, 10.0, 1)
        d1 = i**2 / 5.0
        d2 = -d1

        # test data formats
        test_n = d1
        test_s = pd.Series(d1, t)
        test_df = pd.DataFrame({"d1": d1}, index=t)
        test_df2 = pd.DataFrame({"d1": d1, "d1_duplicate": d1}, index=t)
        test_da = xr.DataArray(
            data=d1,
            dims="time",
            coords=dict(time=t),
        )
        test_ds = xr.Dataset(
            data_vars={"d1": (["time"], d1)}, coords={"time": t, "index": i}
        )
        test_ds2 = xr.Dataset(
            data_vars={
                "d1": (["time"], d1),
                "d2": (["ind"], d2),
            },
            coords={"time": t, "index": i},
        )

        # numpy
        n = utils.convert_to_dataarray(test_n, "test_data")
        self.assertIsInstance(n, xr.DataArray)
        self.assertTrue(all(n.data == d1))
        self.assertEqual(n.name, "test_data")

        # Series
        s = utils.convert_to_dataarray(test_s)
        self.assertIsInstance(s, xr.DataArray)
        self.assertTrue(all(s.data == d1))

        # DataArray
        da = utils.convert_to_dataarray(test_da)
        self.assertIsInstance(da, xr.DataArray)
        self.assertTrue(all(da.data == d1))

        # Dataframe
        df = utils.convert_to_dataarray(test_df)
        self.assertIsInstance(df, xr.DataArray)
        self.assertTrue(all(df.data == d1))

        # Dataset
        ds = utils.convert_to_dataarray(test_ds)
        self.assertIsInstance(ds, xr.DataArray)
        self.assertTrue(all(ds.data == d1))

        # int (error)
        with self.assertRaises(TypeError):
            utils.convert_to_dataarray(a)

        # non-string name (error)
        with self.assertRaises(TypeError):
            utils.convert_to_dataarray(test_n, 5)

        # Multivariate Dataframe (error)
        with self.assertRaises(ValueError):
            utils.convert_to_dataarray(test_df2)

        # Multivariate Dataset (error)
        with self.assertRaises(ValueError):
            utils.convert_to_dataarray(test_ds2)

    def test_convert_to_dataset(self):
        # test data
        a = 5
        t = np.arange(0, 5, 0.5)
        i = np.arange(0, 10, 1)
        d1 = i**2 / 5.0
        d2 = -d1

        # test data formats
        test_n = d1
        test_s = pd.Series(d1, t)
        test_df2 = pd.DataFrame({"d1": d1, "d2": d2}, index=t)
        test_da = xr.DataArray(
            data=d1,
            dims="time",
            coords=dict(time=t),
        )
        test_ds2 = xr.Dataset(
            data_vars={
                "d1": (["time"], d1),
                "d2": (["ind"], d2),
            },
            coords={"time": t, "index": i},
        )

        # Series
        s = utils.convert_to_dataset(test_s)
        self.assertIsInstance(s, xr.Dataset)
        self.assertTrue(all(s["data"].data == d1))

        # DataArray with custom name
        da = utils.convert_to_dataset(test_da, "test_name")
        self.assertIsInstance(da, xr.Dataset)
        self.assertTrue(all(da["test_name"].data == d1))

        # Dataframe
        df = utils.convert_to_dataset(test_df2)
        self.assertIsInstance(df, xr.Dataset)
        self.assertTrue(all(df["d1"].data == d1))
        self.assertTrue(all(df["d2"].data == d2))

        # Dataset
        ds = utils.convert_to_dataset(test_ds2)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertTrue(all(ds["d1"].data == d1))
        self.assertTrue(all(ds["d2"].data == d2))

        # int (error)
        with self.assertRaises(TypeError):
            utils.convert_to_dataset(a)

        # non-string name (error)
        with self.assertRaises(TypeError):
            utils.convert_to_dataset(test_n, 5)


if __name__ == "__main__":
    unittest.main()
