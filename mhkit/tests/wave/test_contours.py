from os.path import abspath, dirname, join, isfile, normpath, relpath
from numpy.testing import assert_allclose
import matplotlib.pylab as plt
import mhkit.wave as wave
import pandas as pd
import numpy as np
import warnings
import unittest
import pickle
import json
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, relpath("../../../examples/data/wave")))


class TestContours(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        f_name = "Hm0_Te_46022.json"
        self.Hm0Te = pd.read_json(join(datadir, f_name))

        file_loc = join(datadir, "principal_component_analysis.pkl")
        with open(file_loc, "rb") as f:
            self.pca = pickle.load(f)
        f.close()

        file_loc = join(datadir, "WDRT_caluculated_countours.json")
        with open(file_loc) as f:
            self.wdrt_copulas = json.load(f)
        f.close()

        ndbc_46050 = pd.read_csv(join(datadir, "NDBC46050.csv"))
        self.wdrt_Hm0 = ndbc_46050["Hm0"]
        self.wdrt_Te = ndbc_46050["Te"]

        self.wdrt_dt = 3600
        self.wdrt_period = 50

    @classmethod
    def tearDownClass(self):
        pass

    def test_environmental_contour(self):
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values

        dt_ss = (Hm0Te.index[2] - Hm0Te.index[1]).seconds
        period = 100

        copula = wave.contours.environmental_contours(Hm0, Te, dt_ss, period, "PCA")

        Hm0_contour = copula["PCA_x1"]
        Te_contour = copula["PCA_x2"]

        file_loc = join(datadir, "Hm0_Te_contours_46022.csv")
        expected_contours = pd.read_csv(file_loc)
        assert_allclose(expected_contours.Hm0_contour.values, Hm0_contour, rtol=1e-3)

    def test_environmental_contours_invalid_inputs(self):
        # Invalid x1 tests
        x1_non_numeric = "not an array"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                x1_non_numeric, self.wdrt_Te, 3600, 50, "PCA"
            )

        x1_scalar = 5
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                x1_scalar, self.wdrt_Te, 3600, 50, "PCA"
            )

        # Invalid x2 tests
        x2_non_numeric = "not an array"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, x2_non_numeric, 3600, 50, "PCA"
            )

        x2_scalar = 10
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, x2_scalar, 3600, 50, "PCA"
            )

        # Unequal lengths of x1 and x2
        x2_unequal_length = self.wdrt_Te[:-1]
        with self.assertRaises(ValueError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, x2_unequal_length, 3600, 50, "PCA"
            )

        # Invalid sea_state_duration tests
        invalid_sea_state_duration_string = "one hour"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                invalid_sea_state_duration_string,
                50,
                "PCA",
            )

        invalid_sea_state_duration_list = [3600]
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, invalid_sea_state_duration_list, 50, "PCA"
            )

        # Invalid return_period tests
        invalid_return_period_string = "fifty years"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, invalid_return_period_string, "PCA"
            )

        invalid_return_period_list = [50]
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, invalid_return_period_list, "PCA"
            )

        # Invalid method tests
        invalid_method = 123
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, invalid_method
            )

        invalid_bin_val_size = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "PCA",
                bin_val_size=invalid_bin_val_size,
            )

        invalid_nb_steps = 100.5
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, "PCA", nb_steps=invalid_nb_steps
            )

        invalid_initial_bin_max_val = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "PCA",
                initial_bin_max_val=invalid_initial_bin_max_val,
            )

        invalid_min_bin_count = 40.5
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "PCA",
                min_bin_count=invalid_min_bin_count,
            )

        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, "bivariate_KDE"
            )

        invalid_PCA = "not a dict"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, "PCA", PCA=invalid_PCA
            )

        invalid_PCA_bin_size = "not an int"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "PCA",
                PCA_bin_size=invalid_PCA_bin_size,
            )

        invalid_return_fit = "not a boolean"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "PCA",
                return_fit=invalid_return_fit,
            )

        invalid_Ndata_bivariate_KDE = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "bivariate_KDE",
                Ndata_bivariate_KDE=invalid_Ndata_bivariate_KDE,
            )

        invalid_max_x1 = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, "PCA", max_x1=invalid_max_x1
            )

        invalid_max_x2 = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0, self.wdrt_Te, 3600, 50, "PCA", max_x2=invalid_max_x2
            )

        invalid_bandwidth = "not a number"
        with self.assertRaises(TypeError):
            wave.contours.environmental_contours(
                self.wdrt_Hm0,
                self.wdrt_Te,
                3600,
                50,
                "bivariate_KDE",
                bandwidth=invalid_bandwidth,
            )

    def test_PCA_contours_invalid_inputs(self):
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values

        dt_ss = (Hm0Te.index[2] - Hm0Te.index[1]).seconds
        period = 100

        copula = wave.contours.environmental_contours(
            Hm0, Te, dt_ss, period, "PCA", return_fit=True
        )

        PCA_args = {
            "nb_steps": 1000,
            "return_fit": False,
            "bin_size": 250,
        }

        # Invalid x1 tests
        x1_non_numeric = "not an array"
        with self.assertRaises(TypeError):
            wave.contours.PCA_contour(
                x1_non_numeric, self.wdrt_Te, copula["PCA_fit"], PCA_args
            )

        x1_scalar = 5
        with self.assertRaises(TypeError):
            wave.contours.PCA_contour(
                x1_scalar, self.wdrt_Te, copula["PCA_fit"], PCA_args
            )

        # Invalid x2 tests
        x2_non_numeric = "not an array"
        with self.assertRaises(TypeError):
            wave.contours.PCA_contour(
                self.wdrt_Hm0, x2_non_numeric, copula["PCA_fit"], PCA_args
            )

        x2_scalar = 10
        with self.assertRaises(TypeError):
            wave.contours.PCA_contour(
                self.wdrt_Hm0, x2_scalar, copula["PCA_fit"], PCA_args
            )

        # Unequal lengths of x1 and x2
        x2_unequal_length = self.wdrt_Te[:-1]
        with self.assertRaises(ValueError):
            wave.contours.PCA_contour(
                self.wdrt_Hm0, x2_unequal_length, copula["PCA_fit"], PCA_args
            )

    def test__principal_component_analysis(self):
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values
        PCA = wave.contours._principal_component_analysis(Hm0, Te, bin_size=250)

        assert_allclose(PCA["principal_axes"], self.pca["principal_axes"])
        self.assertAlmostEqual(PCA["shift"], self.pca["shift"])
        self.assertAlmostEqual(PCA["x1_fit"]["mu"], self.pca["x1_fit"]["mu"])
        self.assertAlmostEqual(PCA["mu_fit"].slope, self.pca["mu_fit"].slope)
        self.assertAlmostEqual(PCA["mu_fit"].intercept, self.pca["mu_fit"].intercept)
        assert_allclose(PCA["sigma_fit"]["x"], self.pca["sigma_fit"]["x"])

    def test__principal_component_analysis_invalid_inputs(self):
        x1_valid = np.array([1, 2, 3])
        x2_valid = np.array([1, 2, 3])

        # Test invalid x1 (non-array input)
        x1_non_array = "not an array"
        with self.assertRaises(TypeError):
            wave.contours._principal_component_analysis(x1_non_array, x2_valid)

        # Test invalid x2 (non-array input)
        x2_non_array = "not an array"
        with self.assertRaises(TypeError):
            wave.contours._principal_component_analysis(x1_valid, x2_non_array)

        # Test invalid bin_size (non-integer input)
        invalid_bin_size = "not an integer"
        with self.assertRaises(TypeError):
            wave.contours._principal_component_analysis(
                x1_valid, x2_valid, bin_size=invalid_bin_size
            )

    def test_principal_component_analysis_bin_size_adjustment_warning(self):
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values

        large_bin_size = 1000000

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Cause all warnings to always be triggered
            wave.contours._principal_component_analysis(
                Hm0, Te, bin_size=large_bin_size
            )

            self.assertTrue(len(w) == 1)  # Check that exactly one warning was raised
            self.assertTrue(
                issubclass(w[-1].category, UserWarning)
            )  # Check the warning category
            self.assertIn(
                "To allow for a minimum of 4 bins, the bin size has been set to",
                str(w[-1].message),
            )

    def test_plot_environmental_contour(self):
        file_loc = join(plotdir, "wave_plot_environmental_contour.png")
        filename = abspath(file_loc)
        if isfile(filename):
            os.remove(filename)

        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values

        dt_ss = (Hm0Te.index[2] - Hm0Te.index[1]).seconds
        time_R = 100

        copulas = wave.contours.environmental_contours(Hm0, Te, dt_ss, time_R, "PCA")

        Hm0_contour = copulas["PCA_x1"]
        Te_contour = copulas["PCA_x2"]

        dt_ss = (Hm0Te.index[2] - Hm0Te.index[1]).seconds
        time_R = 100

        plt.figure()
        (
            wave.graphics.plot_environmental_contour(
                Te,
                Hm0,
                Te_contour,
                Hm0_contour,
                data_label="NDBC 46022",
                contour_label="100-year Contour",
                x_label="Te [s]",
                y_label="Hm0 [m]",
            )
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_environmental_contour_multiyear(self):
        filename = abspath(
            join(plotdir, "wave_plot_environmental_contour_multiyear.png")
        )
        if isfile(filename):
            os.remove(filename)

        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te["Hm0"] < 20]

        Hm0 = df.Hm0.values
        Te = df.Te.values

        dt_ss = (Hm0Te.index[2] - Hm0Te.index[1]).seconds

        time_R = [100, 105, 110, 120, 150]

        Hm0s = []
        Tes = []
        for period in time_R:
            copulas = wave.contours.environmental_contours(
                Hm0, Te, dt_ss, period, "PCA"
            )

            Hm0s.append(copulas["PCA_x1"])
            Tes.append(copulas["PCA_x2"])

        contour_label = [f"{year}-year Contour" for year in time_R]
        plt.figure()
        (
            wave.graphics.plot_environmental_contour(
                Te,
                Hm0,
                Tes,
                Hm0s,
                data_label="NDBC 46022",
                contour_label=contour_label,
                x_label="Te [s]",
                y_label="Hm0 [m]",
            )
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_standard_copulas(self):
        copulas = wave.contours.environmental_contours(
            self.wdrt_Hm0,
            self.wdrt_Te,
            self.wdrt_dt,
            self.wdrt_period,
            method=["gaussian", "gumbel", "clayton"],
        )

        # WDRT slightly vaires Rosenblatt copula parameters from
        #    the other copula default  parameters
        rosen = wave.contours.environmental_contours(
            self.wdrt_Hm0,
            self.wdrt_Te,
            self.wdrt_dt,
            self.wdrt_period,
            method=["rosenblatt"],
            min_bin_count=50,
            initial_bin_max_val=0.5,
            bin_val_size=0.25,
        )
        copulas["rosenblatt_x1"] = rosen["rosenblatt_x1"]
        copulas["rosenblatt_x2"] = rosen["rosenblatt_x2"]

        methods = ["gaussian", "gumbel", "clayton", "rosenblatt"]
        close = []
        for method in methods:
            close.append(
                np.allclose(copulas[f"{method}_x1"], self.wdrt_copulas[f"{method}_x1"])
            )
            close.append(
                np.allclose(copulas[f"{method}_x2"], self.wdrt_copulas[f"{method}_x2"])
            )
        self.assertTrue(all(close))

    def test_nonparametric_copulas(self):
        methods = [
            "nonparametric_gaussian",
            "nonparametric_clayton",
            "nonparametric_gumbel",
        ]

        np_copulas = wave.contours.environmental_contours(
            self.wdrt_Hm0, self.wdrt_Te, self.wdrt_dt, self.wdrt_period, method=methods
        )

        close = []
        for method in methods:
            close.append(
                np.allclose(
                    np_copulas[f"{method}_x1"],
                    self.wdrt_copulas[f"{method}_x1"],
                    atol=0.13,
                )
            )
            close.append(
                np.allclose(
                    np_copulas[f"{method}_x2"],
                    self.wdrt_copulas[f"{method}_x2"],
                    atol=0.13,
                )
            )
        self.assertTrue(all(close))

    def test_kde_copulas(self):
        kde_copula = wave.contours.environmental_contours(
            self.wdrt_Hm0,
            self.wdrt_Te,
            self.wdrt_dt,
            self.wdrt_period,
            method=["bivariate_KDE"],
            bandwidth=[0.23, 0.23],
        )
        log_kde_copula = wave.contours.environmental_contours(
            self.wdrt_Hm0,
            self.wdrt_Te,
            self.wdrt_dt,
            self.wdrt_period,
            method=["bivariate_KDE_log"],
            bandwidth=[0.02, 0.11],
        )

        close = [
            np.allclose(
                kde_copula["bivariate_KDE_x1"], self.wdrt_copulas["bivariate_KDE_x1"]
            ),
            np.allclose(
                kde_copula["bivariate_KDE_x2"], self.wdrt_copulas["bivariate_KDE_x2"]
            ),
            np.allclose(
                log_kde_copula["bivariate_KDE_log_x1"],
                self.wdrt_copulas["bivariate_KDE_log_x1"],
            ),
            np.allclose(
                log_kde_copula["bivariate_KDE_log_x2"],
                self.wdrt_copulas["bivariate_KDE_log_x2"],
            ),
        ]
        self.assertTrue(all(close))

    def test_samples_contours(self):
        te_samples = np.array([10, 15, 20])
        hs_samples_0 = np.array([8.56637939, 9.27612515, 8.70427774])
        hs_contour = np.array(self.wdrt_copulas["gaussian_x1"])
        te_contour = np.array(self.wdrt_copulas["gaussian_x2"])
        hs_samples = wave.contours.samples_contour(te_samples, te_contour, hs_contour)
        assert_allclose(hs_samples, hs_samples_0)

    def test_samples_seastate(self):
        hs_0 = np.array(
            [5.91760129, 4.55185088, 1.41144991, 12.64443154, 7.89753791, 0.93890797]
        )
        te_0 = np.array(
            [14.24199604, 8.25383556, 6.03901866, 16.9836369, 9.51967777, 3.46969355]
        )
        w_0 = np.array(
            [
                2.18127398e-01,
                2.18127398e-01,
                2.18127398e-01,
                2.45437862e-07,
                2.45437862e-07,
                2.45437862e-07,
            ]
        )

        df = self.Hm0Te[self.Hm0Te["Hm0"] < 20]
        dt_ss = (self.Hm0Te.index[2] - self.Hm0Te.index[1]).seconds
        points_per_interval = 3
        return_periods = np.array([50, 100])
        np.random.seed(0)
        hs, te, w = wave.contours.samples_full_seastate(
            df.Hm0.values, df.Te.values, points_per_interval, return_periods, dt_ss
        )
        assert_allclose(hs, hs_0)
        assert_allclose(te, te_0)
        assert_allclose(w, w_0)


if __name__ == "__main__":
    unittest.main()
