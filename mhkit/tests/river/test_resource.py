from os.path import abspath, dirname, join, isfile, normpath
import matplotlib.pylab as plt
import mhkit.river as river
import pandas as pd
import xarray as xr
import numpy as np
import unittest
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, "plots")
isdir = os.path.isdir(plotdir)
if not isdir:
    os.mkdir(plotdir)
datadir = normpath(join(testdir, "..", "..", "..", "examples", "data", "river"))


class TestResource(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data = pd.read_csv(
            join(datadir, "tanana_discharge_data.csv"), index_col=0, parse_dates=True
        )
        self.data.columns = ["Q"]

        self.results = pd.read_csv(
            join(datadir, "tanana_test_results.csv"), index_col=0, parse_dates=True
        )

    @classmethod
    def tearDownClass(self):
        pass

    def test_Froude_number(self):
        v = 2
        h = 5
        Fr = river.resource.Froude_number(v, h)
        self.assertAlmostEqual(Fr, 0.286, places=3)

    def test_froude_number_v_type_error(self):
        v = "invalid_type"  # String instead of int/float
        h = 5
        with self.assertRaises(TypeError):
            river.resource.Froude_number(v, h)

    def test_froude_number_h_type_error(self):
        v = 2
        h = "invalid_type"  # String instead of int/float
        with self.assertRaises(TypeError):
            river.resource.Froude_number(v, h)

    def test_froude_number_g_type_error(self):
        v = 2
        h = 5
        g = "invalid_type"  # String instead of int/float
        with self.assertRaises(TypeError):
            river.resource.Froude_number(v, h, g)

    def test_exceedance_probability(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = pd.Series(np.arange(9))
        # Rank order for non-repeating elements simply adds 1 to each element
        # if N=9, max F = 100((max(Q)+1)/10) =  90%
        # if N=9, min F = 100((min(Q)+1)/10) =  10%
        f = river.resource.exceedance_probability(Q)
        self.assertEqual(f.min().values, 10.0)
        self.assertEqual(f.max().values, 90.0)

    def test_exceedance_probability_xarray(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = xr.DataArray(
            data=np.arange(9), dims="index", coords={"index": np.arange(9)}
        )
        # if N=9, max F = 100((max(Q)+1)/10) =  90%
        # if N=9, min F = 100((min(Q)+1)/10) =  10%
        f = river.resource.exceedance_probability(Q)
        self.assertEqual(f.min().values, 10.0)
        self.assertEqual(f.max().values, 90.0)

    def test_exceedance_probability_type_error(self):
        D = "invalid_type"  # String instead of pd.Series or pd.DataFrame
        with self.assertRaises(TypeError):
            river.resource.exceedance_probability(D)

    def test_polynomial_fit(self):
        # Calculate a first order polynomial on an x=y line
        p, r2 = river.resource.polynomial_fit(np.arange(8), np.arange(8), 1)
        # intercept should be 0
        self.assertAlmostEqual(p[0], 0.0, places=2)
        # slope should be 1
        self.assertAlmostEqual(p[1], 1.0, places=2)
        # r-squared should be perfect
        self.assertAlmostEqual(r2, 1.0, places=2)

    def test_polynomial_fit_x_type_error(self):
        x = "invalid_type"  # String instead of numpy array
        y = np.array([1, 2, 3])
        n = 1
        with self.assertRaises(TypeError):
            river.resource.polynomial_fit(x, y, n)

    def test_polynomial_fit_y_type_error(self):
        x = np.array([1, 2, 3])
        y = "invalid_type"  # String instead of numpy array
        n = 1
        with self.assertRaises(TypeError):
            river.resource.polynomial_fit(x, y, n)

    def test_polynomial_fit_n_type_error(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        n = "invalid_type"  # String instead of int
        with self.assertRaises(TypeError):
            river.resource.polynomial_fit(x, y, n)

    def test_discharge_to_velocity(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = pd.Series(np.arange(9))
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Because the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(Q, p)
        self.assertAlmostEqual(np.sum(10 * Q - V["V"]), 0.00, places=2)

    def test_discharge_to_velocity_xarray(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = xr.DataArray(
            data=np.arange(9), dims="index", coords={"index": np.arange(9)}
        )
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Because the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(Q, p, to_pandas=False)
        self.assertAlmostEqual(np.sum(10 * Q - V["V"]).values, 0.00, places=2)

    def test_discharge_to_velocity_D_type_error(self):
        D = "invalid_type"  # String instead of pd.Series or pd.DataFrame
        polynomial_coefficients = np.poly1d([1, 2])
        with self.assertRaises(TypeError):
            river.resource.discharge_to_velocity(D, polynomial_coefficients)

    def test_discharge_to_velocity_polynomial_coefficients_type_error(self):
        D = pd.Series([1, 2, 3])
        polynomial_coefficients = "invalid_type"  # String instead of np.poly1d
        with self.assertRaises(TypeError):
            river.resource.discharge_to_velocity(D, polynomial_coefficients)

    def test_velocity_to_power(self):
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Because the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(pd.Series(np.arange(9)), p)
        # Calculate a first order polynomial on an VP_Curve x=y line 10 times greater than the V values
        p2, r22 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Set cut in/out to exclude 1 bin on either end of V range
        cut_in = V["V"][1]
        cut_out = V["V"].iloc[-2]
        # Power should be 10x greater and exclude the ends of V
        P = river.resource.velocity_to_power(V["V"], p2, cut_in, cut_out)
        # Cut in power zero
        self.assertAlmostEqual(P["P"][0], 0.00, places=2)
        # Cut out power zero
        self.assertAlmostEqual(P["P"].iloc[-1], 0.00, places=2)
        # Middle 10x greater than velocity
        self.assertAlmostEqual((P["P"][1:-1] - 10 * V["V"][1:-1]).sum(), 0.00, places=2)

    def test_velocity_to_power_xarray(self):
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Because the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(
            pd.Series(np.arange(9)), p, dimension="", to_pandas=False
        )
        # Calculate a first order polynomial on an VP_Curve x=y line 10 times greater than the V values
        p2, r22 = river.resource.polynomial_fit(np.arange(9), 10 * np.arange(9), 1)
        # Set cut in/out to exclude 1 bin on either end of V range
        cut_in = V["V"].values[1]
        cut_out = V["V"].values[-2]
        # Power should be 10x greater and exclude the ends of V
        P = river.resource.velocity_to_power(
            V["V"], p2, cut_in, cut_out, to_pandas=False
        )
        # Cut in power zero
        self.assertAlmostEqual(P["P"][0], 0.00, places=2)
        # Cut out power zero
        self.assertAlmostEqual(P["P"][-1], 0.00, places=2)
        # Middle 10x greater than velocity
        self.assertAlmostEqual(
            (P["P"][1:-1] - 10 * V["V"][1:-1]).sum().values, 0.00, places=2
        )

    def test_velocity_to_power_V_type_error(self):
        V = "invalid_type"  # String instead of pd.Series or pd.DataFrame
        polynomial_coefficients = np.poly1d([1, 2])
        cut_in = 1
        cut_out = 5
        with self.assertRaises(TypeError):
            river.resource.velocity_to_power(
                V, polynomial_coefficients, cut_in, cut_out
            )

    def test_velocity_to_power_polynomial_coefficients_type_error(self):
        V = pd.Series([1, 2, 3])
        polynomial_coefficients = "invalid_type"  # String instead of np.poly1d
        cut_in = 1
        cut_out = 5
        with self.assertRaises(TypeError):
            river.resource.velocity_to_power(
                V, polynomial_coefficients, cut_in, cut_out
            )

    def test_velocity_to_power_cut_in_type_error(self):
        V = pd.Series([1, 2, 3])
        polynomial_coefficients = np.poly1d([1, 2])
        cut_in = "invalid_type"  # String instead of int/float
        cut_out = 5
        with self.assertRaises(TypeError):
            river.resource.velocity_to_power(
                V, polynomial_coefficients, cut_in, cut_out
            )

    def test_velocity_to_power_cut_out_type_error(self):
        V = pd.Series([1, 2, 3])
        polynomial_coefficients = np.poly1d([1, 2])
        cut_in = 1
        cut_out = "invalid_type"  # String instead of int/float
        with self.assertRaises(TypeError):
            river.resource.velocity_to_power(
                V, polynomial_coefficients, cut_in, cut_out
            )

    def test_energy_produced(self):
        # If power is always X then energy produced with be x*seconds
        X = 1
        seconds = 1
        P = pd.Series(X * np.ones(10))
        EP = river.resource.energy_produced(P, seconds)
        self.assertAlmostEqual(EP, X * seconds, places=1)
        # for a normal distribution of Power EP = mean *seconds
        mu = 5
        sigma = 1
        power_dist = pd.Series(np.random.normal(mu, sigma, 10000))
        EP2 = river.resource.energy_produced(power_dist, seconds)
        self.assertAlmostEqual(EP2, mu * seconds, places=1)

    def test_energy_produced_xarray(self):
        # If power is always X then energy produced with be x*seconds
        X = 1
        seconds = 1
        P = xr.DataArray(data=X * np.ones(10))
        EP = river.resource.energy_produced(P, seconds)
        self.assertAlmostEqual(EP, X * seconds, places=1)

        # for a normal distribution of Power EP = mean *seconds
        mu = 5
        sigma = 1
        power_dist = xr.DataArray(data=np.random.normal(mu, sigma, 10000))
        EP2 = river.resource.energy_produced(power_dist, seconds)
        self.assertAlmostEqual(EP2, mu * seconds, places=1)

    def test_energy_produced_P_type_error(self):
        P = "invalid_type"  # String instead of pd.Series or pd.DataFrame
        seconds = 3600
        with self.assertRaises(TypeError):
            river.resource.energy_produced(P, seconds)

    def test_energy_produced_seconds_type_error(self):
        P = pd.Series([100, 200, 300])
        seconds = "invalid_type"  # String instead of int/float
        with self.assertRaises(TypeError):
            river.resource.energy_produced(P, seconds)

    def test_plot_flow_duration_curve(self):
        filename = abspath(join(plotdir, "river_plot_flow_duration_curve.png"))
        if isfile(filename):
            os.remove(filename)

        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.data["Q"], f["F"])
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_power_duration_curve(self):
        filename = abspath(join(plotdir, "river_plot_power_duration_curve.png"))
        if isfile(filename):
            os.remove(filename)

        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.results["P_control"], f["F"])
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_velocity_duration_curve(self):
        filename = abspath(join(plotdir, "river_plot_velocity_duration_curve.png"))
        if isfile(filename):
            os.remove(filename)

        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_velocity_duration_curve(self.results["V_control"], f["F"])
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_discharge_timeseries(self):
        filename = abspath(join(plotdir, "river_plot_discharge_timeseries.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        river.graphics.plot_discharge_timeseries(self.data["Q"])
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_discharge_vs_velocity(self):
        filename = abspath(join(plotdir, "river_plot_discharge_vs_velocity.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        river.graphics.plot_discharge_vs_velocity(
            self.data["Q"], self.results["V_control"]
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_velocity_vs_power(self):
        filename = abspath(join(plotdir, "river_plot_velocity_vs_power.png"))
        if isfile(filename):
            os.remove(filename)

        plt.figure()
        river.graphics.plot_velocity_vs_power(
            self.results["V_control"], self.results["P_control"]
        )
        plt.savefig(filename, format="png")
        plt.close()

        self.assertTrue(isfile(filename))


if __name__ == "__main__":
    unittest.main()
