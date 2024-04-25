from . import test_read_adp as tr, test_read_adv as tv
from mhkit.tests.dolfyn.base import (
    load_netcdf as load,
    save_netcdf as save,
    assert_allclose,
)
from mhkit.dolfyn import VelBinner, read_example
import mhkit.dolfyn.adv.api as avm
import mhkit.dolfyn.adp.api as apm
from xarray.testing import assert_identical
import unittest
import pytest
import numpy as np

make_data = False


class analysis_testcase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.adv1 = tv.dat.copy(deep=True)
        self.adv2 = read_example("vector_burst_mode01.VEC", nens=90)
        self.adv_tool = VelBinner(n_bin=self.adv1.fs, fs=self.adv1.fs)

        self.adp = tr.dat_sig.copy(deep=True)
        with pytest.warns(UserWarning):
            self.adp_tool = VelBinner(
                n_bin=self.adp.fs * 20, fs=self.adp.fs, n_fft=self.adp.fs * 40
            )

    @classmethod
    def tearDownClass(self):
        pass

    def test_do_func(self):
        ds_vec = self.adv_tool.bin_average(self.adv1)
        ds_vec = self.adv_tool.bin_variance(self.adv1, out_ds=ds_vec)

        # test non-integer bin sizes
        mean_test = self.adv_tool.mean(self.adv1["vel"].values, n_bin=ds_vec.fs * 1.01)

        ds_sig = self.adp_tool.bin_average(self.adp)
        ds_sig = self.adp_tool.bin_variance(self.adp, out_ds=ds_sig)

        if make_data:
            save(ds_vec, "vector_data01_avg.nc")
            save(ds_sig, "BenchFile01_avg.nc")
            return

        assert np.sum(mean_test - ds_vec.vel.values) == 0, "Mean test failed"
        assert_allclose(ds_vec, load("vector_data01_avg.nc"), atol=1e-6)
        assert_allclose(ds_sig, load("BenchFile01_avg.nc"), atol=1e-6)

    def test_calc_func(self):
        c = self.adv_tool
        c2 = self.adp_tool

        test_ds = type(self.adv1)()
        test_ds_adp = type(self.adp)()

        test_ds["acov"] = c.autocovariance(self.adv1.vel)
        test_ds["tke_vec_detrend"] = c.turbulent_kinetic_energy(
            self.adv1.vel, detrend=True
        )
        test_ds["tke_vec_demean"] = c.turbulent_kinetic_energy(
            self.adv1.vel, detrend=False
        )
        test_ds["psd"] = c.power_spectral_density(self.adv1.vel, freq_units="Hz")

        # Test ADCP single vector spectra, cross-spectra to test radians code
        test_ds_adp["psd_b5"] = c2.power_spectral_density(
            self.adp.vel_b5.isel(range_b5=5), freq_units="rad", window="hamm"
        )
        test_ds_adp["tke_b5"] = c2.turbulent_kinetic_energy(self.adp.vel_b5)

        if make_data:
            save(test_ds, "vector_data01_func.nc")
            save(test_ds_adp, "BenchFile01_func.nc")
            return

        assert_allclose(test_ds, load("vector_data01_func.nc"), atol=1e-6)
        assert_allclose(test_ds_adp, load("BenchFile01_func.nc"), atol=1e-6)

    def test_fft_freq(self):
        f = self.adv_tool._fft_freq(units="Hz")
        omega = self.adv_tool._fft_freq(units="rad/s")

        np.testing.assert_equal(f, np.arange(1, 17, 1, dtype="float"))
        np.testing.assert_equal(omega, np.arange(1, 17, 1, dtype="float") * (2 * np.pi))

    def test_adv_turbulence(self):
        dat = tv.dat.copy(deep=True)
        bnr = avm.ADVBinner(n_bin=20.0, fs=dat.fs)
        tdat = bnr(dat)
        acov = bnr.autocovariance(dat["vel"])

        assert_identical(tdat, avm.turbulence_statistics(dat, n_bin=20.0, fs=dat.fs))

        tdat["stress_detrend"] = bnr.reynolds_stress(dat["vel"])
        tdat["stress_demean"] = bnr.reynolds_stress(dat["vel"], detrend=False)
        tdat["csd"] = bnr.cross_spectral_density(
            dat["vel"], freq_units="rad", window="hamm", n_fft_coh=10
        )
        tdat["LT83"] = bnr.dissipation_rate_LT83(tdat["psd"], tdat.velds.U_mag)
        tdat["noise"] = bnr.doppler_noise_level(tdat["psd"], pct_fN=0.8)
        tdat["LT83_noise"] = bnr.dissipation_rate_LT83(
            tdat["psd"], tdat.velds.U_mag, noise=tdat["noise"]
        )
        tdat["SF"] = bnr.dissipation_rate_SF(dat["vel"][0], tdat.velds.U_mag)
        tdat["TE01"] = bnr.dissipation_rate_TE01(dat, tdat)
        tdat["L"] = bnr.integral_length_scales(acov, tdat.velds.U_mag)
        slope_check = bnr.check_turbulence_cascade_slope(
            tdat["psd"][-1].mean("time"), freq_range=[10, 100]
        )
        tdat["psd_noise"] = bnr.power_spectral_density(
            dat["vel"], freq_units="rad", noise=[0.06, 0.04, 0.01]
        )

        if make_data:
            save(tdat, "vector_data01_bin.nc")
            return

        assert np.round(slope_check[0].values, 4), 0.1713
        assert_allclose(tdat, load("vector_data01_bin.nc"), atol=1e-6)

    def test_adcp_turbulence(self):
        dat = tr.dat_sig_tide.copy(deep=True)
        dat.velds.rotate2("earth")
        dat.attrs["principal_heading"] = apm.calc_principal_heading(
            dat.vel.mean("range")
        )
        bnr = apm.ADPBinner(n_bin=20.0, fs=dat.fs, diff_style="centered")
        tdat = bnr.bin_average(dat)

        tdat["dudz"] = bnr.dudz(tdat["vel"])
        tdat["dvdz"] = bnr.dvdz(tdat["vel"])
        tdat["dwdz"] = bnr.dwdz(tdat["vel"])
        tdat["tau2"] = bnr.shear_squared(tdat["vel"])
        tdat["I"] = tdat.velds.I
        tdat["ti"] = bnr.turbulence_intensity(dat.velds.U_mag, detrend=False)
        dat.velds.rotate2("beam")

        tdat["psd"] = bnr.power_spectral_density(
            dat["vel"].isel(dir=2, range=len(dat.range) // 2), freq_units="Hz"
        )
        tdat["noise"] = bnr.doppler_noise_level(tdat["psd"], pct_fN=0.8)
        tdat["stress_vec4"] = bnr.reynolds_stress_4beam(
            dat, noise=tdat["noise"], orientation="up", beam_angle=25
        )
        tdat["tke_vec5"], tdat["stress_vec5"] = bnr.stress_tensor_5beam(
            dat, noise=tdat["noise"], orientation="up", beam_angle=25, tke_only=False
        )
        tdat["tke"] = bnr.total_turbulent_kinetic_energy(
            dat, noise=tdat["noise"], orientation="up", beam_angle=25
        )
        tdat["ti_noise"] = bnr.turbulence_intensity(
            dat.velds.U_mag, detrend=False, noise=tdat["noise"]
        )
        # This is "negative" for this code check
        tdat["wpwp"] = bnr.turbulent_kinetic_energy(dat["vel_b5"], noise=tdat["noise"])
        tdat["dissipation_rate_LT83"] = bnr.dissipation_rate_LT83(
            tdat["psd"],
            tdat.velds.U_mag.isel(range=len(dat.range) // 2),
            freq_range=[0.2, 0.4],
        )
        tdat["dissipation_rate_LT83_noise"] = bnr.dissipation_rate_LT83(
            tdat["psd"],
            tdat.velds.U_mag.isel(range=len(dat.range) // 2),
            freq_range=[0.2, 0.4],
            noise=tdat["noise"],
        )
        (
            tdat["dissipation_rate_SF"],
            tdat["noise_SF"],
            tdat["D_SF"],
        ) = bnr.dissipation_rate_SF(dat.vel.isel(dir=2), r_range=[1, 5])
        tdat["friction_vel"] = bnr.friction_velocity(
            tdat, upwp_=tdat["stress_vec5"].sel(tau="upwp_"), z_inds=slice(1, 5), H=50
        )
        slope_check = bnr.check_turbulence_cascade_slope(
            tdat["psd"].mean("time"), freq_range=[0.4, 4]
        )
        tdat["psd_noise"] = bnr.power_spectral_density(
            dat["vel"].isel(dir=2, range=len(dat.range) // 2),
            freq_units="Hz",
            noise=0.01,
        )

        if make_data:
            save(tdat, "Sig1000_tidal_bin.nc")
            return

        with pytest.raises(Exception):
            bnr.calc_psd(dat["vel"], freq_units="Hz", noise=0.01)

        with pytest.raises(Exception):
            bnr.calc_psd(dat["vel"][0], freq_units="Hz", noise=0.01)

        assert np.round(slope_check[0].values, 4), -1.0682

        assert_allclose(tdat, load("Sig1000_tidal_bin.nc"), atol=1e-6)
