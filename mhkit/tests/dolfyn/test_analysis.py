from . import test_read_adp as tr, test_read_adv as tv
from .base import load_netcdf as load, save_netcdf as save, assert_allclose
from mhkit.dolfyn import VelBinner, read_example
import mhkit.dolfyn.adv.api as avm
from xarray.testing import assert_identical
import unittest
import pytest
import numpy as np

make_data = False


class analysis_testcase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.adv1 = tv.dat.copy(deep=True)
        self.adv2 = read_example('burst_mode01.VEC', nens=90)
        self.adv_tool = VelBinner(n_bin=self.adv1.fs, fs=self.adv1.fs)

        self.adp = tr.dat_sig.copy(deep=True)
        with pytest.warns(UserWarning):
            self.adp_tool = VelBinner(n_bin=self.adp.fs*20,
                                      fs=self.adp.fs,
                                      n_fft=self.adp.fs*40)

    @classmethod
    def tearDownClass(self):
        pass

    def test_do_func(self):
        adat_vec = self.adv_tool.bin_average(self.adv1)
        adat_vec = self.adv_tool.bin_variance(self.adv1, out_ds=adat_vec)
        adat_vec['tke_vec'] = self.adv_tool.turbulent_kinetic_energy(
            self.adv1.vel)
        adat_vec['stress'] = self.adv_tool.stresses(self.adv1.vel)

        adat_sig = self.adp_tool.bin_average(self.adp)
        adat_sig = self.adp_tool.bin_variance(self.adp, out_ds=adat_sig)

        if make_data:
            save(adat_vec, 'vector_data01_avg.nc')
            save(adat_sig, 'BenchFile01_avg.nc')
            return

        assert_allclose(adat_vec, load('vector_data01_avg.nc'), atol=1e-6)
        assert_allclose(adat_sig, load('BenchFile01_avg.nc'), atol=1e-6)

    def test_calc_func(self):
        test_ds = type(self.adv1)()
        test_ds_demean = type(self.adv1)()
        test_ds_dif = type(self.adv1)()
        c = self.adv_tool

        c2 = self.adp_tool
        test_ds_adp = type(self.adp)()

        test_ds['coh'] = c.coherence(
            self.adv1.vel[0], self.adv1.vel[1], n_fft_coh=self.adv1.fs)
        test_ds['pang'] = c.phase_angle(
            self.adv1.vel[0], self.adv1.vel[1], n_fft_coh=self.adv1.fs)
        test_ds['xcov'] = c.cross_covariance(
            self.adv1.vel[0], self.adv1.vel[1])
        test_ds['acov'] = c.autocovariance(self.adv1.vel)
        test_ds['tke_vec'] = c.turbulent_kinetic_energy(self.adv1.vel)
        test_ds['stress'] = c.stresses(self.adv1.vel)
        test_ds['psd'] = c.power_spectral_density(self.adv1.vel)
        test_ds['csd'] = c.cross_spectral_density(self.adv1.vel)

        test_ds_demean['tke_vec'] = c.turbulent_kinetic_energy(
            self.adv1.vel, detrend=False)
        test_ds_demean['stress'] = c.stresses(
            self.adv1.vel, detrend=False)

        # Different lengths
        test_ds_dif['coh_dif'] = c.coherence(self.adv1.vel, self.adv2.vel)
        test_ds_dif['pang_dif'] = c.phase_angle(
            self.adv1.vel, self.adv2.vel)

        # Test ADCP single vector spectra, cross-spectra to test radians code
        test_ds_adp['psd_b5'] = c2.power_spectral_density(
            self.adp.vel_b5.isel(range_b5=5), window='hamm')
        test_ds_adp['tke_b5'] = c2.turbulent_kinetic_energy(self.adp.vel_b5)
        test_ds_adp['csd'] = c2.cross_spectral_density(self.adp.vel.isel(dir=slice(0, 3), range=0),
                                                       freq_units='rad', window='hamm')

        if make_data:
            save(test_ds, 'vector_data01_func.nc')
            save(test_ds_dif, 'vector_data01_funcdif.nc')
            save(test_ds_demean, 'vector_data01_func_demean.nc')
            save(test_ds_adp, 'BenchFile01_func.nc')
            return

        assert_allclose(test_ds, load('vector_data01_func.nc'), atol=1e-6)
        assert_allclose(test_ds_dif, load(
            'vector_data01_funcdif.nc'), atol=1e-6)
        assert_allclose(test_ds_demean, load(
            'vector_data01_func_demean.nc'), atol=1e-6)
        assert_allclose(test_ds_adp, load('BenchFile01_func.nc'), atol=1e-6)

    def test__fft_freq(self):
        f = self.adv_tool._fft_freq(units='Hz')
        omega = self.adv_tool._fft_freq(units='rad/s')

        np.testing.assert_equal(f, np.arange(1, 17, 1, dtype='float'))
        np.testing.assert_equal(omega, np.arange(
            1, 17, 1, dtype='float')*(2*np.pi))

    def test_adv_turbulence(self):
        dat = tv.dat.copy(deep=True)
        bnr = avm.ADVBinner(n_bin=20.0, fs=dat.fs)
        tdat = bnr(dat)
        acov = bnr.autocovariance(dat.vel)

        assert_identical(tdat, avm.turbulence_statistics(
            dat, n_bin=20.0, fs=dat.fs))

        tdat['LT83'] = bnr.dissipation_rate_LT83(tdat.psd, tdat.velds.U_mag)
        tdat['SF'] = bnr.dissipation_rate_SF(dat.vel[0], tdat.velds.U_mag)
        tdat['TE01'] = bnr.dissipation_rate_TE01(dat, tdat)
        tdat['L'] = bnr.integral_length_scales(acov, tdat.velds.U_mag)

        if make_data:
            save(tdat, 'vector_data01_bin.nc')
            return

        assert_allclose(tdat, load('vector_data01_bin.nc'), atol=1e-6)
