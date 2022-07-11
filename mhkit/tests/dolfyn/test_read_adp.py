import mhkit.dolfyn.io.nortek2 as sig
from mhkit.dolfyn.io.nortek2_lib import crop_ensembles
from mhkit.dolfyn.io.api import read_example as read
from .base import assert_allclose
from . import base as tb
import warnings
import unittest
import pytest
import os

make_data = False
load = tb.load_netcdf
save = tb.save_netcdf

dat_rdi = load('RDI_test01.nc')
dat_rdi_7f79 = load('RDI_7f79.nc')
dat_rdi_bt = load('RDI_withBT.nc')
dat_rdi_vm = load('vmdas01.nc')
dat_wr1 = load('winriver01.nc')
dat_wr2 = load('winriver02.nc')

dat_awac = load('AWAC_test01.nc')
dat_awac_ud = load('AWAC_test01_ud.nc')
dat_hwac = load('H-AWAC_test01.nc')
dat_sig = load('BenchFile01.nc')
dat_sig_i = load('Sig1000_IMU.nc')
dat_sig_i_ud = load('Sig1000_IMU_ud.nc')
dat_sig_ieb = load('VelEchoBT01.nc')
dat_sig_ie = load('Sig500_Echo.nc')
dat_sig_tide = load('Sig1000_tidal.nc')
dat_sig_skip = load('Sig_SkippedPings01.nc')
dat_sig_badt = load('Sig1000_BadTime01.nc')
dat_sig5_leiw = load('Sig500_last_ensemble_is_whole.nc')


class io_adp_testcase(unittest.TestCase):
    def test_io_rdi(self):
        warnings.simplefilter('ignore', UserWarning)
        nens = 100
        td_rdi = tb.drop_config(read('RDI_test01.000'))
        td_7f79 = tb.drop_config(read('RDI_7f79.000'))
        td_rdi_bt = tb.drop_config(read('RDI_withBT.000', nens=nens))
        td_vm = tb.drop_config(read('vmdas01.ENX', nens=nens))
        td_wr1 = tb.drop_config(read('winriver01.PD0'))
        td_wr2 = tb.drop_config(read('winriver02.PD0'))

        if make_data:
            save(td_rdi, 'RDI_test01.nc')
            save(td_7f79, 'RDI_7f79.nc')
            save(td_rdi_bt, 'RDI_withBT.nc')
            save(td_vm, 'vmdas01.nc')
            save(td_wr1, 'winriver01.nc')
            save(td_wr2, 'winriver02.nc')
            return

        assert_allclose(td_rdi, dat_rdi, atol=1e-6)
        assert_allclose(td_7f79, dat_rdi_7f79, atol=1e-6)
        assert_allclose(td_rdi_bt, dat_rdi_bt, atol=1e-6)
        assert_allclose(td_vm, dat_rdi_vm, atol=1e-6)
        assert_allclose(td_wr1, dat_wr1, atol=1e-6)
        assert_allclose(td_wr2, dat_wr2, atol=1e-6)

    def test_io_nortek(self):
        nens = 100
        with pytest.warns(UserWarning):
            td_awac = tb.drop_config(
                read('AWAC_test01.wpr', userdata=False, nens=[0, nens]))
        td_awac_ud = tb.drop_config(read('AWAC_test01.wpr', nens=nens))
        td_hwac = tb.drop_config(read('H-AWAC_test01.wpr'))

        if make_data:
            save(td_awac, 'AWAC_test01.nc')
            save(td_awac_ud, 'AWAC_test01_ud.nc')
            save(td_hwac, 'H-AWAC_test01.nc')
            return

        assert_allclose(td_awac, dat_awac, atol=1e-6)
        assert_allclose(td_awac_ud, dat_awac_ud, atol=1e-6)
        assert_allclose(td_hwac, dat_hwac, atol=1e-6)

    def test_io_nortek2(self):
        nens = 100
        td_sig = tb.drop_config(read('BenchFile01.ad2cp', nens=nens))
        td_sig_i = tb.drop_config(read('Sig1000_IMU.ad2cp', userdata=False,
                                       nens=nens))
        td_sig_i_ud = tb.drop_config(read('Sig1000_IMU.ad2cp', nens=nens))
        td_sig_ieb = tb.drop_config(read('VelEchoBT01.ad2cp', nens=nens))
        td_sig_ie = tb.drop_config(read('Sig500_Echo.ad2cp', nens=nens))
        td_sig_tide = tb.drop_config(read('Sig1000_tidal.ad2cp', nens=nens))

        with pytest.warns(UserWarning):
            # This issues a warning...
            td_sig_skip = tb.drop_config(read('Sig_SkippedPings01.ad2cp'))

        with pytest.warns(UserWarning):
            td_sig_badt = tb.drop_config(sig.read_signature(
                tb.rfnm('Sig1000_BadTime01.ad2cp')))

        # Make sure we read all the way to the end of the file.
        # This file ends exactly at the end of an ensemble.
        td_sig5_leiw = tb.drop_config(
            read('Sig500_last_ensemble_is_whole.ad2cp'))

        os.remove(tb.exdt('BenchFile01.ad2cp.index'))
        os.remove(tb.exdt('Sig1000_IMU.ad2cp.index'))
        os.remove(tb.exdt('VelEchoBT01.ad2cp.index'))
        os.remove(tb.exdt('Sig500_Echo.ad2cp.index'))
        os.remove(tb.exdt('Sig1000_tidal.ad2cp.index'))
        os.remove(tb.exdt('Sig_SkippedPings01.ad2cp.index'))
        os.remove(tb.exdt('Sig500_last_ensemble_is_whole.ad2cp.index'))
        os.remove(tb.rfnm('Sig1000_BadTime01.ad2cp.index'))

        if make_data:
            save(td_sig, 'BenchFile01.nc')
            save(td_sig_i, 'Sig1000_IMU.nc')
            save(td_sig_i_ud, 'Sig1000_IMU_ud.nc')
            save(td_sig_ieb, 'VelEchoBT01.nc')
            save(td_sig_ie, 'Sig500_Echo.nc')
            save(td_sig_tide, 'Sig1000_tidal.nc')
            save(td_sig_skip, 'Sig_SkippedPings01.nc')
            save(td_sig_badt, 'Sig1000_BadTime01.nc')
            save(td_sig5_leiw, 'Sig500_last_ensemble_is_whole.nc')
            return

        assert_allclose(td_sig, dat_sig, atol=1e-6)
        assert_allclose(td_sig_i, dat_sig_i, atol=1e-6)
        assert_allclose(td_sig_i_ud, dat_sig_i_ud, atol=1e-6)
        assert_allclose(td_sig_ieb, dat_sig_ieb, atol=1e-6)
        assert_allclose(td_sig_ie, dat_sig_ie, atol=1e-6)
        assert_allclose(td_sig_tide, dat_sig_tide, atol=1e-6)
        assert_allclose(td_sig5_leiw, dat_sig5_leiw, atol=1e-6)
        assert_allclose(td_sig_skip, dat_sig_skip, atol=1e-6)
        assert_allclose(td_sig_badt, dat_sig_badt, atol=1e-6)

    def test_nortek2_crop(self):
        # Test file cropping function
        crop_ensembles(infile=tb.exdt('Sig500_Echo.ad2cp'),
                       outfile=tb.exdt('Sig500_Echo_crop.ad2cp'),
                       range=[50, 100])
        td_sig_ie_crop = tb.drop_config(read('Sig500_Echo_crop.ad2cp'))

        if make_data:
            save(td_sig_ie_crop, 'Sig500_Echo_crop.nc')
            return

        os.remove(tb.exdt('Sig500_Echo.ad2cp.index'))
        os.remove(tb.exdt('Sig500_Echo_crop.ad2cp'))
        os.remove(tb.exdt('Sig500_Echo_crop.ad2cp.index'))

        cd_sig_ie_crop = load('Sig500_Echo_crop.nc')
        assert_allclose(td_sig_ie_crop, cd_sig_ie_crop, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
