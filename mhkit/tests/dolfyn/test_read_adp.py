from . import base as tb
from mhkit.dolfyn.io.api import read_example as read
import mhkit.dolfyn.io.rdi as wh
import mhkit.dolfyn.io.nortek as awac
import mhkit.dolfyn.io.nortek2 as sig
from .base import assert_allclose
import warnings
import os
import sys
import unittest
import pytest

load = tb.load_ncdata
save = tb.save_ncdata

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
dat_sig5_leiw = load('Sig500_last_ensemble_is_whole.nc')


def test_badtime():
    dat = sig.read_signature(tb.rfnm('Sig1000_BadTime01.ad2cp'))
    os.remove(tb.rfnm('Sig1000_BadTime01.ad2cp.index'))

    assert dat.time[199].isnull(), \
        "A good timestamp was found where a bad value is expected."


def test_io_rdi(make_data=False):
    warnings.simplefilter('ignore', UserWarning)
    nens = 500
    td_rdi = tb.drop_config(read('RDI_test01.000'))
    td_7f79 = tb.drop_config(read('RDI_7f79.000'))
    td_rdi_bt = tb.drop_config(read('RDI_withBT.000', nens=nens))
    td_vm = tb.drop_config(read('vmdas01.ENX', nens=nens))
    td_wr1 = tb.drop_config(read('winriver01.PD0'))
    td_wr2 = tb.drop_config(read('winriver02.PD0'))
    td_debug = tb.drop_config(wh.read_rdi(tb.exdt('RDI_withBT.000'), debug=11,
                                          nens=nens))

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
    assert_allclose(td_debug, td_rdi_bt, atol=1e-6)


def test_io_nortek(make_data=False):
    nens = 500
    td_awac = tb.drop_config(read('AWAC_test01.wpr', userdata=False,
                                  nens=nens))
    td_awac_ud = tb.drop_config(read('AWAC_test01.wpr', nens=nens))
    td_hwac = tb.drop_config(read('H-AWAC_test01.wpr'))
    td_debug = tb.drop_config(awac.read_nortek(tb.exdt('AWAC_test01.wpr'),
                              debug=True, do_checksum=True, nens=nens))

    if make_data:
        save(td_awac, 'AWAC_test01.nc')
        save(td_awac_ud, 'AWAC_test01_ud.nc')
        save(td_hwac, 'H-AWAC_test01.nc')
        return

    assert_allclose(td_awac, dat_awac, atol=1e-6)
    assert_allclose(td_awac_ud, dat_awac_ud, atol=1e-6)
    assert_allclose(td_hwac, dat_hwac, atol=1e-6)
    assert_allclose(td_awac_ud, td_debug, atol=1e-6)


def test_io_nortek2(make_data=False):
    nens = 500
    td_sig = tb.drop_config(read('BenchFile01.ad2cp', nens=nens))
    td_sig_i = tb.drop_config(read('Sig1000_IMU.ad2cp', userdata=False,
                                   nens=nens))
    td_sig_i_ud = tb.drop_config(read('Sig1000_IMU.ad2cp', nens=nens))
    td_sig_ieb = tb.drop_config(read('VelEchoBT01.ad2cp', nens=100))
    td_sig_ie = tb.drop_config(read('Sig500_Echo.ad2cp', nens=nens))
    td_sig_tide = tb.drop_config(read('Sig1000_tidal.ad2cp', nens=nens))

    # Make sure we read all the way to the end of the file.
    # This file ends exactly at the end of an ensemble.
    td_sig5_leiw = read('Sig500_last_ensemble_is_whole.ad2cp')

    os.remove(tb.exdt('BenchFile01.ad2cp.index'))
    os.remove(tb.exdt('Sig1000_IMU.ad2cp.index'))
    os.remove(tb.exdt('VelEchoBT01.ad2cp.index'))
    os.remove(tb.exdt('Sig500_Echo.ad2cp.index'))
    os.remove(tb.exdt('Sig1000_tidal.ad2cp.index'))

    if make_data:
        save(td_sig, 'BenchFile01.nc')
        save(td_sig_i, 'Sig1000_IMU.nc')
        save(td_sig_i_ud, 'Sig1000_IMU_ud.nc')
        save(td_sig_ieb, 'VelEchoBT01.nc')
        save(td_sig5_leiw, 'Sig500_last_ensemble_is_whole.nc')
        save(td_sig_ie, 'Sig500_Echo.nc')
        save(td_sig_tide, 'Sig1000_tidal.nc')
        return

    assert_allclose(td_sig, dat_sig, atol=1e-6)
    assert_allclose(td_sig_i, dat_sig_i, atol=1e-6)
    assert_allclose(td_sig_i_ud, dat_sig_i_ud, atol=1e-6)
    assert_allclose(td_sig_ieb, dat_sig_ieb, atol=1e-6)
    assert_allclose(td_sig_ie, dat_sig_ie, atol=1e-6)
    assert_allclose(td_sig_tide, dat_sig_tide, atol=1e-6)
    assert_allclose(td_sig5_leiw, dat_sig5_leiw, atol=1e-6)


def test_matlab_io(make_data=False):
    td_rdi_bt = tb.drop_config(read('RDI_withBT.000', nens=100))

    # This read should trigger a warning about the declination being
    # defined in two places (in the binary .ENX files), and in the
    # .userdata.json file. NOTE: DOLfYN defaults to using what is in
    # the .userdata.json file.
    with pytest.warns(UserWarning, match='magnetic_var_deg'):
        td_vm = tb.drop_config(read('vmdas01.ENX', nens=100))

    if make_data:
        tb.save_matlab(td_rdi_bt, 'dat_rdi_bt')
        tb.save_matlab(td_vm, 'dat_vm')
        return

    mat_rdi_bt = tb.load_matlab('dat_rdi_bt.mat')
    mat_vm = tb.load_matlab('dat_vm.mat')

    assert_allclose(td_rdi_bt, mat_rdi_bt,  atol=1e-6)
    assert_allclose(td_vm, mat_vm,  atol=1e-6)


class warnings_testcase(unittest.TestCase):
    def test_read_warnings(self):
        with self.assertRaises(Exception):
            wh.read_rdi(tb.exdt('H-AWAC_test01.wpr'))
        with self.assertRaises(Exception):
            awac.read_nortek(tb.exdt('BenchFile01.ad2cp'))
        with self.assertRaises(Exception):
            sig.read_signature(tb.exdt('AWAC_test01.wpr'))


if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)
    sys.stdout = open(os.devnull, 'w')  # block printing output
    test_io_rdi()
    test_io_nortek()
    test_io_nortek2()
    test_matlab_io()
    unittest.main()
    sys.stdout = sys.__stdout__  # restart printing output
