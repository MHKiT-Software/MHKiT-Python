import mhkit.dolfyn.io.rdi as wh
import mhkit.dolfyn.io.nortek as awac
import mhkit.dolfyn.io.nortek2 as sig
from mhkit.dolfyn.io.api import read_example as read
from .base import assert_allclose, save_netcdf, save_matlab, load_matlab, exdt, rfnm, drop_config
from . import test_read_adp as tp
from . import test_read_adv as tv
import contextlib
import unittest
import pytest
import os
import io
make_data = False


class io_testcase(unittest.TestCase):
    def test_save(self):
        ds = tv.dat.copy(deep=True)
        save_netcdf(ds, 'test_save', compression=True)
        save_matlab(ds, 'test_save')

        assert os.path.exists(rfnm('test_save.nc'))
        assert os.path.exists(rfnm('test_save.mat'))

    def test_matlab_io(self):
        nens = 100
        td_vec = drop_config(read('vector_data_imu01.VEC', nens=nens))
        td_rdi_bt = drop_config(read('RDI_withBT.000', nens=nens))

        # This read should trigger a warning about the declination being
        # defined in two places (in the binary .ENX files), and in the
        # .userdata.json file. NOTE: DOLfYN defaults to using what is in
        # the .userdata.json file.
        with pytest.warns(UserWarning, match='magnetic_var_deg'):
            td_vm = drop_config(read('vmdas01.ENX', nens=nens))

        if make_data:
            save_matlab(td_vec, 'dat_vec')
            save_matlab(td_rdi_bt, 'dat_rdi_bt')
            save_matlab(td_vm, 'dat_vm')
            return

        mat_vec = load_matlab('dat_vec.mat')
        mat_rdi_bt = load_matlab('dat_rdi_bt.mat')
        mat_vm = load_matlab('dat_vm.mat')

        assert_allclose(td_vec, mat_vec, atol=1e-6)
        assert_allclose(td_rdi_bt, mat_rdi_bt, atol=1e-6)
        assert_allclose(td_vm, mat_vm, atol=1e-6)

    def test_debugging(self):
        def debug_output(f, func, datafile, nens, *args, **kwargs):
            with contextlib.redirect_stdout(f):
                drop_config(func(exdt(datafile), nens=nens, *args, **kwargs))

        def remove_local_path(stringIO):
            string = stringIO.getvalue()
            start = string.find("Indexing")
            if start != -1:
                start += 8
                end = stringIO.getvalue().find("...")
                string = string[0:start] + string[end+3:]

            start = string.find("Reading file") + 12
            end = string.find(" ...")
            return string[0:start] + string[end:]

        def save_txt(fname, string):
            with open(rfnm(fname), 'w') as f:
                f.write(string)

        def read_txt(fname):
            with open(rfnm(fname), 'r') as f:
                string = f.read()
            return string

        nens = 100
        db_rdi = io.StringIO()
        db_awac = io.StringIO()
        db_vec = io.StringIO()
        db_sig = io.StringIO()

        debug_output(db_rdi, wh.read_rdi, 'RDI_withBT.000', nens, debug=11)
        debug_output(db_awac, awac.read_nortek, 'AWAC_test01.wpr',
                     nens, debug=True, do_checksum=True)
        debug_output(db_vec, awac.read_nortek, 'vector_data_imu01.VEC',
                     nens, debug=True, do_checksum=True)
        debug_output(db_sig, sig.read_signature, 'Sig500_Echo.ad2cp',
                     nens, rebuild_index=True, debug=True)
        os.remove(exdt('Sig500_Echo.ad2cp.index'))

        str_rdi = remove_local_path(db_rdi)
        str_awac = remove_local_path(db_awac)
        str_vec = remove_local_path(db_vec)
        str_sig = remove_local_path(db_sig)

        if make_data:
            save_txt('rdi_debug_out.txt', str_rdi)
            save_txt('awac_debug_out.txt', str_awac)
            save_txt('vec_debug_out.txt', str_vec)
            save_txt('sig_debug_out.txt', str_sig)
            return

        test_rdi = read_txt('rdi_debug_out.txt')
        test_awac = read_txt('awac_debug_out.txt')
        test_vec = read_txt('vec_debug_out.txt')
        test_sig = read_txt('sig_debug_out.txt')

        assert test_rdi == str_rdi
        assert test_awac == str_awac
        assert test_vec == str_vec
        assert test_sig == str_sig

    def test_read_warnings(self):
        with self.assertRaises(Exception):
            wh.read_rdi(exdt('H-AWAC_test01.wpr'))
        with self.assertRaises(Exception):
            awac.read_nortek(exdt('BenchFile01.ad2cp'))
        with self.assertRaises(Exception):
            sig.read_signature(exdt('AWAC_test01.wpr'))
        with self.assertRaises(IOError):
            read(rfnm('AWAC_test01.nc'))
        with self.assertRaises(Exception):
            save_netcdf(tp.dat_rdi, 'test_save.fail')
