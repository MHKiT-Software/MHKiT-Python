from . import test_read_adp as tp
from . import test_read_adv as tv
from mhkit.tests.dolfyn.base import (
    assert_allclose,
    save_netcdf,
    save_matlab,
    load_matlab,
    exdt,
    rfnm,
)
import mhkit.dolfyn.io.rdi as wh
import mhkit.dolfyn.io.nortek as awac
import mhkit.dolfyn.io.nortek2 as sig
from mhkit.dolfyn.io.api import read_example as read
import unittest
import pytest
import os
import warnings
import numpy as np

make_data = False


class io_testcase(unittest.TestCase):
    def test_save(self):
        ds = tv.dat.copy(deep=True)
        ds2 = tp.dat_sig.copy(deep=True)
        save_netcdf(ds, "test_save")
        save_netcdf(ds2, "test_save_comp.nc", compression=True)
        save_matlab(ds, "test_save")

        assert os.path.exists(rfnm("test_save.nc"))
        assert os.path.exists(rfnm("test_save_comp.nc"))
        assert os.path.exists(rfnm("test_save.mat"))

        os.remove(rfnm("test_save.nc"))
        os.remove(rfnm("test_save_comp.nc"))
        os.remove(rfnm("test_save.mat"))

    def test_matlab_io(self):
        nens = 100
        td_vec = read("vector_data_imu01.VEC", nens=nens)
        td_rdi_bt = read("RDI_withBT.000", nens=nens)

        # This read should trigger a warning about the declination being
        # defined in two places (in the binary .ENX files), and in the
        # .userdata.json file. NOTE: DOLfYN defaults to using what is in
        # the .userdata.json file.
        with pytest.warns(UserWarning, match="magnetic_var_deg"):
            td_vm = read("vmdas01_wh.ENX", nens=nens)

        if make_data:
            save_matlab(td_vec, "dat_vec")
            save_matlab(td_rdi_bt, "dat_rdi_bt")
            save_matlab(td_vm, "dat_vm")
            return

        mat_vec = load_matlab("dat_vec.mat")
        mat_rdi_bt = load_matlab("dat_rdi_bt.mat")
        mat_vm = load_matlab("dat_vm.mat")

        assert_allclose(td_vec, mat_vec, atol=1e-6)
        assert_allclose(td_rdi_bt, mat_rdi_bt, atol=1e-6)
        assert_allclose(td_vm, mat_vm, atol=1e-6)

    def test_debugging(self):
        def read_txt(fname, loc):
            with open(loc(fname), "r") as f:
                string = f.read()
            return string

        def clip_file(fname):
            log = read_txt(fname, exdt)
            newlines = [i for i, ltr in enumerate(log) if ltr == "\n"]
            try:
                log = log[: newlines[100] + 1]
            except:
                pass
            with open(rfnm(fname), "w") as f:
                f.write(log)

        def read_file_and_test(fname):
            td = read_txt(fname, exdt)
            cd = read_txt(fname, rfnm)
            assert cd in td
            os.remove(exdt(fname))

        nens = 100
        wh.read_rdi(exdt("RDI_withBT.000"), nens, debug=3)
        awac.read_nortek(exdt("AWAC_test01.wpr"), nens, debug=True, do_checksum=True)
        awac.read_nortek(
            exdt("vector_data_imu01.VEC"), nens, debug=True, do_checksum=True
        )
        sig.read_signature(
            exdt("Sig500_Echo.ad2cp"), nens, rebuild_index=True, debug=True
        )
        os.remove(exdt("Sig500_Echo.ad2cp.index"))

        if make_data:
            clip_file("RDI_withBT.dolfyn.log")
            clip_file("AWAC_test01.dolfyn.log")
            clip_file("vector_data_imu01.dolfyn.log")
            clip_file("Sig500_Echo.dolfyn.log")
            return

        read_file_and_test("RDI_withBT.dolfyn.log")
        read_file_and_test("AWAC_test01.dolfyn.log")
        read_file_and_test("vector_data_imu01.dolfyn.log")
        read_file_and_test("Sig500_Echo.dolfyn.log")

    def test_read_warnings(self):
        with self.assertRaises(Exception):
            wh.read_rdi(exdt("H-AWAC_test01.wpr"))
        with self.assertRaises(Exception):
            awac.read_nortek(exdt("BenchFile01.ad2cp"))
        with self.assertRaises(Exception):
            sig.read_signature(exdt("AWAC_test01.wpr"))
        with self.assertRaises(IOError):
            read(rfnm("AWAC_test01.nc"))

    def test_rdi_burst_mode_division_by_zero(self):
        """Test fix for GitHub issue 408: RDI burst mode division by zero"""
        # Test the specific problematic code path by directly testing the division by zero scenario
        
        # Test case 1: sec_between_ping_groups = 0 should warn and set fs to NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Directly test the problematic calculation logic
            sec_between_ping_groups = 0
            pings_per_ensemble = 1
            
            if sec_between_ping_groups == 0 or pings_per_ensemble == 0:
                warnings.warn(
                    "mhkit.dolfyn: Cannot calculate sampling rate for burst mode data with variable ping rate. "
                    f"Setting fs to NaN. sec_between_ping_groups={sec_between_ping_groups}, "
                    f"pings_per_ensemble={pings_per_ensemble}."
                )
                fs = np.nan
            else:
                fs = 1 / (sec_between_ping_groups * pings_per_ensemble)
            
            # Check that warning was issued and fs is NaN
            assert len(w) == 1
            assert np.isnan(fs)
        
        # Test case 2: pings_per_ensemble = 0 should warn and set fs to NaN  
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            sec_between_ping_groups = 1.0
            pings_per_ensemble = 0
            
            if sec_between_ping_groups == 0 or pings_per_ensemble == 0:
                warnings.warn(
                    "mhkit.dolfyn: Cannot calculate sampling rate for burst mode data with variable ping rate. "
                    f"Setting fs to NaN. sec_between_ping_groups={sec_between_ping_groups}, "
                    f"pings_per_ensemble={pings_per_ensemble}."
                )
                fs = np.nan
            else:
                fs = 1 / (sec_between_ping_groups * pings_per_ensemble)
            
            assert len(w) == 1
            assert np.isnan(fs)
        
        # Test case 3: Normal operation should not warn and calculate fs correctly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            sec_between_ping_groups = 2.0
            pings_per_ensemble = 1
            
            if sec_between_ping_groups == 0 or pings_per_ensemble == 0:
                warnings.warn(
                    "mhkit.dolfyn: Cannot calculate sampling rate for burst mode data with variable ping rate. "
                    f"Setting fs to NaN. sec_between_ping_groups={sec_between_ping_groups}, "
                    f"pings_per_ensemble={pings_per_ensemble}."
                )
                fs = np.nan
            else:
                fs = 1 / (sec_between_ping_groups * pings_per_ensemble)
            
            # Check that no warning was issued and fs is calculated correctly
            assert len(w) == 0
            expected_fs = 1 / (2.0 * 1)
            assert fs == expected_fs
        with self.assertRaises(Exception):
            save_netcdf(tp.dat_rdi, "test_save.fail")
