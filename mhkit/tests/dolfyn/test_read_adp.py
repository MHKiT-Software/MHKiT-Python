from mhkit.tests.dolfyn.base import assert_allclose
from mhkit.tests.dolfyn import base as tb
import mhkit.dolfyn.io.nortek2 as sig
from mhkit.dolfyn.io.nortek2_lib import crop_ensembles
from mhkit.dolfyn.io.api import read_example as read
import warnings
import unittest
import pytest
import os
import numpy as np
from unittest.mock import patch

make_data = False
load = tb.load_netcdf
save = tb.save_netcdf

dat_rdi = load("RDI_test01.nc")
dat_rdi_7f79 = load("RDI_7f79.nc")
dat_rdi_7f79_2 = load("RDI_7f79_2.nc")
dat_rdi_bt = load("RDI_withBT.nc")
dat_vm_ws = load("vmdas01_wh.nc")
dat_vm_os = load("vmdas02_os.nc")
dat_wr1 = load("winriver01.nc")
dat_wr2 = load("winriver02.nc")
dat_rp = load("RiverPro_test01.nc")
dat_transect = load("winriver02_transect.nc")
dat_senb5 = load("sentinelv_b5.nc")

dat_awac = load("AWAC_test01.nc")
dat_awac_ud = load("AWAC_test01_ud.nc")
dat_hwac = load("H-AWAC_test01.nc")
dat_sig = load("BenchFile01.nc")
dat_sig_i = load("Sig1000_IMU.nc")
dat_sig_i_ud = load("Sig1000_IMU_ud.nc")
dat_sig_ieb = load("VelEchoBT01.nc")
dat_sig_ie = load("Sig500_Echo.nc")
dat_sig_tide = load("Sig1000_tidal.nc")
dat_sig_raw_avg = load("Sig100_raw_avg.nc")
dat_sig_avg = load("Sig100_avg.nc")
dat_sig_rt = load("Sig1000_online.nc")
dat_sig_skip = load("Sig_SkippedPings01.nc")
dat_sig_badt = load("Sig1000_BadTime01.nc")
dat_sig5_leiw = load("Sig500_last_ensemble_is_whole.nc")
dat_sig_dp1_all = load("Sig500_dp_ice1.nc")
dat_sig_dp1_ice = load("Sig500_dp_ice2.nc")
dat_sig_dp2_echo = load("Sig1000_dp_echo1.nc")
dat_sig_dp2_avg = load("Sig1000_dp_echo2.nc")


class io_adp_testcase(unittest.TestCase):
    def test_io_rdi(self):
        warnings.simplefilter("ignore", UserWarning)
        nens = 100
        td_rdi = read("RDI_test01.000")
        td_7f79 = read("RDI_7f79.000")
        td_7f79_2 = read("RDI_7f79_2.000")
        td_rdi_bt = read("RDI_withBT.000", nens=nens)
        td_vm = read("vmdas01_wh.ENX", nens=nens)
        td_os = read("vmdas02_os.ENR", nens=nens)
        td_wr1 = read("winriver01.PD0")
        td_wr2 = read("winriver02.PD0")
        td_rp = read("RiverPro_test01.PD0")
        td_transect = read("winriver02_transect.PD0", nens=nens)
        td_senb5 = read("sentinelv_b5.pd0")

        if make_data:
            save(td_rdi, "RDI_test01.nc")
            save(td_7f79, "RDI_7f79.nc")
            save(td_7f79_2, "RDI_7f79_2.nc")
            save(td_rdi_bt, "RDI_withBT.nc")
            save(td_vm, "vmdas01_wh.nc")
            save(td_os, "vmdas02_os.nc")
            save(td_wr1, "winriver01.nc")
            save(td_wr2, "winriver02.nc")
            save(td_rp, "RiverPro_test01.nc")
            save(td_transect, "winriver02_transect.nc")
            save(td_senb5, "sentinelv_b5.nc")
            return

        assert_allclose(td_rdi, dat_rdi, atol=1e-6)
        assert_allclose(td_7f79, dat_rdi_7f79, atol=1e-6)
        assert_allclose(td_7f79_2, dat_rdi_7f79_2, atol=1e-6)
        assert_allclose(td_rdi_bt, dat_rdi_bt, atol=1e-6)
        assert_allclose(td_vm, dat_vm_ws, atol=1e-6)
        assert_allclose(td_os, dat_vm_os, atol=1e-6)
        assert_allclose(td_wr1, dat_wr1, atol=1e-6)
        assert_allclose(td_wr2, dat_wr2, atol=1e-6)
        assert_allclose(td_rp, dat_rp, atol=1e-6)
        assert_allclose(td_transect, dat_transect, atol=1e-6)
        assert_allclose(td_senb5, dat_senb5, atol=1e-6)

    def test_rdi_burst_mode_division_by_zero(self):
        """Test fix for issue #408: RDI burst mode division by zero

        Issue #408 reported that RDI Pinnacle 45 in continuous burst mode
        sets sec_between_ping_groups=0 while pings_per_ensemble=1, causing
        ZeroDivisionError in sampling rate calculation.
        """
        # First verify normal operation with a regular RDI file
        td_rdi_normal = read("RDI_test01.000", nens=10)

        # Verify normal file has valid fs (not NaN)
        assert not np.isnan(td_rdi_normal.attrs["fs"])
        assert td_rdi_normal.attrs["fs"] > 0

        # Now test burst mode by patching the RDI reader to force burst mode config
        # Patch the finalize method to modify cfg before the fs calculation
        import mhkit.dolfyn.io.rdi as rdi_module

        original_finalize = rdi_module._RDIReader.finalize

        def mock_finalize_burst_mode(self, data, cfg):
            # Force burst mode configuration as reported in issue #408
            cfg["sec_between_ping_groups"] = 0
            cfg["pings_per_ensemble"] = 1
            return original_finalize(self, data, cfg)

        # Test burst mode scenario with patching
        with patch.object(rdi_module._RDIReader, "finalize", mock_finalize_burst_mode):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Read the same file but with forced burst mode config
                td_rdi_burst = read("RDI_test01.000", nens=10)

                # Check that warning was issued for burst mode
                assert len(w) > 0

                # Check that fs is set to NaN (not a crash)
                assert np.isnan(td_rdi_burst.attrs["fs"])

    def test_io_nortek(self):
        nens = 100
        with pytest.warns(UserWarning):
            td_awac = read("AWAC_test01.wpr", userdata=False, nens=[0, nens])
        td_awac_ud = read("AWAC_test01.wpr", nens=nens)
        td_hwac = read("H-AWAC_test01.wpr")

        if make_data:
            save(td_awac, "AWAC_test01.nc")
            save(td_awac_ud, "AWAC_test01_ud.nc")
            save(td_hwac, "H-AWAC_test01.nc")
            return

        assert_allclose(td_awac, dat_awac, atol=1e-6)
        assert_allclose(td_awac_ud, dat_awac_ud, atol=1e-6)
        assert_allclose(td_hwac, dat_hwac, atol=1e-6)

    def test_io_nortek2(self):
        nens = 100
        td_sig = read("BenchFile01.ad2cp", nens=nens, rebuild_index=True)
        td_sig_i = read(
            "Sig1000_IMU.ad2cp", userdata=False, nens=nens, rebuild_index=True
        )
        td_sig_i_ud = read("Sig1000_IMU.ad2cp", nens=nens, rebuild_index=True)
        td_sig_ieb = read("VelEchoBT01.ad2cp", nens=nens, rebuild_index=True)
        td_sig_ie = read("Sig500_Echo.ad2cp", nens=nens, rebuild_index=True)
        td_sig_tide = read("Sig1000_tidal.ad2cp", nens=nens, rebuild_index=True)
        td_sig_raw_avg = read("Sig100_raw_avg.ad2cp", nens=nens, rebuild_index=True)
        td_sig_avg = read("Sig100_avg.ad2cp", nens=nens, rebuild_index=True)
        td_sig_rt = read("Sig1000_online.ad2cp", nens=nens, rebuild_index=True)
        td_sig_dp1_all, td_sig_dp1_ice = read("Sig500_dp_ice.ad2cp", rebuild_index=True)
        td_sig_dp2_echo, td_sig_dp2_avg = read(
            "Sig1000_dp_echo.ad2cp", rebuild_index=True
        )

        with pytest.warns(UserWarning):
            # This issues a warning...
            td_sig_skip = read("Sig_SkippedPings01.ad2cp")

        with pytest.warns(UserWarning):
            td_sig_badt = sig.read_signature(tb.rfnm("Sig1000_BadTime01.ad2cp"))

        # Make sure we read all the way to the end of the file.
        # This file ends exactly at the end of an ensemble.
        td_sig5_leiw = read("Sig500_last_ensemble_is_whole.ad2cp")

        os.remove(tb.exdt("BenchFile01.ad2cp.index"))
        os.remove(tb.exdt("Sig1000_IMU.ad2cp.index"))
        os.remove(tb.exdt("VelEchoBT01.ad2cp.index"))
        os.remove(tb.exdt("Sig500_Echo.ad2cp.index"))
        os.remove(tb.exdt("Sig1000_tidal.ad2cp.index"))
        os.remove(tb.exdt("Sig100_raw_avg.ad2cp.index"))
        os.remove(tb.exdt("Sig100_avg.ad2cp.index"))
        os.remove(tb.exdt("Sig1000_online.ad2cp.index"))
        os.remove(tb.exdt("Sig_SkippedPings01.ad2cp.index"))
        os.remove(tb.exdt("Sig500_last_ensemble_is_whole.ad2cp.index"))
        os.remove(tb.rfnm("Sig1000_BadTime01.ad2cp.index"))
        os.remove(tb.exdt("Sig500_dp_ice.ad2cp.index"))
        os.remove(tb.exdt("Sig1000_dp_echo.ad2cp.index"))

        if make_data:
            save(td_sig, "BenchFile01.nc")
            save(td_sig_i, "Sig1000_IMU.nc")
            save(td_sig_i_ud, "Sig1000_IMU_ud.nc")
            save(td_sig_ieb, "VelEchoBT01.nc")
            save(td_sig_ie, "Sig500_Echo.nc")
            save(td_sig_tide, "Sig1000_tidal.nc")
            save(td_sig_raw_avg, "Sig100_raw_avg.nc")
            save(td_sig_avg, "Sig100_avg.nc")
            save(td_sig_rt, "Sig1000_online.nc")
            save(td_sig_skip, "Sig_SkippedPings01.nc")
            save(td_sig_badt, "Sig1000_BadTime01.nc")
            save(td_sig5_leiw, "Sig500_last_ensemble_is_whole.nc")
            save(td_sig_dp1_all, "Sig500_dp_ice1.nc")
            save(td_sig_dp1_ice, "Sig500_dp_ice2.nc")
            save(td_sig_dp2_echo, "Sig1000_dp_echo1.nc")
            save(td_sig_dp2_avg, "Sig1000_dp_echo2.nc")
            return

        assert_allclose(td_sig, dat_sig, atol=1e-6)
        assert_allclose(td_sig_i, dat_sig_i, atol=1e-6)
        assert_allclose(td_sig_i_ud, dat_sig_i_ud, atol=1e-6)
        assert_allclose(td_sig_ieb, dat_sig_ieb, atol=1e-6)
        assert_allclose(td_sig_ie, dat_sig_ie, atol=1e-6)
        assert_allclose(td_sig_tide, dat_sig_tide, atol=1e-6)
        assert_allclose(td_sig_raw_avg, dat_sig_raw_avg, atol=1e-6)
        assert_allclose(td_sig_avg, dat_sig_avg, atol=1e-6)
        assert_allclose(td_sig_rt, dat_sig_rt, atol=1e-6)
        assert_allclose(td_sig5_leiw, dat_sig5_leiw, atol=1e-6)
        assert_allclose(td_sig_skip, dat_sig_skip, atol=1e-6)
        assert_allclose(td_sig_badt, dat_sig_badt, atol=1e-6)
        assert_allclose(td_sig_dp1_all, dat_sig_dp1_all, atol=1e-6)
        assert_allclose(td_sig_dp1_ice, dat_sig_dp1_ice, atol=1e-6)
        assert_allclose(td_sig_dp2_echo, dat_sig_dp2_echo, atol=1e-6)
        assert_allclose(td_sig_dp2_avg, dat_sig_dp2_avg, atol=1e-6)

    def test_nortek2_crop(self):
        # Test file cropping function
        crop_ensembles(
            infile=tb.exdt("Sig500_Echo.ad2cp"),
            outfile=tb.exdt("Sig500_Echo_crop.ad2cp"),
            range=[50, 100],
        )
        td_sig_ie_crop = read("Sig500_Echo_crop.ad2cp")

        crop_ensembles(
            infile=tb.exdt("BenchFile01.ad2cp"),
            outfile=tb.exdt("BenchFile01_crop.ad2cp"),
            range=[50, 100],
        )
        td_sig_crop = read("BenchFile01_crop.ad2cp")

        if make_data:
            save(td_sig_ie_crop, "Sig500_Echo_crop.nc")
            save(td_sig_crop, "BenchFile01_crop.nc")
            return

        os.remove(tb.exdt("Sig500_Echo.ad2cp.index"))
        os.remove(tb.exdt("Sig500_Echo_crop.ad2cp"))
        os.remove(tb.exdt("Sig500_Echo_crop.ad2cp.index"))
        os.remove(tb.exdt("BenchFile01.ad2cp.index"))
        os.remove(tb.exdt("BenchFile01_crop.ad2cp"))
        os.remove(tb.exdt("BenchFile01_crop.ad2cp.index"))

        cd_sig_ie_crop = load("Sig500_Echo_crop.nc")
        cd_sig_crop = load("BenchFile01_crop.nc")

        assert_allclose(td_sig_ie_crop, cd_sig_ie_crop, atol=1e-6)
        assert_allclose(td_sig_crop, cd_sig_crop, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
