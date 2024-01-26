from . import test_read_adp as tr
from .base import load_netcdf as load, save_netcdf as save, assert_allclose
from mhkit.dolfyn.rotate.api import rotate2, calc_principal_heading
import numpy as np
import numpy.testing as npt
import unittest

make_data = False


class rotate_adp_testcase(unittest.TestCase):
    def test_rotate_beam2inst(self):
        td_rdi = rotate2(tr.dat_rdi, "inst", inplace=False)
        td_sig = rotate2(tr.dat_sig, "inst", inplace=False)
        td_sig_i = rotate2(tr.dat_sig_i, "inst", inplace=False)
        td_sig_ieb = rotate2(tr.dat_sig_ieb, "inst", inplace=False)

        if make_data:
            save(td_rdi, "RDI_test01_rotate_beam2inst.nc")
            save(td_sig, "BenchFile01_rotate_beam2inst.nc")
            save(td_sig_i, "Sig1000_IMU_rotate_beam2inst.nc")
            save(td_sig_ieb, "VelEchoBT01_rotate_beam2inst.nc")
            return

        cd_rdi = load("RDI_test01_rotate_beam2inst.nc")
        cd_sig = load("BenchFile01_rotate_beam2inst.nc")
        cd_sig_i = load("Sig1000_IMU_rotate_beam2inst.nc")
        cd_sig_ieb = load("VelEchoBT01_rotate_beam2inst.nc")

        assert_allclose(td_rdi, cd_rdi, atol=1e-5)
        assert_allclose(td_sig, cd_sig, atol=1e-5)
        assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
        assert_allclose(td_sig_ieb, cd_sig_ieb, atol=1e-5)

    def test_rotate_inst2beam(self):
        td = load("RDI_test01_rotate_beam2inst.nc")
        rotate2(td, "beam", inplace=True)
        td_awac = load("AWAC_test01_earth2inst.nc")
        rotate2(td_awac, "beam", inplace=True)
        td_sig = load("BenchFile01_rotate_beam2inst.nc")
        rotate2(td_sig, "beam", inplace=True)
        td_sig_i = load("Sig1000_IMU_rotate_beam2inst.nc")
        rotate2(td_sig_i, "beam", inplace=True)
        td_sig_ie = load("Sig500_Echo_earth2inst.nc")
        rotate2(td_sig_ie, "beam", inplace=True)

        if make_data:
            save(td_awac, "AWAC_test01_inst2beam.nc")
            save(td_sig_ie, "Sig500_Echo_inst2beam.nc")
            return

        cd_td = tr.dat_rdi.copy(deep=True)
        cd_awac = load("AWAC_test01_inst2beam.nc")
        cd_sig = tr.dat_sig.copy(deep=True)
        cd_sig_i = tr.dat_sig_i.copy(deep=True)
        cd_sig_ie = load("Sig500_Echo_inst2beam.nc")

        # # The reverse RDI rotation doesn't work b/c of NaN's in one beam
        # # that propagate to others, so we impose that here.
        cd_td["vel"].values[:, np.isnan(cd_td["vel"].values).any(0)] = np.NaN

        assert_allclose(td, cd_td, atol=1e-5)
        assert_allclose(td_awac, cd_awac, atol=1e-5)
        assert_allclose(td_sig, cd_sig, atol=1e-5)
        assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
        assert_allclose(td_sig_ie, cd_sig_ie, atol=1e-5)

    def test_rotate_inst2earth(self):
        # AWAC & Sig500 are loaded in earth
        td_awac = tr.dat_awac.copy(deep=True)
        rotate2(td_awac, "inst", inplace=True)
        td_sig_ie = tr.dat_sig_ie.copy(deep=True)
        rotate2(td_sig_ie, "inst", inplace=True)
        td_sig_o = td_sig_ie.copy(deep=True)

        td = rotate2(tr.dat_rdi, "earth", inplace=False)
        tdwr2 = rotate2(tr.dat_wr2, "earth", inplace=False)
        td_sig = load("BenchFile01_rotate_beam2inst.nc")
        rotate2(td_sig, "earth", inplace=True)
        td_sig_i = load("Sig1000_IMU_rotate_beam2inst.nc")
        rotate2(td_sig_i, "earth", inplace=True)

        if make_data:
            save(td_awac, "AWAC_test01_earth2inst.nc")
            save(td, "RDI_test01_rotate_inst2earth.nc")
            save(tdwr2, "winriver02_rotate_ship2earth.nc")
            save(td_sig, "BenchFile01_rotate_inst2earth.nc")
            save(td_sig_i, "Sig1000_IMU_rotate_inst2earth.nc")
            save(td_sig_ie, "Sig500_Echo_earth2inst.nc")
            return

        td_awac = rotate2(load("AWAC_test01_earth2inst.nc"), "earth", inplace=False)
        td_sig_ie = rotate2(load("Sig500_Echo_earth2inst.nc"), "earth", inplace=False)
        td_sig_o = rotate2(td_sig_o.drop_vars("orientmat"), "earth", inplace=False)

        cd = load("RDI_test01_rotate_inst2earth.nc")
        cdwr2 = load("winriver02_rotate_ship2earth.nc")
        cd_sig = load("BenchFile01_rotate_inst2earth.nc")
        cd_sig_i = load("Sig1000_IMU_rotate_inst2earth.nc")

        assert_allclose(td, cd, atol=1e-5)
        assert_allclose(tdwr2, cdwr2, atol=1e-5)
        assert_allclose(td_awac, tr.dat_awac, atol=1e-5)
        assert_allclose(td_sig, cd_sig, atol=1e-5)
        assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
        assert_allclose(td_sig_ie, tr.dat_sig_ie, atol=1e-5)
        npt.assert_allclose(td_sig_o.vel, tr.dat_sig_ie.vel, atol=1e-5)

    def test_rotate_earth2inst(self):
        td_rdi = load("RDI_test01_rotate_inst2earth.nc")
        rotate2(td_rdi, "inst", inplace=True)
        tdwr2 = load("winriver02_rotate_ship2earth.nc")
        rotate2(tdwr2, "inst", inplace=True)

        td_awac = tr.dat_awac.copy(deep=True)
        rotate2(td_awac, "inst", inplace=True)  # AWAC is in earth coords
        td_sig = load("BenchFile01_rotate_inst2earth.nc")
        rotate2(td_sig, "inst", inplace=True)
        td_sig_i = load("Sig1000_IMU_rotate_inst2earth.nc")
        rotate2(td_sig_i, "inst", inplace=True)

        cd_rdi = load("RDI_test01_rotate_beam2inst.nc")
        cd_wr2 = tr.dat_wr2
        # ship and inst are considered equivalent in dolfy
        cd_wr2.attrs["coord_sys"] = "inst"
        cd_awac = load("AWAC_test01_earth2inst.nc")
        cd_sig = load("BenchFile01_rotate_beam2inst.nc")
        cd_sig_i = load("Sig1000_IMU_rotate_beam2inst.nc")

        assert_allclose(td_rdi, cd_rdi, atol=1e-5)
        assert_allclose(tdwr2, cd_wr2, atol=1e-5)
        assert_allclose(td_awac, cd_awac, atol=1e-5)
        assert_allclose(td_sig, cd_sig, atol=1e-5)
        # known failure due to orientmat, see test_vs_nortek
        # assert_allclose(td_sig_i, cd_sig_i, atol=1e-3)
        npt.assert_allclose(td_sig_i.accel.values, cd_sig_i.accel.values, atol=1e-3)

    def test_rotate_earth2principal(self):
        td_rdi = load("RDI_test01_rotate_inst2earth.nc")
        td_sig = load("BenchFile01_rotate_inst2earth.nc")
        td_awac = tr.dat_awac.copy(deep=True)

        td_rdi.attrs["principal_heading"] = calc_principal_heading(
            td_rdi.vel.mean("range")
        )
        td_sig.attrs["principal_heading"] = calc_principal_heading(
            td_sig.vel.mean("range")
        )
        td_awac.attrs["principal_heading"] = calc_principal_heading(
            td_awac.vel.mean("range"), tidal_mode=False
        )
        rotate2(td_rdi, "principal", inplace=True)
        rotate2(td_sig, "principal", inplace=True)
        rotate2(td_awac, "principal", inplace=True)

        if make_data:
            save(td_rdi, "RDI_test01_rotate_earth2principal.nc")
            save(td_sig, "BenchFile01_rotate_earth2principal.nc")
            save(td_awac, "AWAC_test01_earth2principal.nc")
            return

        cd_rdi = load("RDI_test01_rotate_earth2principal.nc")
        cd_sig = load("BenchFile01_rotate_earth2principal.nc")
        cd_awac = load("AWAC_test01_earth2principal.nc")

        assert_allclose(td_rdi, cd_rdi, atol=1e-5)
        assert_allclose(td_awac, cd_awac, atol=1e-5)
        assert_allclose(td_sig, cd_sig, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
