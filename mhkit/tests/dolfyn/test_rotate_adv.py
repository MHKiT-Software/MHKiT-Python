from . import test_read_adv as tr
from .base import load_netcdf as load, save_netcdf as save, assert_allclose
from mhkit.dolfyn.rotate.api import (
    rotate2,
    calc_principal_heading,
    set_declination,
    set_inst2head_rotmat,
)
from mhkit.dolfyn.rotate.base import euler2orient, orient2euler
import numpy as np
import numpy.testing as npt
import unittest

make_data = False


class rotate_adv_testcase(unittest.TestCase):
    def test_heading(self):
        td = tr.dat_imu.copy(deep=True)

        head, pitch, roll = orient2euler(td)
        td["pitch"].values = pitch
        td["roll"].values = roll
        td["heading"].values = head

        if make_data:
            save(td, "vector_data_imu01_head_pitch_roll.nc")
            return
        cd = load("vector_data_imu01_head_pitch_roll.nc")

        assert_allclose(td, cd, atol=1e-6)

    def test_inst2head_rotmat(self):
        # Validated test
        td = tr.dat.copy(deep=True)

        # Swap x,y, reverse z
        set_inst2head_rotmat(td, [[0, 1, 0], [1, 0, 0], [0, 0, -1]], inplace=True)

        # Coords don't get altered here
        npt.assert_allclose(td.vel[0].values, tr.dat.vel[1].values, atol=1e-6)
        npt.assert_allclose(td.vel[1].values, tr.dat.vel[0].values, atol=1e-6)
        npt.assert_allclose(td.vel[2].values, -tr.dat.vel[2].values, atol=1e-6)

        # Validation for non-symmetric rotations
        td = tr.dat.copy(deep=True)
        R = euler2orient(20, 30, 60, units="degrees")  # arbitrary angles
        td = set_inst2head_rotmat(td, R, inplace=False)
        vel1 = td.vel
        # validate that a head->inst rotation occurs (transpose of inst2head_rotmat)
        vel2 = np.dot(R, tr.dat.vel)

        npt.assert_allclose(vel1.values, vel2, atol=1e-6)

    def test_rotate_inst2earth(self):
        td = tr.dat.copy(deep=True)
        rotate2(td, "earth", inplace=True)
        tdm = tr.dat_imu.copy(deep=True)
        rotate2(tdm, "earth", inplace=True)
        tdo = tr.dat.copy(deep=True)
        omat = tdo["orientmat"]
        tdo = rotate2(tdo.drop_vars("orientmat"), "earth", inplace=False)
        tdo["orientmat"] = omat

        if make_data:
            save(td, "vector_data01_rotate_inst2earth.nc")
            save(tdm, "vector_data_imu01_rotate_inst2earth.nc")
            return

        cd = load("vector_data01_rotate_inst2earth.nc")
        cdm = load("vector_data_imu01_rotate_inst2earth.nc")

        assert_allclose(td, cd, atol=1e-6)
        assert_allclose(tdm, cdm, atol=1e-6)
        assert_allclose(tdo, cd, atol=1e-6)

    def test_rotate_earth2inst(self):
        td = load("vector_data01_rotate_inst2earth.nc")
        rotate2(td, "inst", inplace=True)
        tdm = load("vector_data_imu01_rotate_inst2earth.nc")
        rotate2(tdm, "inst", inplace=True)

        cd = tr.dat.copy(deep=True)
        cdm = tr.dat_imu.copy(deep=True)
        # The heading/pitch/roll data gets modified during rotation, so it
        # doesn't go back to what it was.
        cdm = cdm.drop_vars(["heading", "pitch", "roll"])
        tdm = tdm.drop_vars(["heading", "pitch", "roll"])

        assert_allclose(td, cd, atol=1e-6)
        assert_allclose(tdm, cdm, atol=1e-6)

    def test_rotate_inst2beam(self):
        td = tr.dat.copy(deep=True)
        rotate2(td, "beam", inplace=True)
        tdm = tr.dat_imu.copy(deep=True)
        rotate2(tdm, "beam", inplace=True)

        if make_data:
            save(td, "vector_data01_rotate_inst2beam.nc")
            save(tdm, "vector_data_imu01_rotate_inst2beam.nc")
            return

        cd = load("vector_data01_rotate_inst2beam.nc")
        cdm = load("vector_data_imu01_rotate_inst2beam.nc")

        assert_allclose(td, cd, atol=1e-6)
        assert_allclose(tdm, cdm, atol=1e-6)

    def test_rotate_beam2inst(self):
        td = load("vector_data01_rotate_inst2beam.nc")
        rotate2(td, "inst", inplace=True)
        tdm = load("vector_data_imu01_rotate_inst2beam.nc")
        rotate2(tdm, "inst", inplace=True)

        cd = tr.dat.copy(deep=True)
        cdm = tr.dat_imu.copy(deep=True)

        assert_allclose(td, cd, atol=1e-5)
        assert_allclose(tdm, cdm, atol=1e-5)

    def test_rotate_earth2principal(self):
        td = load("vector_data01_rotate_inst2earth.nc")
        td.attrs["principal_heading"] = calc_principal_heading(td["vel"])
        rotate2(td, "principal", inplace=True)
        tdm = load("vector_data_imu01_rotate_inst2earth.nc")
        tdm.attrs["principal_heading"] = calc_principal_heading(tdm["vel"])
        rotate2(tdm, "principal", inplace=True)

        if make_data:
            save(td, "vector_data01_rotate_earth2principal.nc")
            save(tdm, "vector_data_imu01_rotate_earth2principal.nc")
            return

        cd = load("vector_data01_rotate_earth2principal.nc")
        cdm = load("vector_data_imu01_rotate_earth2principal.nc")

        assert_allclose(td, cd, atol=1e-6)
        assert_allclose(tdm, cdm, atol=1e-6)

    def test_rotate_earth2principal_set_declination(self):
        declin = 3.875
        td = load("vector_data01_rotate_inst2earth.nc")
        td0 = td.copy(deep=True)

        td.attrs["principal_heading"] = calc_principal_heading(td["vel"])
        rotate2(td, "principal", inplace=True)
        set_declination(td, declin, inplace=True)
        rotate2(td, "earth", inplace=True)

        set_declination(td0, -1, inplace=True)
        set_declination(td0, declin, inplace=True)
        td0.attrs["principal_heading"] = calc_principal_heading(td0["vel"])
        rotate2(td0, "earth", inplace=True)

        assert_allclose(td0, td, atol=1e-6)

    def test_rotate_warnings(self):
        warn1 = tr.dat.copy(deep=True)
        warn2 = tr.dat.copy(deep=True)
        warn2.attrs["coord_sys"] = "flow"
        warn3 = tr.dat.copy(deep=True)
        warn3.attrs["inst_model"] = "ADV"
        warn4 = tr.dat.copy(deep=True)
        warn4.attrs["inst_model"] = "adv"

        with self.assertRaises(Exception):
            rotate2(warn1, "ship")
        with self.assertRaises(Exception):
            rotate2(warn2, "earth")
        with self.assertRaises(Exception):
            set_inst2head_rotmat(warn3, np.eye(3))
        with self.assertRaises(Exception):
            set_inst2head_rotmat(warn4, np.eye(3))


if __name__ == "__main__":
    unittest.main()
