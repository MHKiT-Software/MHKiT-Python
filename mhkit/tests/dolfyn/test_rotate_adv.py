from . import test_read_adv as tr
from .base import load_ncdata as load, save_ncdata as save, assert_allclose
from mhkit.dolfyn.rotate.api import rotate2, calc_principal_heading, \
    set_declination, set_inst2head_rotmat
from mhkit.dolfyn.rotate.base import euler2orient, orient2euler
import numpy.testing as npt
import numpy as np
import unittest


def test_heading(make_data=False):
    td = tr.dat_imu.copy(deep=True)

    head, pitch, roll = orient2euler(td)
    td['pitch'].values = pitch
    td['roll'].values = roll
    td['heading'].values = head

    if make_data:
        save(td, 'vector_data_imu01_head_pitch_roll.nc')
        return
    cd = load('vector_data_imu01_head_pitch_roll.nc')

    assert_allclose(td, cd, atol=1e-6)


def test_inst2head_rotmat():
    # Validated test
    td = tr.dat.copy(deep=True)

    # Swap x,y, reverse z
    td = set_inst2head_rotmat(td, [[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, -1]])

    # Coords don't get altered here
    npt.assert_allclose(td.vel[0].values, tr.dat.vel[1].values, atol=1e-6)
    npt.assert_allclose(td.vel[1].values, tr.dat.vel[0].values, atol=1e-6)
    npt.assert_allclose(td.vel[2].values, -tr.dat.vel[2].values, atol=1e-6)

    # Validation for non-symmetric rotations
    td = tr.dat.copy(deep=True)
    R = euler2orient(20, 30, 60, units='degrees')  # arbitrary angles
    td = set_inst2head_rotmat(td, R)
    vel1 = td.vel
    # validate that a head->inst rotation occurs (transpose of inst2head_rotmat)
    vel2 = np.dot(R, tr.dat.vel)
    #assert (vel1 == vel2).all(), "head->inst rotations give unexpeced results."
    npt.assert_allclose(vel1.values, vel2, atol=1e-6)


def test_rotate_inst2earth(make_data=False):
    td = tr.dat.copy(deep=True)
    td = rotate2(td, 'earth')
    tdm = tr.dat_imu.copy(deep=True)
    tdm = rotate2(tdm, 'earth')
    tdo = tr.dat.copy(deep=True)
    tdo = rotate2(tdo.drop_vars('orientmat'), 'earth')

    if make_data:
        save(td, 'vector_data01_rotate_inst2earth.nc')
        save(tdm, 'vector_data_imu01_rotate_inst2earth.nc')
        return

    cd = load('vector_data01_rotate_inst2earth.nc')
    cdm = load('vector_data_imu01_rotate_inst2earth.nc')

    assert_allclose(td, cd, atol=1e-6)
    assert_allclose(tdm, cdm, atol=1e-6)
    assert_allclose(td, cd, atol=1e-6)


def test_rotate_earth2inst():
    td = load('vector_data01_rotate_inst2earth.nc')
    td = rotate2(td, 'inst')
    tdm = load('vector_data_imu01_rotate_inst2earth.nc')
    tdm = rotate2(tdm, 'inst')

    cd = tr.dat.copy(deep=True)
    cdm = tr.dat_imu.copy(deep=True)
    # The heading/pitch/roll data gets modified during rotation, so it
    # doesn't go back to what it was.
    cdm = cdm.drop_vars(['heading', 'pitch', 'roll'])
    tdm = tdm.drop_vars(['heading', 'pitch', 'roll'])

    assert_allclose(td, cd, atol=1e-6)
    assert_allclose(tdm, cdm, atol=1e-6)


def test_rotate_inst2beam(make_data=False):
    td = tr.dat.copy(deep=True)
    td = rotate2(td, 'beam')
    tdm = tr.dat_imu.copy(deep=True)
    tdm = rotate2(tdm, 'beam')

    if make_data:
        save(td, 'vector_data01_rotate_inst2beam.nc')
        save(tdm, 'vector_data_imu01_rotate_inst2beam.nc')
        return

    cd = load('vector_data01_rotate_inst2beam.nc')
    cdm = load('vector_data_imu01_rotate_inst2beam.nc')

    assert_allclose(td, cd, atol=1e-6)
    assert_allclose(tdm, cdm, atol=1e-6)


def test_rotate_beam2inst():
    td = load('vector_data01_rotate_inst2beam.nc')
    td = rotate2(td, 'inst')
    tdm = load('vector_data_imu01_rotate_inst2beam.nc')
    tdm = rotate2(tdm, 'inst')

    cd = tr.dat.copy(deep=True)
    cdm = tr.dat_imu.copy(deep=True)

    assert_allclose(td, cd, atol=1e-6)
    assert_allclose(tdm, cdm, atol=1e-6)


def test_rotate_earth2principal(make_data=False):
    td = load('vector_data01_rotate_inst2earth.nc')
    td.attrs['principal_heading'] = calc_principal_heading(td['vel'])
    td = rotate2(td, 'principal')
    tdm = load('vector_data_imu01_rotate_inst2earth.nc')
    tdm.attrs['principal_heading'] = calc_principal_heading(tdm['vel'])
    tdm = rotate2(tdm, 'principal')

    if make_data:
        save(td, 'vector_data01_rotate_earth2principal.nc')
        save(tdm, 'vector_data_imu01_rotate_earth2principal.nc')
        return

    cd = load('vector_data01_rotate_earth2principal.nc')
    cdm = load('vector_data_imu01_rotate_earth2principal.nc')

    assert_allclose(td, cd, atol=1e-6)
    assert_allclose(tdm, cdm, atol=1e-6)


def test_rotate_earth2principal_set_declination():
    declin = 3.875
    td = load('vector_data01_rotate_inst2earth.nc')
    td0 = td.copy(deep=True)

    td.attrs['principal_heading'] = calc_principal_heading(td['vel'])
    td = rotate2(td, 'principal')
    td = set_declination(td, declin)
    td = rotate2(td, 'earth')

    td0 = set_declination(td0, -1)
    td0 = set_declination(td0, declin)
    td0.attrs['principal_heading'] = calc_principal_heading(td0['vel'])
    td0 = rotate2(td0, 'earth')

    assert_allclose(td0, td, atol=1e-6)


class warnings_testcase(unittest.TestCase):
    def test_rotate_warnings(self):
        warn1 = tr.dat.copy(deep=True)
        warn2 = tr.dat.copy(deep=True)
        warn2.attrs['coord_sys'] = 'flow'
        warn3 = tr.dat.copy(deep=True)
        warn3.attrs['inst_model'] = 'ADV'
        warn4 = tr.dat.copy(deep=True)
        warn4.attrs['inst_model'] = 'adv'

        with self.assertRaises(Exception):
            rotate2(warn1, 'ship')
        with self.assertRaises(Exception):
            rotate2(warn2, 'earth')
        with self.assertRaises(Exception):
            set_inst2head_rotmat(warn3, np.eye(3))
            set_inst2head_rotmat(warn3, np.eye(3))
        with self.assertRaises(Exception):
            set_inst2head_rotmat(warn4, np.eye(3))


if __name__ == '__main__':
    test_heading()
    test_inst2head_rotmat()
    test_rotate_inst2earth()
    test_rotate_earth2inst()
    test_rotate_beam2inst()
    test_rotate_inst2beam()
    test_rotate_earth2principal()
    test_rotate_earth2principal_set_declination()
    unittest.main()
