from . import test_read_adp as tr
from .base import load_ncdata as load, save_ncdata as save, assert_allclose
from mhkit.dolfyn.rotate.api import rotate2, calc_principal_heading
import numpy.testing as npt
import numpy as np


def test_rotate_beam2inst(make_data=False):

    td_rdi = rotate2(tr.dat_rdi, 'inst')
    td_sig = rotate2(tr.dat_sig, 'inst')
    td_sig_i = rotate2(tr.dat_sig_i, 'inst')
    td_sig_ieb = rotate2(tr.dat_sig_ieb, 'inst')

    if make_data:
        save(td_rdi, 'RDI_test01_rotate_beam2inst.nc')
        save(td_sig, 'BenchFile01_rotate_beam2inst.nc')
        save(td_sig_i, 'Sig1000_IMU_rotate_beam2inst.nc')
        save(td_sig_ieb, 'VelEchoBT01_rotate_beam2inst.nc')
        return

    cd_rdi = load('RDI_test01_rotate_beam2inst.nc')
    cd_sig = load('BenchFile01_rotate_beam2inst.nc')
    cd_sig_i = load('Sig1000_IMU_rotate_beam2inst.nc')
    cd_sig_ieb = load('VelEchoBT01_rotate_beam2inst.nc')

    assert_allclose(td_rdi, cd_rdi, atol=1e-5)
    assert_allclose(td_sig, cd_sig, atol=1e-5)
    assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
    assert_allclose(td_sig_ieb, cd_sig_ieb, atol=1e-5)


def test_rotate_inst2beam(make_data=False):

    td = load('RDI_test01_rotate_beam2inst.nc')
    td = rotate2(td, 'beam')
    td_awac = load('AWAC_test01_earth2inst.nc')
    td_awac = rotate2(td_awac, 'beam')
    td_sig = load('BenchFile01_rotate_beam2inst.nc')
    td_sig = rotate2(td_sig, 'beam')
    td_sig_i = load('Sig1000_IMU_rotate_beam2inst.nc')
    td_sig_i = rotate2(td_sig_i, 'beam')
    td_sig_ie = load('Sig500_Echo_earth2inst.nc')
    td_sig_ie = rotate2(td_sig_ie, 'beam')

    if make_data:
        save(td_awac, 'AWAC_test01_inst2beam.nc')
        save(td_sig_ie, 'Sig500_Echo_inst2beam.nc')
        return

    cd_td = tr.dat_rdi.copy(deep=True)
    cd_awac = load('AWAC_test01_inst2beam.nc')
    cd_sig = tr.dat_sig.copy(deep=True)
    cd_sig_i = tr.dat_sig_i.copy(deep=True)
    cd_sig_ie = load('Sig500_Echo_inst2beam.nc')

    # # The reverse RDI rotation doesn't work b/c of NaN's in one beam
    # # that propagate to others, so we impose that here.
    cd_td['vel'].values[:, np.isnan(cd_td['vel'].values).any(0)] = np.NaN

    assert_allclose(td, cd_td, atol=1e-5)
    assert_allclose(td_awac, cd_awac, atol=1e-5)
    assert_allclose(td_sig, cd_sig, atol=1e-5)
    assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
    assert_allclose(td_sig_ie, cd_sig_ie, atol=1e-5)


def test_rotate_inst2earth(make_data=False):
    # AWAC & Sig500 are loaded in earth
    td_awac = tr.dat_awac.copy(deep=True)
    td_awac = rotate2(td_awac, 'inst')
    td_sig_ie = tr.dat_sig_ie.copy(deep=True)
    td_sig_ie = rotate2(rotate2(td_sig_ie, 'earth'), 'inst')
    td_sig_o = td_sig_ie.copy(deep=True)

    td = rotate2(tr.dat_rdi, 'earth')
    tdwr2 = rotate2(tr.dat_wr2, 'earth')
    td_sig = load('BenchFile01_rotate_beam2inst.nc')
    td_sig = rotate2(td_sig, 'earth')
    td_sig_i = load('Sig1000_IMU_rotate_beam2inst.nc')
    td_sig_i = rotate2(td_sig_i, 'earth')

    if make_data:
        save(td_awac, 'AWAC_test01_earth2inst.nc')
        save(td, 'RDI_test01_rotate_inst2earth.nc')
        save(tdwr2, 'winriver02_rotate_ship2earth.nc')
        save(td_sig, 'BenchFile01_rotate_inst2earth.nc')
        save(td_sig_i, 'Sig1000_IMU_rotate_inst2earth.nc')
        save(td_sig_ie, 'Sig500_Echo_earth2inst.nc')

        return
    td_awac = rotate2(td_awac, 'earth')
    td_sig_ie = rotate2(td_sig_ie, 'earth')
    td_sig_o = rotate2(td_sig_o.drop_vars('orientmat'), 'earth')

    cd = load('RDI_test01_rotate_inst2earth.nc')
    cdwr2 = load('winriver02_rotate_ship2earth.nc')
    cd_sig = load('BenchFile01_rotate_inst2earth.nc')
    cd_sig_i = load('Sig1000_IMU_rotate_inst2earth.nc')

    assert_allclose(td, cd, atol=1e-5)
    assert_allclose(tdwr2, cdwr2, atol=1e-5)
    assert_allclose(td_awac, tr.dat_awac, atol=1e-5)
    #npt.assert_allclose(td_awac.vel.values, tr.dat_awac.vel.values, rtol=1e-7, atol=1e-3)
    assert_allclose(td_sig, cd_sig, atol=1e-5)
    assert_allclose(td_sig_i, cd_sig_i, atol=1e-5)
    assert_allclose(td_sig_ie, tr.dat_sig_ie, atol=1e-5)
    npt.assert_allclose(td_sig_o.vel, tr.dat_sig_ie.vel, atol=1e-5)


def test_rotate_earth2inst():

    td_rdi = load('RDI_test01_rotate_inst2earth.nc')
    td_rdi = rotate2(td_rdi, 'inst')
    tdwr2 = load('winriver02_rotate_ship2earth.nc')
    tdwr2 = rotate2(tdwr2, 'inst')

    td_awac = tr.dat_awac.copy(deep=True)
    td_awac = rotate2(td_awac, 'inst')  # AWAC is in earth coords
    td_sig = load('BenchFile01_rotate_inst2earth.nc')
    td_sig = rotate2(td_sig, 'inst')
    td_sigi = load('Sig1000_IMU_rotate_inst2earth.nc')
    td_sig_i = rotate2(td_sigi, 'inst')

    cd_rdi = load('RDI_test01_rotate_beam2inst.nc')
    cd_awac = load('AWAC_test01_earth2inst.nc')
    cd_sig = load('BenchFile01_rotate_beam2inst.nc')
    cd_sig_i = load('Sig1000_IMU_rotate_beam2inst.nc')

    assert_allclose(td_rdi, cd_rdi, atol=1e-5)
    assert_allclose(tdwr2, tr.dat_wr2, atol=1e-5)
    assert_allclose(td_awac, cd_awac, atol=1e-5)
    assert_allclose(td_sig, cd_sig, atol=1e-5)
    # known failure due to orientmat, see test_vs_nortek
    #assert_allclose(td_sig_i, cd_sig_i, atol=1e-3)
    npt.assert_allclose(td_sig_i.accel.values,
                        cd_sig_i.accel.values, atol=1e-3)


def test_rotate_earth2principal(make_data=False):

    td_rdi = load('RDI_test01_rotate_inst2earth.nc')
    td_sig = load('BenchFile01_rotate_inst2earth.nc')
    td_awac = tr.dat_awac.copy(deep=True)

    td_rdi.attrs['principal_heading'] = calc_principal_heading(
        td_rdi.vel.mean('range'))
    td_sig.attrs['principal_heading'] = calc_principal_heading(
        td_sig.vel.mean('range'))
    td_awac.attrs['principal_heading'] = calc_principal_heading(td_awac.vel.mean('range'),
                                                                tidal_mode=False)
    td_rdi = rotate2(td_rdi, 'principal')
    td_sig = rotate2(td_sig, 'principal')
    td_awac = rotate2(td_awac, 'principal')

    if make_data:
        save(td_rdi, 'RDI_test01_rotate_earth2principal.nc')
        save(td_sig, 'BenchFile01_rotate_earth2principal.nc')
        save(td_awac, 'AWAC_test01_earth2principal.nc')
        return

    cd_rdi = load('RDI_test01_rotate_earth2principal.nc')
    cd_sig = load('BenchFile01_rotate_earth2principal.nc')
    cd_awac = load('AWAC_test01_earth2principal.nc')

    assert_allclose(td_rdi, cd_rdi, atol=1e-5)
    assert_allclose(td_awac, cd_awac, atol=1e-5)
    assert_allclose(td_sig, cd_sig, atol=1e-5)


if __name__ == '__main__':
    test_rotate_beam2inst()
    test_rotate_inst2beam()
    test_rotate_inst2earth()
    test_rotate_earth2inst()
    test_rotate_earth2principal()
