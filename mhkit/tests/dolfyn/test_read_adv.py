from . import base as tb
from mhkit.dolfyn.rotate.api import set_inst2head_rotmat
import mhkit.dolfyn.io.nortek as vector
from mhkit.dolfyn.io.api import read_example as read
import numpy as np
import os
import sys


load = tb.load_ncdata
save = tb.save_ncdata
assert_allclose = tb.assert_allclose

dat = load('vector_data01.nc')
dat_imu = load('vector_data_imu01.nc')
dat_imu_json = load('vector_data_imu01-json.nc')
dat_burst = load('burst_mode01.nc')


def test_save():
    save(dat, 'test_save.nc')
    tb.save_matlab(dat, 'test_save.mat')

    assert os.path.exists(tb.rfnm('test_save.nc'))
    assert os.path.exists(tb.rfnm('test_save.mat'))


def test_read(make_data=False):
    td = tb.drop_config(read('vector_data01.VEC', nens=100))
    tdm = tb.drop_config(read('vector_data_imu01.VEC', userdata=False,
                              nens=100))
    tdb = tb.drop_config(read('burst_mode01.VEC', nens=100))
    tdm2 = tb.drop_config(read('vector_data_imu01.VEC',
                               userdata=tb.exdt(
                                   'vector_data_imu01.userdata.json'),
                               nens=100))
    td_debug = tb.drop_config(vector.read_nortek(tb.exdt('vector_data_imu01.VEC'),
                              debug=True, do_checksum=True, nens=100))

    # These values are not correct for this data but I'm adding them for
    # test purposes only.
    tdm = set_inst2head_rotmat(tdm, np.eye(3))
    tdm.attrs['inst2head_vec'] = np.array([-1.0, 0.5, 0.2])

    if make_data:
        save(td, 'vector_data01.nc')
        save(tdm, 'vector_data_imu01.nc')
        save(tdb, 'burst_mode01.nc')
        save(tdm2, 'vector_data_imu01-json.nc')
        return

    assert_allclose(td, dat, atol=1e-6)
    assert_allclose(tdm, dat_imu, atol=1e-6)
    assert_allclose(tdb, dat_burst, atol=1e-6)
    assert_allclose(tdm2, dat_imu_json, atol=1e-6)
    assert_allclose(td_debug, tdm2, atol=1e-6)


if __name__ == '__main__':
    sys.stdout = open(os.devnull, 'w')  # block printing output
    test_save()
    test_read()
    sys.stdout = sys.__stdout__  # restart printing output
