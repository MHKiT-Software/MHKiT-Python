from mhkit.tests.dolfyn import base as tb
from mhkit.dolfyn.rotate.api import set_inst2head_rotmat
from mhkit.dolfyn.io.api import read_example as read
import numpy as np
import unittest

make_data = False
load = tb.load_netcdf
save = tb.save_netcdf
assert_allclose = tb.assert_allclose

dat = load("vector_data01")
dat_imu = load("vector_data_imu01")
dat_imu_json = load("vector_data_imu01-json")
dat_burst = load("vector_burst_mode01")


class io_adv_testcase(unittest.TestCase):
    def test_io_adv(self):
        nens = 100
        td = read("vector_data01.VEC", nens=nens)
        tdm = read("vector_data_imu01.VEC", userdata=False, nens=nens)
        tdb = read("vector_burst_mode01.VEC", nens=nens)
        tdm2 = read(
            "vector_data_imu01.VEC",
            userdata=tb.exdt("vector_data_imu01.userdata.json"),
            nens=nens,
        )

        # These values are not correct for this data but I'm adding them for
        # test purposes only.
        set_inst2head_rotmat(tdm, np.eye(3), inplace=True)
        tdm.attrs["inst2head_vec"] = [-1.0, 0.5, 0.2]

        if make_data:
            save(td, "vector_data01.nc")
            save(tdm, "vector_data_imu01.nc")
            save(tdb, "vector_burst_mode01.nc")
            save(tdm2, "vector_data_imu01-json.nc")
            return

        assert_allclose(td, dat, atol=1e-6)
        assert_allclose(tdm, dat_imu, atol=1e-6)
        assert_allclose(tdb, dat_burst, atol=1e-6)
        assert_allclose(tdm2, dat_imu_json, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
