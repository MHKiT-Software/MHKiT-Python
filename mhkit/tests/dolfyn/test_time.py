from . import test_read_adv as trv
from . import test_read_adp as trp
import mhkit.dolfyn.time as time
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from datetime import datetime
import unittest


class time_testcase(unittest.TestCase):
    def test_time_conversion(self):
        td = trv.dat_imu.copy(deep=True)
        dat_sig = trp.dat_sig_i.copy(deep=True)

        dt = time.dt642date(td.time)
        dt1 = time.dt642date(td.time[0])
        dt_off = time.epoch2date(time.dt642epoch(td.time), offset_hr=-7)
        t_str = time.epoch2date(time.dt642epoch(td.time), to_str=True)

        assert_equal(dt[0], datetime(2012, 6, 12, 12, 0, 2, 687283))
        assert_equal(dt1, [datetime(2012, 6, 12, 12, 0, 2, 687283)])
        assert_equal(dt_off[0], datetime(2012, 6, 12, 5, 0, 2, 687283))
        assert_equal(t_str[0], "2012-06-12 12:00:02.687283")

        # Validated based on data in ad2cp.index file
        assert_equal(
            time.dt642date(dat_sig.time[0])[0], datetime(2017, 7, 24, 17, 0, 0, 63500)
        )
        # This should always be true
        assert_equal(time.epoch2date([0])[0], datetime(1970, 1, 1, 0, 0))

    def test_datetime(self):
        td = trv.dat_imu.copy(deep=True)

        dt = time.dt642date(td.time)
        epoch = np.array(time.date2epoch(dt))

        assert_allclose(time.dt642epoch(td.time.values), epoch, atol=1e-7)

    def test_datenum(self):
        td = trv.dat_imu.copy(deep=True)

        dt = time.dt642date(td.time)
        dn = time.date2matlab(dt)
        dt2 = time.matlab2date(dn)
        epoch = np.array(time.date2epoch(dt2))

        assert_allclose(time.dt642epoch(td.time.values), epoch, atol=1e-6)
        assert_equal(dn[0], 735032.5000311028)


if __name__ == "__main__":
    unittest.main()
