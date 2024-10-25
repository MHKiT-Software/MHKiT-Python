from struct import unpack
import numpy as np
from datetime import datetime

from .. import time


def _bcd2char(cBCD):
    """Taken from the Nortek System Integrator Manual
    "Example Program" Chapter.
    """
    cBCD = min(cBCD, 153)
    c = cBCD & 15
    c += 10 * (cBCD >> 4)
    return c


def _bitshift8(val):
    return val >> 8


def _int2binarray(val, n):
    out = np.zeros(n, dtype="bool")
    for idx, n in enumerate(range(n)):
        out[idx] = val & (2**n)
    return out


def _crop_data(obj, range, n_lastdim):
    for nm, dat in obj.items():
        if isinstance(dat, np.ndarray) and (dat.shape[-1] == n_lastdim):
            obj[nm] = dat[..., range]


def _recatenate(obj):
    out = type(obj[0])()
    for ky in list(obj[0].keys()):
        if ky in ["__data_groups__", "_type"]:
            continue
        val0 = obj[0][ky]
        if isinstance(val0, np.ndarray) and val0.size > 1:
            out[ky] = np.concatenate([val[ky][..., None] for val in obj], axis=-1)
        else:
            out[ky] = np.array([val[ky] for val in obj])
    return out


def rd_time(strng):
    """Read the time from the first 6bytes of the input string."""
    min, sec, day, hour, year, month = unpack("BBBBBB", strng[:6])
    return time.date2epoch(
        datetime(
            time._fullyear(_bcd2char(year)),
            _bcd2char(month),
            _bcd2char(day),
            _bcd2char(hour),
            _bcd2char(min),
            _bcd2char(sec),
        )
    )[0]
