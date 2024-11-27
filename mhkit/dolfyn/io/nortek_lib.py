from struct import unpack
import numpy as np
from datetime import datetime

from .. import time


def _bcd2char(cBCD):
    """
    Converts a Binary-Coded Decimal (BCD) value to a character integer.

    This method is based on the Nortek System Integrator Manual's example
    program and ensures that values do not exceed 153.

    Parameters
    ----------
    cBCD : int
        BCD-encoded integer.

    Returns
    -------
    int
        The decoded integer from BCD format.
    """
    cBCD = min(cBCD, 153)
    c = cBCD & 15
    c += 10 * (cBCD >> 4)
    return c


def _bitshift8(val):
    """
    Performs an 8-bit right bit shift on an integer value.

    Parameters
    ----------
    val : int
        The value to shift.

    Returns
    -------
    int
        The integer result after shifting right by 8 bits.
    """
    return val >> 8


def _int2binarray(val, n):
    """
    Converts an integer to a binary array of length `n`.

    Parameters
    ----------
    val : int
        Integer to convert to binary.
    n : int
        Length of the output binary array.

    Returns
    -------
    np.ndarray
        Binary array of boolean values representing the integer.
    """
    out = np.zeros(n, dtype="bool")
    for idx, n in enumerate(range(n)):
        out[idx] = val & (2**n)
    return out


def _crop_data(obj, range, n_lastdim):
    """
    Crops data in a dictionary of arrays based on the specified range.

    Parameters
    ----------
    obj : dict
        Dictionary containing data arrays.
    range : slice
        The range to crop the last dimension of each array.
    n_lastdim : int
        The size of the last dimension to check for cropping eligibility.

    Notes
    -----
    Modifies `obj` in place by cropping the last dimension of arrays.
    """
    for nm, dat in obj.items():
        if isinstance(dat, np.ndarray) and (dat.shape[-1] == n_lastdim):
            obj[nm] = dat[..., range]


def _recatenate(obj):
    """
    Concatenates data from a list of dictionaries along the last axis.

    Parameters
    ----------
    obj : list of dict
        List of dictionaries to concatenate.

    Returns
    -------
    dict
        A dictionary with concatenated values across the list.
    """
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
    """
    Reads the time from the first 6 bytes of a binary string.

    Parameters
    ----------
    strng : bytes
        A byte string where the first 6 bytes encode the time.

    Returns
    -------
    float
        The epoch time extracted from the input byte string.

    Notes
    -----
    Uses BCD conversion to extract year, month, day, hour, minute, and
    second from the byte string, then converts to an epoch time.
    """

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
