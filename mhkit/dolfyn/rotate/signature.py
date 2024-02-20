from .vector import _earth2principal, _euler2orient
from .base import _beam2inst
from . import base as rotb
import numpy as np
import warnings
from numpy.linalg import inv


def _inst2earth(adcpo, reverse=False, rotate_vars=None, force=False):
    """
    Rotate data in an ADCP object to the earth from the instrument
    frame (or vice-versa).

    Parameters
    ----------
    adcpo : xarray.Dataset
      The adcp dataset containing the data.
    reverse : bool
      If True, this function performs the inverse rotation (earth->inst).
      Default = False
    rotate_vars : iterable
      The list of variables to rotate. By default this is taken from
      adcpo.rotate_vars.
    force : bool
      Do not check which frame the data is in prior to performing
      this rotation. Default = False
    """

    if reverse:
        # The transpose of the rotation matrix gives the inverse
        # rotation, so we simply reverse the order of the einsum:
        sumstr = "jik,j...k->i...k"
        cs_now = "earth"
        cs_new = "inst"
    else:
        sumstr = "ijk,j...k->i...k"
        cs_now = "inst"
        cs_new = "earth"

    # if ADCP is upside down
    if adcpo.orientation == "down":
        down = True
    else:  # orientation = 'up' or 'AHRS'
        down = False

    rotate_vars = rotb._check_rotate_vars(adcpo, rotate_vars)

    cs = adcpo.coord_sys.lower()
    if not force:
        if cs == cs_new:
            print("Data is already in the '%s' coordinate system" % cs_new)
            return
        elif cs != cs_now:
            raise ValueError(
                "Data must be in the '%s' frame when using this function" % cs_now
            )

    if "orientmat" in adcpo:
        omat = adcpo["orientmat"]
    else:
        omat = _euler2orient(
            adcpo["time"],
            adcpo["heading"].values,
            adcpo["pitch"].values,
            adcpo["roll"].values,
        )

    # Take the transpose of the orientation to get the inst->earth rotation
    # matrix.
    rmat = np.rollaxis(omat.data, 1)

    _dcheck = rotb._check_rotmat_det(rmat)
    if not _dcheck.all():
        warnings.warn(
            "Invalid orientation matrix (determinant != 1) at indices: {}. "
            "If rotated, data at these indices will be erroneous.".format(
                np.nonzero(~_dcheck)[0]
            ),
            UserWarning,
        )

    # The dictionary of rotation matrices for different sized arrays.
    rmd = {
        3: rmat,
    }

    # The 4-row rotation matrix assume that rows 0,1 are u,v,
    # and 2,3 are independent estimates of w.
    tmp = rmd[4] = np.zeros((4, 4, rmat.shape[-1]), dtype=np.float64)
    tmp[:3, :3] = rmat
    # Copy row 2 to 3
    tmp[3, :2] = rmat[2, :2]
    tmp[3, 3] = rmat[2, 2]
    # Extend rows 0,1
    tmp[0, 2:] = rmat[0, 2] / 2
    tmp[1, 2:] = rmat[1, 2] / 2

    if reverse:
        # 3-element inverse handled by sumstr definition (transpose)
        rmd[4] = np.moveaxis(inv(np.moveaxis(rmd[4], -1, 0)), 0, -1)

    for nm in rotate_vars:
        dat = adcpo[nm].values
        n = dat.shape[0]
        # Nortek documents sign change for upside-down instruments
        if down:
            # This is equivalent to adding 180 degrees to roll axis in _calc_omat()
            sign = np.array([1, -1, -1, -1], ndmin=dat.ndim).T
            signIMU = np.array([1, -1, -1], ndmin=dat.ndim).T
            if not reverse:
                if n == 3:
                    dat = np.einsum(sumstr, rmd[3], signIMU * dat)
                elif n == 4:
                    dat = np.einsum("ijk,j...k->i...k", rmd[4], sign * dat)
                else:
                    raise Exception(
                        "The entry {} is not a vector, it cannot"
                        "be rotated.".format(nm)
                    )

            elif reverse:
                if n == 3:
                    dat = signIMU * np.einsum(sumstr, rmd[3], dat)
                elif n == 4:
                    dat = sign * np.einsum("ijk,j...k->i...k", rmd[4], dat)
                else:
                    raise Exception(
                        "The entry {} is not a vector, it cannot"
                        "be rotated.".format(nm)
                    )

        else:  # 'up' and AHRS
            if n == 3:
                dat = np.einsum(sumstr, rmd[3], dat)
            elif n == 4:
                dat = np.einsum("ijk,j...k->i...k", rmd[4], dat)
            else:
                raise Exception(
                    "The entry {} is not a vector, it cannot" "be rotated.".format(nm)
                )
        adcpo[nm].values = dat.copy()

    adcpo = rotb._set_coords(adcpo, cs_new)

    return adcpo
