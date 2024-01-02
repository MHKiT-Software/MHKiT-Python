import numpy as np
import xarray as xr
import warnings
from . import base as rotb


def _beam2inst(dat, reverse=False, force=False):
    # Order of rotations matters
    # beam->head(ADV instrument head)->inst(ADV battery case|imu)
    if reverse:
        # First rotate velocities from ADV inst frame back to head frame
        dat = _rotate_inst2head(dat, reverse=reverse)
        # Now rotate from the head frame to the beam frame
        dat = rotb._beam2inst(dat, reverse=reverse, force=force)

    # inst(ADV battery case|imu)->head(ADV instrument head)->beam
    else:
        # First rotate velocities from beam to ADV head frame
        dat = rotb._beam2inst(dat, force=force)
        # Then rotate from ADV head frame to ADV inst frame
        dat = _rotate_inst2head(dat)

    # Set the docstring to match the default rotation func
    _beam2inst.__doc_ = rotb._beam2inst.__doc__

    return dat


def _rotate_inst2head(advo, reverse=False):
    """
    Rotates the velocity vector from the instrument frame to the ADV probe (head) frame or
    vice versa.

    This function uses the rotation matrix 'inst2head_rotmat' to rotate the velocity vector 'vel'
    from the instrument frame to the head frame ('inst->head') or from the head frame to the
    instrument frame ('head->inst').

    Parameters
    ----------
    advo: dict
      A dictionary-like object that includes the rotation matrix 'inst2head_rotmat'
      and the velocity vector 'vel' to be rotated.

    reverse: bool, optional
      A boolean value indicating the direction of the rotation.
      If False (default), the function rotates 'vel' from the instrument frame to the head frame.
      If True, the function rotates 'vel' from the head frame to the instrument frame.

    Returns
    -------
    advo: dict
      The input dictionary-like object with the rotated velocity vector.
      If 'inst2head_rotmat' doesn't exist in 'advo', the function returns the input 'advo' unmodified.
    """

    if not _check_inst2head_rotmat(advo):
        # This object doesn't have a head2inst_rotmat, so we do nothing.
        return advo
    if reverse:  # head->inst
        advo["vel"].values = np.dot(advo["inst2head_rotmat"].T, advo["vel"])
    else:  # inst->head
        advo["vel"].values = np.dot(advo["inst2head_rotmat"], advo["vel"])

    return advo


def _check_inst2head_rotmat(advo):
    """
    Verify that the 'inst2head_rotmat' exists, was set using 'set_inst2head_rotmat', and
    the determinant of the rotation matrix is unity.

    Parameters
    ----------
    advo: dict
      A dictionary-like object that should include the rotation matrix 'inst2head_rotmat'.

    Returns
    -------
    bool
      Returns True if 'inst2head_rotmat' exists, was set correctly, and is valid (False if not).
    """

    if advo.get("inst2head_rotmat", None) is None:
        # This is the default value, and we do nothing.
        return False
    if not advo.inst2head_rotmat_was_set:
        raise Exception(
            "The inst2head rotation matrix exists in props, "
            "but it was not set using `set_inst2head_rotmat."
        )
    if not rotb._check_rotmat_det(advo.inst2head_rotmat.values):
        raise ValueError("Invalid inst2head_rotmat (determinant != 1).")
    return True


def _inst2earth(advo, reverse=False, rotate_vars=None, force=False):
    """
    Rotate data in an ADV object to the earth from the instrument
    frame (or vice-versa).

    Parameters
    ----------
    advo : xarray.Dataset
      The adv dataset containing the data.
    reverse : bool
      If True, this function performs the inverse rotation (earth->inst).
      Default = False
    rotate_vars : iterable
      The list of variables to rotate. By default this is taken from
      advo.attrs['rotate_vars'].
    force : bool
      Do not check which frame the data is in prior to performing
      this rotation. Default = False
    """

    if reverse:  # earth->inst
        # The transpose of the rotation matrix gives the inverse
        # rotation, so we simply reverse the order of the einsum:
        sumstr = "jik,j...k->i...k"
        cs_now = "earth"
        cs_new = "inst"
    else:  # inst->earth
        sumstr = "ijk,j...k->i...k"
        cs_now = "inst"
        cs_new = "earth"

    rotate_vars = rotb._check_rotate_vars(advo, rotate_vars)

    cs = advo.coord_sys.lower()
    if not force:
        if cs == cs_new:
            print("Data is already in the '%s' coordinate system" % cs_new)
            return
        elif cs != cs_now:
            raise ValueError(
                "Data must be in the '%s' frame when using this function" % cs_now
            )

    if hasattr(advo, "orientmat"):
        omat = advo["orientmat"]
    else:
        if "vector" in advo.inst_model.lower():
            orientation_down = advo["orientation_down"]

        omat = _calc_omat(
            advo["time"], advo["heading"], advo["pitch"], advo["roll"], orientation_down
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

    for nm in rotate_vars:
        n = advo[nm].shape[0]
        if n != 3:
            raise Exception(
                "The entry {} is not a vector, it cannot " "be rotated.".format(nm)
            )
        advo[nm].values = np.einsum(sumstr, rmat, advo[nm])

    advo = rotb._set_coords(advo, cs_new)

    return advo


def _earth2principal(advo, reverse=False, rotate_vars=None):
    """
    Rotate data in an ADV dataset to/from principal axes. Principal
    heading must be within the dataset.

    All data in the advo.attrs['rotate_vars'] list will be
    rotated by the principal heading, and also if the data objet has an
    orientation matrix (orientmat) it will be rotated so that it
    represents the orientation of the ADV in the principal
    (reverse:earth) frame.

    Parameters
    ----------
    advo : xarray.Dataset
      The adv dataset containing the data.
    reverse : bool
      If True, this function performs the inverse rotation
      (principal->earth). Default = False
    """

    # This is in degrees CW from North
    ang = np.deg2rad(90 - advo.principal_heading)
    # convert this to radians CCW from east (which is expected by
    # the rest of the function)

    if reverse:
        cs_now = "principal"
        cs_new = "earth"
    else:
        ang *= -1
        cs_now = "earth"
        cs_new = "principal"

    rotate_vars = rotb._check_rotate_vars(advo, rotate_vars)

    cs = advo.coord_sys.lower()
    if cs == cs_new:
        print("Data is already in the %s coordinate system" % cs_new)
        return
    elif cs != cs_now:
        raise ValueError(
            "Data must be in the {} frame " "to use this function".format(cs_now)
        )

    # Calculate the rotation matrix:
    cp, sp = np.cos(ang), np.sin(ang)
    rotmat = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]], dtype=np.float32)

    # Perform the rotation:
    for nm in rotate_vars:
        dat = advo[nm].values
        dat[:2] = np.einsum("ij,j...->i...", rotmat[:2, :2], dat[:2])
        advo[nm].values = dat.copy()

    # Finalize the output.
    advo = rotb._set_coords(advo, cs_new)

    return advo


def _calc_omat(time, hh, pp, rr, orientation_down=None):
    """
    Calculates the dolfyn-defined orientation matrix from Euler angles.

    Parameters
    ----------
    time: array-like
      Time points corresponding to the Euler angles.

    hh: array-like
      Heading Euler angle in degrees.

    pp: array-like
      Pitch Euler angle in degrees.

    rr: array-like
      Roll Euler angle in degrees.

    orientation_down: array-like or bool, optional
      Set to true if instrument is facing downwards

    Returns
    -------
    omat: array-like
      The calculated orientation matrix.
    """

    rr = rr.data.copy()
    pp = pp.data.copy()
    hh = hh.data.copy()
    if np.isnan(rr[-1]) and np.isnan(pp[-1]) and np.isnan(hh[-1]):
        # The end of the data may not have valid orientations
        lastgd = np.nonzero(~np.isnan(rr + pp + hh))[0][-1]
        rr[lastgd:] = rr[lastgd]
        pp[lastgd:] = pp[lastgd]
        hh[lastgd:] = hh[lastgd]
    if orientation_down is not None:
        # For Nortek Vector ADVs: 'down' configuration means the head was
        # pointing 'up', where the 'up' orientation corresponds to the
        # communication cable being up.  Check the Nortek coordinate
        # transform matlab script for more info.
        rr[orientation_down.astype(bool)] += 180

    return _euler2orient(time, hh, pp, rr)


def _euler2orient(time, heading, pitch, roll, units="degrees"):
    # For Nortek data only.
    # The heading, pitch, roll used here are from the Nortek binary files.

    # Heading input is clockwise from North
    # Returns a rotation matrix that rotates earth (ENU) -> inst.
    # This is based on the Nortek `Transforms.m` file, available in
    # the refs folder.
    if units.lower() == "degrees":
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        heading = np.deg2rad(heading)

    # The definition of heading below is consistent with the right-hand-rule;
    # heading is the angle positive counterclockwise from North of the y-axis.

    # This also involved swapping the sign on sh in the def of omat
    # below from the values provided in the Nortek Matlab script.
    heading = np.pi / 2 - heading

    ch = np.cos(heading)
    sh = np.sin(heading)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # Note that I've transposed these values (from what is defined in
    # Nortek matlab script), so that the omat is earth->inst
    omat = np.empty((3, 3, len(sh)), dtype=np.float32)
    omat[0, 0, :] = ch * cp
    omat[1, 0, :] = -ch * sp * sr - sh * cr
    omat[2, 0, :] = -ch * cr * sp + sh * sr
    omat[0, 1, :] = sh * cp
    omat[1, 1, :] = -sh * sp * sr + ch * cr
    omat[2, 1, :] = -sh * cr * sp - ch * sr
    omat[0, 2, :] = sp
    omat[1, 2, :] = sr * cp
    omat[2, 2, :] = cp * cr

    earth = xr.DataArray(
        ["E", "N", "U"],
        dims=["earth"],
        name="earth",
        attrs={
            "units": "1",
            "long_name": "Earth Reference Frame",
            "coverage_content_type": "coordinate",
        },
    )
    inst = xr.DataArray(
        ["X", "Y", "Z"],
        dims=["inst"],
        name="inst",
        attrs={
            "units": "1",
            "long_name": "Instrument Reference Frame",
            "coverage_content_type": "coordinate",
        },
    )
    return xr.DataArray(
        omat,
        coords={"earth": earth, "inst": inst, "time": time},
        dims=["earth", "inst", "time"],
        attrs={"units": "1", "long_name": "Orientation Matrix"},
    )
