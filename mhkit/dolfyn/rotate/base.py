import numpy as np
import xarray as xr
from numpy.linalg import det, inv
from scipy.spatial.transform import Rotation as R
import warnings


def _make_model(ds):
    """
    The make and model of the instrument that collected the data
    in this data object.
    """
    return "{} {}".format(ds.attrs["inst_make"], ds.attrs["inst_model"]).lower()


def _check_rotmat_det(rotmat, thresh=1e-3):
    """
    Check that the absolute error of the determinant is small.

          abs(det(rotmat) - 1) < thresh

    Returns a boolean array.
    """

    if rotmat.ndim > 2:
        rotmat = np.transpose(rotmat)
    return np.abs(det(rotmat) - 1) < thresh


def _check_rotate_vars(ds, rotate_vars):
    if rotate_vars is None:
        if "rotate_vars" in ds.attrs:
            rotate_vars = ds.rotate_vars
        else:
            warnings.warn("    'rotate_vars' attribute not found." "Rotating `vel`.")
            rotate_vars = ["vel"]

    return rotate_vars


def _set_coords(ds, ref_frame, forced=False):
    """
    Checks the current reference frame and adjusts xarray coords/dims
    as necessary.
    Makes sure assigned dataarray coordinates match what DOLfYN is reading in.
    """

    make = _make_model(ds)

    XYZ = ["X", "Y", "Z"]
    ENU = ["E", "N", "U"]
    beam = ds.beam.values
    principal = ["streamwise", "x-stream", "vert"]

    # check make/model
    if "rdi" in make:
        inst = ["X", "Y", "Z", "err"]
        earth = ["E", "N", "U", "err"]
        princ = ["streamwise", "x-stream", "vert", "err"]

    elif "nortek" in make:
        if "signature" in make or "ad2cp" in make:
            inst = ["X", "Y", "Z1", "Z2"]
            earth = ["E", "N", "U1", "U2"]
            princ = ["streamwise", "x-stream", "vert1", "vert2"]

        else:  # AWAC or Vector
            inst = XYZ
            earth = ENU
            princ = principal

    orient = {
        "beam": beam,
        "inst": inst,
        "ship": inst,
        "earth": earth,
        "principal": princ,
    }
    orientIMU = {
        "beam": XYZ,
        "inst": XYZ,
        "ship": XYZ,
        "earth": ENU,
        "principal": principal,
    }

    if forced:
        ref_frame += "-forced"

    # Update 'dir' and 'dirIMU' dimensions
    attrs = ds["dir"].attrs
    attrs.update({"ref_frame": ref_frame})

    ds["dir"] = orient[ref_frame]
    ds["dir"].attrs = attrs
    if hasattr(ds, "dirIMU"):
        ds["dirIMU"] = orientIMU[ref_frame]
        ds["dirIMU"].attrs = attrs

    ds.attrs["coord_sys"] = ref_frame

    # These are essentially one extra line to scroll through
    tag = ["", "_echo", "_bt"]
    for tg in tag:
        if hasattr(ds, "coord_sys_axes" + tg):
            ds.attrs.pop("coord_sys_axes" + tg)

    return ds


def _beam2inst(dat, reverse=False, force=False):
    """
    Rotate velocities from beam to instrument coordinates.

    Parameters
    ----------
    dat : xarray.Dataset
      The ADCP dataset
    reverse : bool
      If True, this function performs the inverse rotation (inst->beam).
      Default = False
    force : bool, list
      When true do not check which coordinate system the data is in
      prior to performing this rotation. When forced-rotations are
      applied, the string '-forced!' is appended to the
      dat.props['coord_sys'] string. If force is a list, it contains
      a list of variables that should be rotated (rather than the
      default values in adpo.rotate_vars).
      Default = False
    """

    if not force:
        if not reverse and dat.coord_sys.lower() != "beam":
            raise ValueError("The input must be in beam coordinates.")
        if reverse and dat.coord_sys != "inst":
            raise ValueError("The input must be in inst coordinates.")

    rotmat = dat["beam2inst_orientmat"]

    if isinstance(force, (list, set, tuple)):
        # You can force a distinct set of variables to be rotated by
        # specifying it here.
        rotate_vars = force
    else:
        rotate_vars = [
            ky for ky in dat.rotate_vars if dat[ky].shape[0] == rotmat.shape[0]
        ]

    cs = "inst"
    if reverse:
        # Can't use transpose because rotation is not between
        # orthogonal coordinate systems
        rotmat = inv(rotmat)
        cs = "beam"
    for ky in rotate_vars:
        dat[ky].values = np.einsum("ij,j...->i...", rotmat, dat[ky].values)

    if force:
        dat = _set_coords(dat, cs, forced=True)
    else:
        dat = _set_coords(dat, cs)

    return dat


def euler2orient(heading, pitch, roll, units="degrees"):
    """
    Calculate the orientation matrix from DOLfYN-defined euler angles.

    This function is not likely to be called during data processing since it requires
    DOLfYN-defined euler angles. It is intended for testing DOLfYN.

    The matrices H, P, R are the transpose of the matrices for rotation about z, y, x
    as shown here https://en.wikipedia.org/wiki/Rotation_matrix. The transpose is used
    because in DOLfYN the orientation matrix is organized for
    rotation from EARTH --> INST, while the wiki's matrices are organized for
    rotation from INST --> EARTH.

    Parameters
    ----------
    heading : numpy.ndarray
      The heading angle.
    pitch : numpy.ndarray
      The pitch angle.
    roll : numpy.ndarray
      The roll angle.
    units : str
      Units in degrees or radians.  is 'degrees'

    Returns
    =======
    omat : numpy.ndarray (3x3xtime)
      The orientation matrix of the data. The returned orientation
      matrix obeys the following conventions:

       - a "ZYX" rotation order. That is, these variables are computed
         assuming that rotation from the earth -> instrument frame happens
         by rotating around the z-axis first (heading), then rotating
         around the y-axis (pitch), then rotating around the x-axis (roll).
         Note this requires matrix multiplication in the reverse order.

       - heading is defined as the direction the x-axis points, positive
         clockwise from North (this is *opposite* the right-hand-rule
         around the Z-axis), range 0-360 degrees.

       - pitch is positive when the x-axis pitches up (this is *opposite* the
         right-hand-rule around the Y-axis)

       - roll is positive according to the right-hand-rule around the
         instrument's x-axis
    """

    if units.lower() == "degrees":
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        heading = np.deg2rad(heading)
    elif units.lower() == "radians":
        pass
    else:
        raise Exception("Invalid units")

    # Converts the DOLfYN-defined heading to one that follows the right-hand-rule
    # reports heading as rotation of the y-axis positive counterclockwise from North
    heading = np.pi / 2 - heading

    # Converts the DOLfYN-defined pitch to one that follows the right-hand-rule.
    pitch = -pitch

    ch = np.cos(heading)
    sh = np.sin(heading)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    zero = np.zeros_like(sr)
    one = np.ones_like(sr)

    H = np.array(
        [
            [ch, sh, zero],
            [-sh, ch, zero],
            [zero, zero, one],
        ]
    )
    P = np.array(
        [
            [cp, zero, -sp],
            [zero, one, zero],
            [sp, zero, cp],
        ]
    )
    R = np.array(
        [
            [one, zero, zero],
            [zero, cr, sr],
            [zero, -sr, cr],
        ]
    )

    return np.einsum("ij...,jk...,kl...->il...", R, P, H)


def orient2euler(omat):
    """
    Calculate DOLfYN-defined euler angles from the orientation matrix.

    Parameters
    ----------
    omat : numpy.ndarray
      The orientation matrix

    Returns
    -------
    heading : numpy.ndarray
      The heading angle. Heading is defined as the direction the x-axis points,
      positive clockwise from North (this is *opposite* the right-hand-rule
      around the Z-axis), range 0-360 degrees.
    pitch : np.ndarray
      The pitch angle (degrees). Pitch is positive when the x-axis
      pitches up (this is *opposite* the right-hand-rule around the Y-axis).
    roll : np.ndarray
      The roll angle (degrees). Roll is positive according to the
      right-hand-rule around the instrument's x-axis.
    """

    if isinstance(omat, np.ndarray) and omat.shape[:2] == (3, 3):
        pass
    elif hasattr(omat, "orientmat"):
        omat = omat["orientmat"].values

    # Note: orientation matrix is earth->inst unless supplied by an external IMU
    hh = np.rad2deg(np.arctan2(omat[0, 0], omat[0, 1]))
    hh %= 360
    return (
        # heading
        hh,
        # pitch
        np.rad2deg(np.arcsin(omat[0, 2])),
        # roll
        np.rad2deg(np.arctan2(omat[1, 2], omat[2, 2])),
    )


def quaternion2orient(quaternions):
    """
    Calculate orientation from Nortek AHRS quaternions, where q = [W, X, Y, Z]
    instead of the standard q = [X, Y, Z, W] = [q1, q2, q3, q4]

    Parameters
    ----------
    quaternions : xarray.DataArray
      Quaternion dataArray from the raw dataset

    Returns
    -------
    orientmat : numpy.ndarray
      The earth2inst rotation maxtrix as calculated from the quaternions

    See Also
    --------
    scipy.spatial.transform.Rotation
    """

    omat = type(quaternions)(np.empty((3, 3, quaternions.time.size)))
    omat = omat.rename({"dim_0": "earth", "dim_1": "inst", "dim_2": "time"})

    for i in range(quaternions.time.size):
        r = R.from_quat(
            [
                quaternions.isel(q=1, time=i),
                quaternions.isel(q=2, time=i),
                quaternions.isel(q=3, time=i),
                quaternions.isel(q=0, time=i),
            ]
        )
        omat[..., i] = r.as_matrix()

    # quaternions in inst2earth reference frame, need to rotate to earth2inst
    omat.values = np.rollaxis(omat.values, 1)

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
    return omat.assign_coords({"earth": earth, "inst": inst, "time": quaternions.time})


def calc_tilt(pitch, roll):
    """
    Calculate "tilt", the vertical inclination, from pitch and roll.

    Parameters
    ----------
    roll : numpy.ndarray or xarray.DataArray
      Instrument roll in degrees
    pitch : numpy.ndarray or xarray.DataArray
      Instrument pitch in degrees

    Returns
    -------
    tilt : numpy.ndarray
      Vertical inclination of the instrument in degrees
    """

    if "xarray" in type(pitch).__module__:
        pitch = pitch.values
    if "xarray" in type(roll).__module__:
        roll = roll.values

    tilt = np.arctan(
        np.sqrt(np.tan(np.deg2rad(roll)) ** 2 + np.tan(np.deg2rad(pitch)) ** 2)
    )

    return np.rad2deg(tilt)
