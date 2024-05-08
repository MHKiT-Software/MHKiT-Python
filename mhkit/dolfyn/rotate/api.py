from . import vector as r_vec
from . import awac as r_awac
from . import signature as r_sig
from . import rdi as r_rdi
from .base import _make_model
import numpy as np
import xarray as xr
import warnings


# The 'rotation chain'
rc = ["beam", "inst", "earth", "principal"]

rot_module_dict = {
    # Nortek instruments
    "vector": r_vec,
    "awac": r_awac,
    "signature": r_sig,
    "ad2cp": r_sig,
    # TRDI instruments
    "rdi": r_rdi,
}


def rotate2(ds, out_frame="earth", inplace=True):
    """
    Rotate a dataset to a new coordinate system.

    Parameters
    ----------
    ds : xr.Dataset
      The dolfyn dataset (ADV or ADCP) to rotate.
    out_frame : string {'beam', 'inst', 'earth', 'principal'}
      The coordinate system to rotate the data into.
    inplace : bool
      When True ``ds`` is modified. When False a copy is returned.
      Default = True

    Returns
    -------
    ds : xarray.Dataset or None
      Returns a new rotated dataset **when ``inplace=False``**, otherwise
      returns None.

    Notes
    -----
    - This function rotates all variables in ``ds.attrs['rotate_vars']``.

    - In order to rotate to the 'principal' frame, a value should exist for
      ``ds.attrs['principal_heading']``. The function
      :func:`calc_principal_heading <dolfyn.calc_principal_heading>`
      is recommended for this purpose, e.g.:

          ds.attrs['principal_heading'] = dolfyn.calc_principal_heading(ds['vel'].mean(range))

      where here we are using the depth-averaged velocity to calculate
      the principal direction.
    """

    # Create and return deep copy if not writing "in place"
    if not inplace:
        ds = ds.copy(deep=True)

    csin = ds.coord_sys.lower()
    if csin == "ship":
        csin = "inst"

    # Returns True/False if head2inst_rotmat has been set/not-set.
    r_vec._check_inst2head_rotmat(ds)

    if out_frame == "principal" and csin != "earth":
        warnings.warn(
            "You are attempting to rotate into the 'principal' "
            "coordinate system, but the dataset is in the {} "
            "coordinate system. Be sure that 'principal_heading' is "
            "defined based on the earth coordinate system.".format(csin)
        )

    rmod = None
    for ky in rot_module_dict:
        if ky in _make_model(ds):
            rmod = rot_module_dict[ky]
            break
    if rmod is None:
        raise ValueError(
            "Rotations are not defined for " "instrument '{}'.".format(_make_model(ds))
        )

    # Get the 'indices' of the rotation chain
    try:
        iframe_in = rc.index(csin)
    except ValueError:
        raise Exception(
            "The coordinate system of the input "
            "dataset, '{}', is invalid.".format(ds.coord_sys)
        )
    try:
        iframe_out = rc.index(out_frame.lower())
    except ValueError:
        raise Exception(
            "The specified output coordinate system "
            "is invalid, please select one of: 'beam', 'inst', "
            "'earth', 'principal'."
        )

    if iframe_out == iframe_in:
        print("Data is already in the {} coordinate system".format(out_frame))

    if iframe_out > iframe_in:
        reverse = False
    else:
        reverse = True

    while ds.coord_sys.lower() != out_frame.lower():
        csin = ds.coord_sys
        if csin == "ship":
            csin = "inst"
        inow = rc.index(csin)
        if reverse:
            func = getattr(rmod, "_" + rc[inow - 1] + "2" + rc[inow])
        else:
            func = getattr(rmod, "_" + rc[inow] + "2" + rc[inow + 1])
        ds = func(ds, reverse=reverse)

    if not inplace:
        return ds


def calc_principal_heading(vel, tidal_mode=True):
    """
    Compute the principal angle of the horizontal velocity.

    Parameters
    ----------
    vel : np.ndarray (2,...,Nt), or (3,...,Nt)
      The 2D or 3D velocity array (3rd-dim is ignored in this calculation)
    tidal_mode : bool
      If true, range is set from 0 to +/-180 degrees. If false, range is 0 to
      360 degrees. Default = True

    Returns
    -------
    p_heading : float or ndarray
      The principal heading in degrees clockwise from North.

    Notes
    -----
    When tidal_mode=True, this tool calculates the heading that is
    aligned with the bidirectional flow. It does so following these
    steps:
      1. rotates vectors with negative velocity by 180 degrees
      2. then doubles those angles to make a complete circle again
      3. computes a mean direction from this, and halves that angle
         (to undo the doubled-angles in step 2)
      4. The returned angle is forced to be between 0 and 180. So, you
         may need to add 180 to this if you want your positive
         direction to be in the western-half of the plane.

    Otherwise, this function simply computes the average direction
    using a vector method.
    """

    dt = vel[0] + vel[1] * 1j
    if tidal_mode:
        # Flip all vectors that are below the x-axis
        dt[dt.imag <= 0] *= -1
        # Now double the angle, so that angles near pi and 0 get averaged
        # together correctly:
        dt *= np.exp(1j * np.angle(dt))
        dt = np.ma.masked_invalid(dt)
        # Divide the angle by 2 to remove the doubling done on the previous
        # line.
        pang = np.angle(np.nanmean(dt, -1, dtype=np.complex128)) / 2
    else:
        pang = np.angle(np.nanmean(dt, -1))

    return np.round((90 - np.rad2deg(pang)), decimals=4)


def set_declination(ds, declin, inplace=True):
    """
    Set the magnetic declination

    Parameters
    ----------
    ds : xarray.Dataset or :class:`dolfyn.velocity.Velocity`
      The input dataset or velocity class
    declination : float
      The value of the magnetic declination in degrees (positive
      values specify that Magnetic North is clockwise from True North)
    inplace : bool
      When True ``ds`` is modified. When False a copy is returned.
      Default = True

    Returns
    -------
    ds : xarray.Dataset or None
      Returns a new dataset with declination set **when
      ``inplace=False``**, otherwise returns None.

    Notes
    -----
    This function modifies the data object in the following ways:

    - If the dataset is in the *earth* reference frame at the time of
      setting declination, it will be rotated into the "*True-East*,
      *True-North*, Up" (hereafter, ETU) coordinate system

    - ``dat['orientmat']`` is modified to be an ETU to
      instrument (XYZ) rotation matrix (rather than the magnetic-ENU to
      XYZ rotation matrix). Therefore, all rotations to/from the 'earth'
      frame will now be to/from this ETU coordinate system.

    - The value of the specified declination will be stored in
      ``dat.attrs['declination']``

    - ``dat['heading']`` is adjusted for declination
      (i.e., it is relative to True North).

    - If ``dat.attrs['principal_heading']`` is set, it is
      adjusted to account for the orientation of the new 'True'
      earth coordinate system (i.e., calling set_declination on a
      data object in the principal coordinate system, then calling
      dat.rotate2('earth') will yield a data object in the new
      'True' earth coordinate system)
    """

    # Create and return deep copy if not writing "in place"
    if not inplace:
        ds = ds.copy(deep=True)

    if "declination" in ds.attrs:
        angle = declin - ds.attrs.pop("declination")
    else:
        angle = declin
    cd = np.cos(-np.deg2rad(angle))
    sd = np.sin(-np.deg2rad(angle))

    # The ordering is funny here because orientmat is the
    # transpose of the inst->earth rotation matrix:
    Rdec = np.array([[cd, -sd, 0], [sd, cd, 0], [0, 0, 1]])

    if ds.coord_sys == "earth":
        rotate2earth = True
        rotate2(ds, "inst", inplace=True)
    else:
        rotate2earth = False

    ds["orientmat"].values = np.einsum(
        "kj...,ij->ki...",
        ds["orientmat"].values,
        Rdec,
    )
    if "heading" in ds:
        ds["heading"] += angle
    if rotate2earth:
        rotate2(ds, "earth", inplace=True)
    if "principal_heading" in ds.attrs:
        ds.attrs["principal_heading"] += angle

    ds.attrs["declination"] = declin
    ds.attrs["declination_in_orientmat"] = 1  # logical

    if not inplace:
        return ds


def set_inst2head_rotmat(ds, rotmat, inplace=True):
    """
    Set the instrument to head rotation matrix for the Nortek ADV if it
    hasn't already been set through a '.userdata.json' file.

    Parameters
    ----------
    ds : xarray.Dataset
      The data set to assign inst2head_rotmat
    rotmat : float
      3x3 rotation matrix
    inplace : bool
      When True ``ds`` is modified. When False a copy is returned.
      Default = True

    Returns
    -------
    ds : xarray.Dataset or None
      Returns a new dataset with inst2head_rotmat set **when
      ``inplace=False``**, otherwise returns None.

    Notes
    -----
    If the data object is in earth or principal coords, it is first
    rotated to 'inst' before assigning inst2head_rotmat, it is then
    rotated back to the coordinate system in which it was input. This
    way the inst2head_rotmat gets applied correctly (in inst
    coordinate system).
    """

    # Create and return deep copy if not writing "in place"
    if not inplace:
        ds = ds.copy(deep=True)

    if not ds.inst_model.lower() == "vector":
        raise Exception(
            "Setting 'inst2head_rotmat' is only supported " "for Nortek Vector ADVs."
        )
    if ds.get("inst2head_rotmat", None) is not None:
        raise Exception(
            "You are setting 'inst2head_rotmat' after it has already "
            "been set. You can only set it once."
        )

    csin = ds.coord_sys
    if csin not in ["inst", "beam"]:
        rotate2(ds, "inst", inplace=True)

    ds["inst2head_rotmat"] = xr.DataArray(
        np.array(rotmat), dims=["x1", "x2"], coords={"x1": [1, 2, 3], "x2": [1, 2, 3]}
    )

    ds.attrs["inst2head_rotmat_was_set"] = 1  # logical
    # Note that there is no validation that the user doesn't
    # change `ds.attrs['inst2head_rotmat']` after calling this
    # function.

    if not csin == "beam":  # csin not 'beam', then we're in inst
        ds = r_vec._rotate_inst2head(ds)
    if csin not in ["inst", "beam"]:
        rotate2(ds, csin, inplace=True)

    if not inplace:
        return ds
