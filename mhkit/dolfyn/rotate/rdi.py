import numpy as np
import xarray as xr
from .vector import _earth2principal
from .base import _beam2inst, _set_coords, _check_rotate_vars


def _inst2earth(adcpo, reverse=False, rotate_vars=None, force=False):
    """
    Rotate velocities from the instrument to earth coordinates.

    This function also rotates data from the 'ship' frame, into the
    earth frame when it is in the ship frame (and
    ``adcpo.use_pitchroll == 'yes'``). It does not support the
    'reverse' rotation back into the ship frame.

    Parameters
    ----------
    adcpo : xarray.Dataset
      The adcp dataset containing the data.
    reverse : bool
      If True, this function performs the inverse rotation (earth->inst).
      Default = False
    force : bool
      When true do not check which coordinate system the data is in
      prior to performing this rotation. Default = False

    Notes
    -----
    The rotation matrix is taken from the Teledyne RDI ADCP Coordinate
    Transformation manual January 2008
    """

    csin = adcpo.coord_sys.lower()
    cs_allowed = ["inst", "ship"]
    if reverse:
        cs_allowed = ["earth"]
    if not force and csin not in cs_allowed:
        raise ValueError(
            "Invalid rotation for data in {}-frame " "coordinate system.".format(csin)
        )

    if "orientmat" in adcpo:
        omat = adcpo["orientmat"]
    else:
        omat = _calc_orientmat(adcpo)

    rotate_vars = _check_rotate_vars(adcpo, rotate_vars)

    # rollaxis gives transpose of orientation matrix.
    # The 'rotation matrix' is the transpose of the 'orientation matrix'
    # NOTE: the double 'rollaxis' within this function, and here, has
    # minimal computational impact because np.rollaxis returns a
    # view (not a new array)
    rotmat = np.rollaxis(omat.data, 1)
    if reverse:
        cs_new = "inst"
        sumstr = "jik,j...k->i...k"
    else:
        cs_new = "earth"
        sumstr = "ijk,j...k->i...k"

    # Only operate on the first 3-components, b/c the 4th is err_vel
    for nm in rotate_vars:
        dat = adcpo[nm].values
        dat[:3] = np.einsum(sumstr, rotmat, dat[:3])
        adcpo[nm].values = dat.copy()

    adcpo = _set_coords(adcpo, cs_new)

    return adcpo


def _calc_beam_orientmat(theta=20, convex=True, degrees=True):
    """
    Calculate the rotation matrix from beam coordinates to
    instrument head coordinates for an RDI ADCP.

    Parameters
    ----------
    theta : int
      Angle of the heads (usually 20 or 30 degrees). Default = 20
    convex : bool
      Flag for convex or concave head configuration. Default = True
    degrees : bool
      Flag which specifies whether theta is in degrees or radians.
      Default = True
    """

    if degrees:
        theta = np.deg2rad(theta)
    if convex == 0 or convex == -1:
        c = -1
    else:
        c = 1
    a = 1 / (2.0 * np.sin(theta))
    b = 1 / (4.0 * np.cos(theta))
    d = a / (2.0**0.5)
    return np.array(
        [[c * a, -c * a, 0, 0], [0, 0, -c * a, c * a], [b, b, b, b], [d, d, -d, -d]]
    )


def _calc_orientmat(adcpo):
    """
    Calculate the orientation matrix using the raw
    heading, pitch, roll values from the RDI binary file.

    Parameters
    ----------
    adcpo : xarray.Dataset
      The adcp dataset containing the data.

    ## RDI-ADCP-MANUAL (Jan 08, section 5.6 page 18)
    The internal tilt sensors do not measure exactly the same
    pitch as a set of gimbals would (the roll is the same). Only in
    the case of the internal pitch sensor being selected (EZxxx1xxx),
    the measured pitch is modified using the following algorithm.

        P = arctan[tan(Tilt1)*cos(Tilt2)]    (Equation 18)

    Where: Tilt1 is the measured pitch from the internal sensor, and
    Tilt2 is the measured roll from the internal sensor The raw pitch
    (Tilt 1) is recorded in the variable leader. P is set to 0 if the
    "use tilt" bit of the EX command is not set."""

    r = np.deg2rad(adcpo["roll"].values)
    p = np.arctan(np.tan(np.deg2rad(adcpo["pitch"].values)) * np.cos(r))
    h = np.deg2rad(adcpo["heading"].values)

    if "rdi" in adcpo.inst_make.lower():
        if adcpo.orientation == "up":
            """
            ## RDI-ADCP-MANUAL (Jan 08, section 5.6 page 18)
            Since the roll describes the ship axes rather than the
            instrument axes, in the case of upward-looking
            orientation, 180 degrees must be added to the measured
            roll before it is used to calculate M. This is equivalent
            to negating the first and third columns of M. R is set
            to 0 if the "use tilt" bit of the EX command is not set.
            """
            r += np.pi
        if adcpo.coord_sys == "ship" and adcpo.use_pitchroll == "yes":
            r[:] = 0
            p[:] = 0

    ch = np.cos(h)
    sh = np.sin(h)
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    rotmat = np.empty((3, 3, len(r)))
    rotmat[0, 0, :] = ch * cr + sh * sp * sr
    rotmat[0, 1, :] = sh * cp
    rotmat[0, 2, :] = ch * sr - sh * sp * cr
    rotmat[1, 0, :] = -sh * cr + ch * sp * sr
    rotmat[1, 1, :] = ch * cp
    rotmat[1, 2, :] = -sh * sr - ch * sp * cr
    rotmat[2, 0, :] = -cp * sr
    rotmat[2, 1, :] = sp
    rotmat[2, 2, :] = cp * cr

    # The 'orientation matrix' is the transpose of the 'rotation matrix'.
    omat = np.rollaxis(rotmat, 1)

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
        coords={"earth": earth, "inst": inst, "time": adcpo.time},
        dims=["earth", "inst", "time"],
        attrs={"units": "1", "long_name": "Orientation Matrix"},
    )
