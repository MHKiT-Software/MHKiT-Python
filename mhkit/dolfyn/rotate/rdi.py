import numpy as np
from .vector import earth2principal, inst2earth as nortek_inst2earth
from .base import beam2inst, _set_coords


def inst2earth(adcpo, reverse=False,
               fixed_orientation=False, force=False):
    """
    Rotate velocities from the instrument to earth coordinates.

    This function also rotates data from the 'ship' frame, into the
    earth frame when it is in the ship frame (and
    ``adcpo.use_pitchroll == 'yes'``). It does not support the
    'reverse' rotation back into the ship frame.

    Parameters
    ----------
    adpo : The ADP object containing the data.

    reverse : bool (default: False)
           If True, this function performs the inverse rotation
           (earth->inst).
    fixed_orientation : bool (default: False)
        When true, take the average orientation and apply it over the
        whole record.
    force : bool (default: False)
        When true do not check which coordinate system the data is in
        prior to performing this rotation.

    Notes
    -----
    The rotation matrix is taken from the Teledyne RDI ADCP Coordinate
    Transformation manual January 2008
    
    """
    if adcpo.inst_make.lower() == 'nortek':
        # Handle nortek rotations with the nortek (adv) rotate fn.
        return nortek_inst2earth(adcpo, reverse=reverse, force=force)

    csin = adcpo.coord_sys.lower()
    cs_allowed = ['inst', 'ship']
    if reverse:
        cs_allowed = ['earth']
    if not force and csin not in cs_allowed:
        raise ValueError("Invalid rotation for data in {}-frame "
                         "coordinate system.".format(csin))
        
    if 'orientmat' in adcpo:
        rmat = adcpo['orientmat'].values
    else:
        rmat = calc_orientmat(adcpo)

    # rollaxis gives transpose of orientation matrix.
    # The 'rotation matrix' is the transpose of the 'orientation matrix'
    # NOTE the double 'rollaxis' within this function, and here, has
    # minimal computational impact because np.rollaxis returns a
    # view (not a new array)
    rotmat = np.rollaxis(rmat, 1)
    if reverse:
        cs_new = 'inst'
        sumstr = 'jik,j...k->i...k'
    else:
        cs_new = 'earth'
        sumstr = 'ijk,j...k->i...k'

    # Only operate on the first 3-components, b/c the 4th is err_vel
    for nm in adcpo.rotate_vars:
        dat = adcpo[nm].values
        dat[:3] = np.einsum(sumstr, rotmat, dat[:3])
        adcpo[nm].values = dat.copy()
   
    adcpo = _set_coords(adcpo, cs_new)
    
    return adcpo


def calc_beam_orientmat(theta=20, convex=True, degrees=True):
    """Calculate the rotation matrix from beam coordinates to
    instrument head coordinates for an RDI ADCP.

    Parameters
    ----------
    theta : is the angle of the heads (usually 20 or 30 degrees)

    convex : is a flag for convex or concave head configuration.

    degrees : is a flag which specifies whether theta is in degrees
        or radians (default: degrees=True)
    """
    if degrees:
        theta = np.deg2rad(theta)
    if convex == 0 or convex == -1:
        c = -1
    else:
        c = 1
    a = 1 / (2. * np.sin(theta))
    b = 1 / (4. * np.cos(theta))
    d = a / (2. ** 0.5)
    return np.array([[c * a, -c * a, 0, 0],
                     [0, 0, -c * a, c * a],
                     [b, b, b, b],
                     [d, d, -d, -d]])


def calc_orientmat(adcpo):
    """
    Calculate the orientation matrix using the raw 
    heading, pitch, roll values from the RDI binary file.

    Parameters
    ----------
    adcpo : The ADP object containing the data.
    
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

    r = np.deg2rad(adcpo['roll'].values)
    p = np.arctan(np.tan(np.deg2rad(adcpo['pitch'].values)) * np.cos(r))
    h = np.deg2rad(adcpo['heading'].values)
    
    if 'rdi' in adcpo.inst_make.lower():
        if adcpo.orientation == 'up':
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
        if (adcpo.coord_sys == 'ship' and adcpo.use_pitchroll == 'yes'):
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

    return omat
