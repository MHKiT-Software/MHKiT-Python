from . import vector as r_vec
from . import awac as r_awac
from . import signature as r_sig
from . import rdi as r_rdi
from .base import _make_model
import numpy as np
import xarray as xr
import warnings


# The 'rotation chain'
rc = ['beam', 'inst', 'earth', 'principal']

rot_module_dict = {
    # Nortek instruments
    'vector': r_vec,
    'awac': r_awac,
    'signature': r_sig,
    'ad2cp': r_sig,

    # TRDI instruments
    'rdi': r_rdi}


def rotate2(ds, out_frame='earth', inplace=False):
    """Rotate a dataset to a new coordinate system.

    Parameters
    ----------
    ds : xr.Dataset
      The dolfyn dataset (ADV or ADCP) to rotate.
    out_frame : string {'beam', 'inst', 'earth', 'principal'}
      The coordinate system to rotate the data into.
    inplace : bool
      Operate on the input data dataset (True), or return a copy that
      has been rotated (False, default).

    Returns
    -------
    ds : xarray.Dataset
      The rotated dataset
      
    Notes
    -----
    This function rotates all variables in ``ds.attrs['rotate_vars']``.

    """
    csin = ds.coord_sys.lower()
    if csin == 'ship':
        csin = 'inst'

    # Returns True/False if head2inst_rotmat has been set/not-set.
    r_vec._check_inst2head_rotmat(ds)

    if out_frame == 'principal' and csin != 'earth':
        warnings.warn(
            "You are attempting to rotate into the 'principal' "
            "coordinate system, but the dataset is in the {} "
            "coordinate system. Be sure that 'principal_angle' is "
            "defined based on the earth coordinate system.".format(csin))

    rmod = None
    for ky in rot_module_dict:
        if ky in _make_model(ds):
            rmod = rot_module_dict[ky]
            break
    if rmod is None:
        raise ValueError("Rotations are not defined for "
                         "instrument '{}'.".format(_make_model(ds)))
    if not inplace:
        ds = ds.copy(deep=True)

    # Get the 'indices' of the rotation chain
    try:
        iframe_in = rc.index(csin)
    except ValueError:
        raise Exception("The coordinate system of the input "
                        "dataset, '{}', is invalid."
                        .format(ds.coord_sys))
    try:
        iframe_out = rc.index(out_frame.lower())
    except ValueError:
        raise Exception("The specified output coordinate system "
                        "is invalid, please select one of: 'beam', 'inst', "
                        "'earth', 'principal'.")

    if iframe_out == iframe_in:
        print("Data is already in the {} coordinate system".format(out_frame))
        return ds

    if iframe_out > iframe_in:
        reverse = False
    else:
        reverse = True

    while ds.coord_sys.lower() != out_frame.lower():
        csin = ds.coord_sys
        if csin == 'ship':
            csin = 'inst'
        inow = rc.index(csin)
        if reverse:
            func = getattr(rmod, '_' + rc[inow - 1] + '2' + rc[inow])
        else:
            func = getattr(rmod, '_' + rc[inow] + '2' + rc[inow + 1])
        ds = func(ds, reverse=reverse)

    return ds


def calc_principal_heading(vel, tidal_mode=True):
    """Compute the principal angle of the horizontal velocity.

    Parameters
    ----------
    vel : np.ndarray (2,...,Nt), or (3,...,Nt)
      The 2D or 3D Veldata array (3rd-dim is ignored in this calculation)
    tidal_mode : bool (default: True)

    Returns
    -------
    p_heading : float or ndarray
      The principal heading in degrees clockwise from North.

    Notes
    -----
    The tidal mode follows these steps:
      1. rotates vectors with negative velocity by 180 degrees
      2. then doubles those angles to make a complete circle again
      3. computes a mean direction from this, and halves that angle again.
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
        pang = np.angle(
            np.nanmean(dt, -1, dtype=np.complex128)) / 2
    else:
        pang = np.angle(np.nanmean(dt, -1))
        
    return np.round((90 - np.rad2deg(pang)), decimals=4)


def set_declination(ds, declin):
    """Set the magnetic declination

    Parameters
    ----------
    declination : float
       The value of the magnetic declination in degrees (positive
       values specify that Magnetic North is clockwise from True North)

    Returns
    ----------
    ds : xarray.Dataset
        Dataset adjusted for the magnetic declination
        
    Notes
    -----
    This method modifies the data object in the following ways:

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
    if 'declination' in ds.attrs:
        angle = declin - ds.attrs.pop('declination')
    else:
        angle = declin
    cd = np.cos(-np.deg2rad(angle))
    sd = np.sin(-np.deg2rad(angle))

    # The ordering is funny here because orientmat is the
    # transpose of the inst->earth rotation matrix:
    Rdec = np.array([[cd, -sd, 0],
                     [sd, cd, 0],
                     [0, 0, 1]])

    if ds.coord_sys == 'earth':
        rotate2earth = True
        ds = rotate2(ds, 'inst', inplace=True)
    else:
        rotate2earth = False

    ds['orientmat'].values = np.einsum('kj...,ij->ki...',
                                            ds['orientmat'].values,
                                            Rdec, )
    if 'heading' in ds:
        ds['heading'] += angle
    if rotate2earth:
        ds = rotate2(ds, 'earth', inplace=True)
    if 'principal_heading' in ds.attrs:
        ds.attrs['principal_heading'] += angle

    ds.attrs['declination'] = declin
    ds.attrs['declination_in_orientmat'] = 1 # logical
    
    return ds


def set_inst2head_rotmat(ds, rotmat):
    """
    Set the instrument to head rotation matrix for the Nortek ADV if it
    hasn't already been set through a '.userdata.json' file.
    
    Parameters
    ----------
    rotmat : float
        3x3 rotation matrix
    
    Returns
    ----------
    ds : xarray.Dataset
        Dataset with rotation matrix applied
        
    """
    if not ds.inst_model.lower()=='vector':
        raise Exception("Setting 'inst2head_rotmat' is only supported "
                        "for Nortek Vector ADVs.")
    if ds.get('inst2head_rotmat', None) is not None:
        raise Exception(
            "You are setting 'inst2head_rotmat' after it has already "
            "been set. You can only set it once.")
        
    csin = ds.coord_sys
    if csin not in ['inst', 'beam']:
        ds = rotate2(ds, 'inst', inplace=True)

    ds['inst2head_rotmat'] = xr.DataArray(np.array(rotmat),
                                               dims=['x','x*'])
    
    ds.attrs['inst2head_rotmat_was_set'] = 1 # logical
    # Note that there is no validation that the user doesn't
    # change `ds.attrs['inst2head_rotmat']` after calling this
    # function.

    if not csin == 'beam': # csin not 'beam', then we're in inst
        ds = r_vec._rotate_inst2head(ds)
    if csin not in ['inst', 'beam']:
        ds = rotate2(ds, csin, inplace=True)
        
    return ds
