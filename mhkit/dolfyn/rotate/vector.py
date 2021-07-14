from __future__ import division
import numpy as np
import warnings
from . import base as rotb


def beam2inst(dat, reverse=False, force=False):
    # Remember: order of rotations matters!
    if not reverse: # beam->inst
        dat = rotb.beam2inst(dat, reverse=reverse, force=force)
        dat = _rotate_head2inst(dat)
    else: # inst->beam
        # First rotate velocities back to head frame
        dat = _rotate_head2inst(dat, reverse)
        # Now rotate to beam
        dat = rotb.beam2inst(dat, reverse=reverse, force=force)

    # Set the docstring to match the default rotation func
    beam2inst.__doc_ = rotb.beam2inst.__doc__
    
    return dat

def inst2earth(advo, reverse=False, rotate_vars=None, force=False):
    """
    Rotate data in an ADV object to the earth from the instrument
    frame (or vice-versa).

    Parameters
    ----------
    advo : The adv object containing the data.

    reverse : bool (default: False)
           If True, this function performs the inverse rotation
           (principal->earth).

    rotate_vars : iterable
      The list of variables to rotate. By default this is taken from
      advo.props['rotate_vars'].

    force : Do not check which frame the data is in prior to
      performing this rotation.

    """
    if reverse: # earth->inst
        # The transpose of the rotation matrix gives the inverse
        # rotation, so we simply reverse the order of the einsum:
        sumstr = 'jik,j...k->i...k'
        cs_now = 'earth'
        cs_new = 'inst'
    else: # inst->earth
        sumstr = 'ijk,j...k->i...k'
        cs_now = 'inst'
        cs_new = 'earth'

    if rotate_vars is None:
        if 'rotate_vars' in advo.attrs:
            rotate_vars = advo.rotate_vars
        else:
            rotate_vars = ['vel']

    cs = advo.coord_sys.lower()
    if not force:
        if cs == cs_new:
            print("Data is already in the '%s' coordinate system" % cs_new)
            return
        elif cs != cs_now:
            raise ValueError(
                "Data must be in the '%s' frame when using this function" %
                cs_now)

    if hasattr(advo, 'orientmat'):
        omat = advo['orientmat'].values
    else:
        if 'vector' in advo.inst_model.lower():
            orientation_down = advo['orientation_down']
        else:
            orientation_down = None
        omat = calc_omat(advo['heading'].values, advo['pitch'].values,
                         advo['roll'].values, orientation_down)

    # Take the transpose of the orientation to get the inst->earth rotation
    # matrix.
    rmat = np.rollaxis(omat, 1)

    _dcheck = rotb._check_rotmat_det(rmat)
    if not _dcheck.all():
        warnings.warn("Invalid orientation matrix"
                      " (determinant != 1) at"
                      " indices: {}."
                      .format(np.nonzero(~_dcheck)[0]),
                      rotb.BadDeterminantWarning)

    for nm in rotate_vars:
        n = advo[nm].shape[0]
        if n != 3:
            raise Exception("The entry {} is not a vector, it cannot "
                            "be rotated.".format(nm))
        advo[nm].values = np.einsum(sumstr, rmat, advo[nm])
    
    advo = rotb._set_coords(advo, cs_new)

    return advo
      

def calc_omat(hh, pp, rr, orientation_down=None):
    rr = rr.copy()
    pp = pp.copy()
    hh = hh.copy()
    if np.isnan(rr[-1]) and np.isnan(pp[-1]) and np.isnan(hh[-1]):
        # The end of the data may not have valid orientations
        lastgd = np.nonzero(~np.isnan(rr + pp + hh))[0][-1]
        rr[lastgd:] = rr[lastgd]
        pp[lastgd:] = pp[lastgd]
        hh[lastgd:] = hh[lastgd]
    if orientation_down is not None:
        # NOTE: For Nortek Vector ADVs: 'down' configuration means the
        #       head was pointing UP!  Check the Nortek coordinate
        #       transform matlab script for more info.  The 'up'
        #       orientation corresponds to the communication cable
        #       being up.  This is ridiculous, but apparently a
        #       reality.
        #       In DOLfYN 'orientation_down' = True indicates the ADV probe head
        #       is pointing UP.
        rr[orientation_down] += 180

    # Take the transpose of the orientation to get the inst->earth rotation
    # matrix.
    return _euler2orient(hh, pp, rr)


def _rotate_head2inst(advo, reverse=False):
    if not _check_inst2head_rotmat(advo): # RAISE on bad values.
        # This object doesn't have a head2inst_rotmat, so we do nothing.
        return advo
    if reverse: 
        # transpose of inst2head gives head->inst
        advo['vel'].values = np.dot(advo['inst2head_rotmat'].T, advo['vel'])
    else: # head->inst
        # velocity is recorded by Nortek in inst coordinates
        advo['vel'].values = np.dot(advo['inst2head_rotmat'], advo['vel'])

    return advo

def _check_inst2head_rotmat(advo):
    # if advo.get('body2head_rotmat', None) is not None:
    #     warnings.warn(
    #         "body2head_rotmat will be deprecated in future versions of DOLfYN."
    #         "Use the `set_inst2head_rotmat` method instead.",
    #         DeprecationWarning)
    #     # head2inst is transpose of body2head
    #     advo.set_inst2head_rotmat(advo.drop('body2head_rotmat'))
    if advo.get('inst2head_rotmat', None) is None:
        # This is the default value, and we do nothing.
        return False
    if not advo.inst2head_rotmat_was_set:
        raise Exception(
            "The inst2head rotation matrix exists in props, "
            "but it was not set using `set_inst2head_rotmat.")
    if not rotb._check_rotmat_det(advo.inst2head_rotmat.values):
        raise ValueError("Invalid inst2head_rotmat"
                         " (determinant != 1).")
    return True


def earth2principal(advo, reverse=False):
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
    advo : The adv object containing the data.
    reverse : bool (default: False)
           If True, this function performs the inverse rotation
           (principal->earth).

    """
    #if 'principal_heading' not in advo.attrs:
    #    advo.attrs['principal_heading'] = calc_principal_heading(advo.vel)
    
    #try:
    # this is in degrees CW from North
    ang = np.deg2rad(90 - advo.principal_heading)
    # convert this to radians CCW from east (which is expected by
    # the rest of the function)
    # except KeyError:
    #     if 'principal_angle' in advo.attrs:
    #         warnings.warn(
    #             "'principal_angle' will be deprecated in a future release of "
    #             "DOLfYN. Please update your file to use ``principal_heading = "
    #             "(90 - np.rad2deg(principal_angle))``."
    #         )
    #         # This is in radians CCW from east
    #         ang = advo.principal_angle

    if reverse:
        cs_now = 'principal'
        cs_new = 'earth'
    else:
        ang *= -1
        cs_now = 'earth'
        cs_new = 'principal'

    cs = advo.coord_sys.lower()
    if cs == cs_new:
        print('Data is already in the %s coordinate system' % cs_new)
        return
    elif cs != cs_now:
        raise ValueError(
            'Data must be in the {} frame '
            'to use this function'.format(cs_now))

    # Calculate the rotation matrix:
    cp, sp = np.cos(ang), np.sin(ang)
    rotmat = np.array([[cp, -sp, 0],
                       [sp, cp, 0],
                       [0, 0, 1]], dtype=np.float32)

    # Perform the rotation:
    for nm in advo.rotate_vars:
        dat = advo[nm].values
        dat[:2] = np.einsum('ij,j...->i...', rotmat[:2, :2], dat[:2])
        advo[nm].values = dat.copy()
    
    # Finalize the output.
    advo = rotb._set_coords(advo, cs_new)
    
    return advo


def _euler2orient(heading, pitch, roll, units='degrees'):
    # THIS IS FOR NORTEK data ONLY!
    # The heading, pitch, roll used here are from the Nortek binary files.

    # Heading input is clockwise from North
    # Returns a rotation matrix that rotates earth (ENU) -> inst.
    # This is based on the Nortek `Transforms.m` file, available in
    # the refs folder.
    if units.lower() == 'degrees':
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        heading = np.deg2rad(heading)
    # I've fixed the definition of heading below to be consistent with the
    # right-hand-rule; heading is the angle positive counterclockwise from North
    # of the y-axis.
    # This also involved swapping the sign on sh in the def of omat
    # below from the values provided in the Nortek Matlab script.
    heading = (np.pi / 2 - heading)

    ch = np.cos(heading)
    sh = np.sin(heading)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # Note that I've transposed these values (from what is defined in
    # Nortek matlab script), so that this is earth->inst (as
    # orientation matrices are typically defined)
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

    return omat
