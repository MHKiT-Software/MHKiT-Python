import numpy as np
import scipy.signal as ss
from scipy.integrate import cumtrapz
import xarray as xr
from ..rotate import vector as rot
#from ..rotate import signature as sig
from ..rotate.api import _make_model, rotate2
import warnings


def _get_body2imu(make_model):
    if make_model == 'nortek vector':
        # In inches it is: (0.25, 0.25, 5.9)
        return np.array([0.00635, 0.00635, 0.14986])
    # Placeholder
    # elif make_model.startswith('nortek signature'):
    #     return np.array([0.0, 0.0, 0.0])
    else:
        raise Exception("The imu->body vector is unknown for this instrument.")


class _CalcMotion():
    """
    A 'calculator' for computing the velocity of points that are
    rigidly connected to an ADV-body with an IMU.

    Parameters
    ----------
    ds : xarray.Dataset
           The IMU-adv data that will be used to compute motion.

    accel_filtfreq : float
      the frequency at which to high-pass filter the acceleration
      sigal to remove low-frequency drift. (default: 0.03 Hz)

    vel_filtfreq : float (optional)
      a second frequency to high-pass filter the integrated
      acceleration.  (default: 1/3 of accel_filtfreq)

    """
    _default_accel_filtfreq = 0.03

    def __init__(self, ds,
                 accel_filtfreq=None,
                 vel_filtfreq=None,
                 to_earth=True):

        self.ds = ds
        self._check_filtfreqs(accel_filtfreq,
                              vel_filtfreq)
        self.to_earth = to_earth

        self._set_accel()
        self._set_acclow()
        self.angrt = ds['angrt'].values  # No copy because not modified.

    def _check_filtfreqs(self, accel_filtfreq, vel_filtfreq):
        datval = self.ds.attrs.get('motion accel_filtfreq Hz', None)
        if datval is None:
            if accel_filtfreq is None:
                accel_filtfreq = self._default_accel_filtfreq
                # else use the accel_filtfreq value
        else:
            if accel_filtfreq is None:
                accel_filtfreq = datval
            else:
                if datval != accel_filtfreq:
                    warnings.warn(
                        f"The default accel_filtfreq is {datval} Hz. "
                        "Overriding this with the user-specified "
                        "value: {accel_filtfreq} Hz.")
        if vel_filtfreq is None:
            vel_filtfreq = self.ds.attrs.get('motion vel_filtfreq Hz', None)
        if vel_filtfreq is None:
            vel_filtfreq = accel_filtfreq / 3.0
        self.accel_filtfreq = accel_filtfreq
        self.accelvel_filtfreq = vel_filtfreq

    def _set_accel(self, ):
        ds = self.ds
        if ds.coord_sys == 'inst':
            self.accel = np.einsum('ij...,i...->j...',
                                   ds['orientmat'].values,
                                   ds['accel'].values)
        elif self.ds.coord_sys == 'earth':
            self.accel = ds['accel'].values.copy()
        else:
            raise Exception(("Invalid coordinate system '%s'. The coordinate "
                             "system must either be 'earth' or 'inst' to "
                             "perform motion correction.")
                            % (self.ds.coord_sys))

    def _set_acclow(self, ):
        self.acclow = acc = self.accel.copy()
        if self.accel_filtfreq == 0:
            acc[:] = acc.mean(-1)[..., None]
        else:
            flt = ss.butter(1, self.accel_filtfreq / (self.ds.fs / 2))
            for idx in range(3):
                acc[idx] = ss.filtfilt(flt[0], flt[1], acc[idx])

    def calc_velacc(self, ):
        """
        Calculates the translational velocity from the high-pass
        filtered acceleration sigal.

        Returns
        -------
        velacc : |np.ndarray| (3 x n_time)
               The acceleration-induced velocity array (3, n_time).
        """
        samp_freq = self.ds.fs

        # Get high-pass accelerations
        hp = self.accel - self.acclow

        # Integrate in time to get velocities
        dat = np.concatenate((np.zeros(list(hp.shape[:-1]) + [1]),
                              cumtrapz(hp, dx=1 / samp_freq)), axis=-1)

        # Filter again?
        if self.accelvel_filtfreq > 0:
            filt_freq = self.accelvel_filtfreq
            # 2nd order Butterworth filter
            # Applied twice by 'filtfilt' = 4th order butterworth
            filt = ss.butter(2, float(filt_freq) / (samp_freq / 2))
            for idx in range(hp.shape[0]):
                dat[idx] = dat[idx] - ss.filtfilt(filt[0], filt[1], dat[idx])
        return dat

    def calc_velrot(self, vec, to_earth=None):
        """
        Calculate the induced velocity due to rotations of the instrument
        about the IMU center.

        Parameters
        ----------
        vec : |np.ndarray| (len(3) or 3 x M)
          The vector in meters (or vectors) from the body-origin
          (center of head end-cap) to the point of interest (in the
          body coord-sys).

        Returns
        -------
        velrot : |np.ndarray| (3 x M x N_time)
          The rotation-induced velocity array (3, n_time).

        """
        if to_earth is None:
            to_earth = self.to_earth

        dimflag = False
        if vec.ndim == 1:
            vec = vec[:3].copy().reshape((3, 1))
            dimflag = True

        # Correct for the body->imu distance.
        # The nortek_body2imu vector is subtracted because of
        # vector addition:
        # body2head = body2imu + imu2head
        # Thus:
        # imu2head = body2head - body2imu
        vec = vec - _get_body2imu(_make_model(self.ds))[:, None]

        # This motion of the point *vec* due to rotations should be the
        # cross-product of omega (rotation vector) and the vector.
        #   u=dz*omegaY-dy*omegaZ,v=dx*omegaZ-dz*omegaX,w=dy*omegaX-dx*omegaY
        # where vec=[dx,dy,dz], and angrt=[omegaX,omegaY,omegaZ]
        velrot = np.array([(vec[2][:, None] * self.angrt[1] -
                            vec[1][:, None] * self.angrt[2]),
                           (vec[0][:, None] * self.angrt[2] -
                            vec[2][:, None] * self.angrt[0]),
                           (vec[1][:, None] * self.angrt[0] -
                            vec[0][:, None] * self.angrt[1]),
                           ])

        if to_earth:
            velrot = np.einsum('ji...,j...->i...',
                               self.ds['orientmat'].values, velrot)

        if dimflag:
            return velrot[:, 0, :]

        return velrot


def _calc_probe_pos(ds, separate_probes=False):
    """
    Currently this only works for Nortek Vectors!

    In the future, we could use the transformation matrix (and a
    probe-length lookup-table?)

    """
    # According to the ADV_DataSheet, the probe-length radius is
    # 8.6cm @ 120deg from probe-stem axis.  If I subtract 1cm
    # to get acoustic receiver center, this is 7.6cm.
    # In the coordinate sys of the center of the probe
    # then, the positions of the centers of the receivers is:
    # if separate_probes and p['inst_make'].lower() == 'nortek' and\
    #    p['inst_model'].lower == 'vector':
    vec = ds.inst2head_vec
    if type(vec) != np.ndarray:
        vec = np.array(vec)

    if separate_probes and _make_model(ds) == 'nortek vector':
        r = 0.076
        # The angle between the x-y plane and the probes
        phi = np.deg2rad(-30)
        # The angles of the probes from the x-axis:
        theta = np.deg2rad(np.array([0., 120., 240.]))
        return (np.dot(ds['inst2head_rotmat'].values.T,
                       np.array([r * np.cos(theta),
                                 r * np.sin(theta),
                                 r * np.tan(phi) * np.ones(3)])) +
                vec[:, None])
    else:
        return vec


def correct_motion(ds,
                   accel_filtfreq=None,
                   vel_filtfreq=None,
                   to_earth=True,
                   separate_probes=False):
    """
    This function performs motion correction on an IMU-ADV data
    object. The IMU and ADV data should be tightly synchronized and
    contained in a single data object.

    Parameters
    ----------
    ds : xarray.Dataset
      Cleaned ADV dataset

    accel_filtfreq : float
      the frequency at which to high-pass filter the acceleration
      sigal to remove low-frequency drift.

    vel_filtfreq : float (optional)
      a second frequency to high-pass filter the integrated
      acceleration.  (default: 1/3 of accel_filtfreq)

    to_earth : bool (optional, default: True)
      All variables in the ds.props['rotate_vars'] list will be
      rotated into either the earth frame (to_earth=True) or the
      instrument frame (to_earth=False).

    separate_probes : bool (optional, default: False)
      a flag to perform motion-correction at the probe tips, and
      perform motion correction in beam-coordinates, then transform
      back into XYZ/earth coordinates. This correction seems to be
      lower than the noise levels of the ADV, so the default is to not
      use it (False).

    Returns
    -------
    This function returns None, it operates on the input data object,
    ``ds``. The following attributes are added to `ds`:

      ``velraw`` is the uncorrected velocity

      ``velrot`` is the rotational component of the head motion (from
                 angrt)

      ``velacc`` is the translational component of the head motion (from
                 accel, the high-pass filtered accel sigal)

      ``acclow`` is the low-pass filtered accel sigal (i.e., 

    The primary velocity vector attribute, ``vel``, is motion corrected
    such that:

          vel = velraw + velrot + velacc

    The sigs are correct in this equation. The measured velocity
    induced by head-motion is *in the opposite direction* of the head
    motion.  i.e. when the head moves one way in stationary flow, it
    measures a velocity in the opposite direction. Therefore, to
    remove the motion from the raw sigal we *add* the head motion.

    Notes
    -----

    Acceleration signals from inertial sensors are notorious for
    having a small bias that can drift slowly in time. When
    integrating these sigals to estimate velocity the bias is
    amplified and leads to large errors in the estimated
    velocity. There are two methods for removing these errors,

    1) high-pass filter the acceleration sigal prior and/or after
       integrating. This implicitly assumes that the low-frequency
       translational velocity is zero.
    2) provide a slowly-varying reference position (often from a GPS)
       to an IMU that can use the sigal (usually using Kalman
       filters) to debias the acceleration sigal.

    Because method (1) removes `real` low-frequency acceleration,
    method (2) is more accurate. However, providing reference position
    estimates to undersea instruments is practically challenging and
    expensive. Therefore, lacking the ability to use method (2), this
    function utilizes method (1).

    For deployments in which the ADV is mounted on a mooring, or other
    semi-fixed structure, the assumption of zero low-frequency
    translational velocity is a reasonable one. However, for
    deployments on ships, gliders, or other moving objects it is
    not. The measured velocity, after motion-correction, will still
    hold some of this contamination and will be a sum of the ADV
    motion and the measured velocity on long time scales.  If
    low-frequency motion is known separate from the ADV (e.g. from a
    bottom-tracking ADP, or from a ship's GPS), it may be possible to
    remove that sigal from the ADV sigal in post-processing.

    """
    # Ensure acting on new dataset
    ds = ds.copy(deep=True)

    if hasattr(ds, 'velrot') or ds.attrs.get('motion corrected', False):
        raise Exception('The data appears to already have been '
                        'motion corrected.')

    if not hasattr(ds, 'has_imu') or ('accel' not in ds):
        raise Exception('The instrument does not appear to have an IMU.')

    if ds.coord_sys != 'inst':
        rotate2(ds, 'inst', inplace=True)

    # Returns True/False if head2inst_rotmat has been set/not-set.
    # Bad configs raises errors (this is to check for those)
    rot._check_inst2head_rotmat(ds)

    # Create the motion 'calculator':
    calcobj = _CalcMotion(ds,
                          accel_filtfreq=accel_filtfreq,
                          vel_filtfreq=vel_filtfreq,
                          to_earth=to_earth)

    ##########
    # Calculate the translational velocity (from the accel):
    ds['velacc'] = xr.DataArray(calcobj.calc_velacc(),
                                dims=['dirIMU', 'time'])
    # Copy acclow to the adv-object.
    ds['acclow'] = xr.DataArray(calcobj.acclow,
                                dims=['dirIMU', 'time'])

    ##########
    # Calculate rotational velocity (from angrt):
    pos = _calc_probe_pos(ds, separate_probes)
    # Calculate the velocity of the head (or probes).
    velrot = calcobj.calc_velrot(pos, to_earth=False)
    if separate_probes:
        # The head->beam transformation matrix
        transMat = ds.get('beam2inst_orientmat', None)
        # The inst->head transformation matrix
        rmat = ds['inst2head_rotmat']

        # 1) Rotate body-coordinate velocities to head-coord.
        velrot = np.dot(rmat, velrot)
        # 2) Rotate body-coord to beam-coord (einsum),
        # 3) Take along beam-component (diagonal),
        # 4) Rotate back to head-coord (einsum),
        velrot = np.einsum('ij,kj->ik',
                           transMat,
                           np.diagonal(np.einsum('ij,j...->i...',
                                                 np.linalg.inv(transMat),
                                                 velrot)))
        # 5) Rotate back to body-coord.
        velrot = np.dot(rmat.T, velrot)
    # try:
    ds['velrot'] = xr.DataArray(velrot, dims=['dirIMU', 'time'])
    # except:
    #    ds['velrot'] = xr.DataArray(velrot, dims=['dirIMU','range','time'])

    ##########
    # Rotate the data into the correct coordinate system.
    # inst2earth expects a 'rotate_vars' property.
    # Add velrot, velacc, acclow, to it.
    if 'rotate_vars' not in ds.attrs:
        ds.attrs['rotate_vars'] = ['vel', 'velrot', 'velacc', 'accel',
                                   'acclow', 'angrt', 'mag']
    else:
        ds.attrs['rotate_vars'].extend(['velrot', 'velacc', 'acclow'])

    # NOTE: accel, acclow, and velacc are in the earth-frame after
    #       calc_velacc() call.
    # if _make_model(ds).startswith('nortek signature'):
    #     inst2earth = sig._inst2earth
    # else: # nortek vector
    inst2earth = rot._inst2earth
    if to_earth:
        ds['accel'].values = calcobj.accel
        to_remove = ['accel', 'acclow', 'velacc']
        ds = inst2earth(ds, rotate_vars=[e for e in
                                         ds.attrs['rotate_vars']
                                         if e not in to_remove])
    else:
        # rotate these variables back to the instrument frame.
        ds = inst2earth(ds, reverse=True,
                        rotate_vars=['accel', 'acclow', 'velacc'],
                        force=True)

    ##########
    # Copy vel -> velraw prior to motion correction:
    ds['vel_raw'] = xr.DataArray(ds.vel.copy(deep=True), dims=ds.vel.dims)
    # Add it to rotate_vars:
    ds.attrs['rotate_vars'].append('vel_raw')

    ##########
    # Remove motion from measured velocity! <woot!>
    # NOTE: The plus sign is because the measured-induced velocities
    #       are in the opposite direction of the head motion.
    #       i.e. when the head moves one way in stationary flow, it
    #       measures a velocity in the opposite direction.

    # use xarray to keep dimensions consistent
    velmot = ds['velrot'] + ds['velacc']
    velmot = velmot.values

    # if _make_model(ds).startswith('nortek signature'):
    #     # drop xarray to not break code between ADV and AD2CP
    #     ds['vel'][:3] += velmot[:,None,:]
    #     # This assumes these are w.
    #     ds['vel'][3:] += velmot[2:]
    # else: # nortek vector
    ds['vel'][:3] += velmot

    ds.attrs['motion corrected'] = 1
    ds.attrs['motion accel_filtfreq Hz'] = calcobj.accel_filtfreq

    return ds
