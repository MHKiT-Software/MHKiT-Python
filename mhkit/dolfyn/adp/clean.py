"""Module containing functions to clean data
"""
import numpy as np
import xarray as xr
from scipy.signal import medfilt
from ..tools.misc import medfiltnan
from ..rotate.api import rotate2
from ..rotate.base import _make_model, quaternion2orient


def set_range_offset(ds, h_deploy):
    """
    Adds an instrument's height above seafloor (for an up-facing instrument) 
    or depth below water surface (for a down-facing instrument) to the range
    coordinate. Also adds an attribute to the Dataset with the current 
    "h_deploy" distance.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to ajust 'range' on
    h_deploy : numeric
      Deployment location in the water column, in [m]

    Returns
    -------
    None, operates "in place"

    Notes
    -----
    `Center of bin 1 = h_deploy + blank_dist + cell_size`

    Nortek doesn't take `h_deploy` into account, so the range that DOLfYN 
    calculates distance is from the ADCP transducers. TRDI asks for `h_deploy` 
    input in their deployment software and is thereby known by DOLfYN.

    If the ADCP is mounted on a tripod on the seafloor, `h_deploy` will be
    the height of the tripod +/- any extra distance to the transducer faces.
    If the instrument is vessel-mounted, `h_deploy` is the distance between 
    the surface and downward-facing ADCP's transducers.

    """
    r = [s for s in ds.dims if 'range' in s]
    for val in r:
        ds[val] = ds[val].values + h_deploy
        ds[val].attrs['units'] = 'm'

    if hasattr(ds, 'h_deploy'):
        ds.attrs['h_deploy'] += h_deploy
    else:
        ds.attrs['h_deploy'] = h_deploy


def find_surface(ds, thresh=10, nfilt=None):
    """
    Find the surface (water level or seafloor) from amplitude data and
    adds the variable "depth" to the input Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
      The full adcp dataset
    thresh : int
      Specifies the threshold used in detecting the surface.
      (The amount that amplitude must increase by near the surface for it to
      be considered a surface hit)
    nfilt : int
      Specifies the width of the median filter applied, must be odd

    Returns
    -------
    None, operates "in place"

    """
    # This finds the maximum of the echo profile:
    inds = np.argmax(ds.amp.values, axis=1)
    # This finds the first point that increases (away from the profiler) in
    # the echo profile
    edf = np.diff(ds.amp.values.astype(np.int16), axis=1)
    inds2 = np.max((edf < 0) *
                   np.arange(ds.vel.shape[1] - 1,
                             dtype=np.uint8)[None, :, None], axis=1) + 1

    # Calculate the depth of these quantities
    d1 = ds.range.values[inds]
    d2 = ds.range.values[inds2]
    # Combine them:
    D = np.vstack((d1, d2))
    # Take the median value as the estimate of the surface:
    d = np.median(D, axis=0)

    # Throw out values that do not increase near the surface by *thresh*
    for ip in range(ds.vel.shape[1]):
        itmp = np.min(inds[:, ip])
        if (edf[itmp:, :, ip] < thresh).all():
            d[ip] = np.NaN

    if nfilt:
        dfilt = medfiltnan(d, nfilt, thresh=.4)
        dfilt[dfilt == 0] = np.NaN
        d = dfilt

    ds['depth'] = xr.DataArray(d, dims=['time'], attrs={'units': 'm'})


def find_surface_from_P(ds, salinity=35):
    """
    Calculates the distance to the water surface. Temperature and salinity
    are used to calculate seawater density, which is in turn used with the
    pressure data to calculate depth.

    Parameters
    ----------
    ds : xarray.Dataset
      The full adcp dataset
    salinity: numeric
      Water salinity in psu

    Returns
    -------
    None, operates "in place" and adds the variables "water_density" and 
    "depth" to the input dataset.

    Notes
    -----
    Requires that the instrument's pressure sensor was calibrated/zeroed
    before deployment to remove atmospheric pressure.

    Calculates seawater density at normal atmospheric pressure according 
    to the UNESCO 1981 equation of state. Does not include hydrostatic pressure.

    """
    # Density calcation
    T = ds.temp.values
    S = salinity
    # standard mean ocean water
    rho_smow = 999.842594 + 6.793953e-2*T - 9.095290e-3*T**2 + \
        1.001685e-4*T**3 - 1.120083e-6*T**4 + 6.536332e-9*T**5
    # at water surface
    B1 = 8.2449e-1 - 4.0899e-3*T + 7.6438e-5*T**2 - 8.2467e-7*T**3 + 5.3875e-9*T**4
    C1 = -5.7246e-3 + 1.0227e-4*T - 1.6546e-6*T**2
    d0 = 4.8314e-4
    rho_atm0 = rho_smow + B1*S + C1*S**1.5 + d0*S**2

    # Depth = pressure (conversion from dbar to MPa) / water weight
    d = (ds.pressure*10000)/(9.81*rho_atm0)

    if hasattr(ds, 'h_deploy'):
        d += ds.h_deploy
        description = "Water depth to seafloor"
    else:
        description = "Water depth to ADCP"

    ds['water_density'] = xr.DataArray(
        rho_atm0,
        dims=['time'],
        attrs={'units': 'kg/m^3',
               'description': 'Water density according to UNESCO 1981 equation of state'})
    ds['depth'] = xr.DataArray(d, dims=['time'], attrs={
                               'units': 'm', 'description': description})


def nan_beyond_surface(ds, val=np.nan):
    """
    Mask the values of 3D data (vel, amp, corr, echo) that are beyond the surface.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean
    val : nan or numeric
      Specifies the value to set the bad values to (default np.nan).

    Returns 
    -------
    ds : xarray.Dataset
      The adcp dataset where relevant arrays with values greater than 
      `depth` are set to NaN

    Notes
    -----
    Surface interference expected to happen at `r > depth * cos(beam_angle)`

    """
    ds = ds.copy(deep=True)

    # Get all variables with 'range' coordinate
    var = [h for h in ds.keys() if any(s for s in ds[h].dims if 'range' in s)]

    if 'nortek' in _make_model(ds):
        beam_angle = 25 * (np.pi/180)
    else:  # TRDI
        try:
            beam_angle = ds.beam_angle
        except:
            beam_angle = 20 * (np.pi/180)

    bds = ds.range > (ds.depth * np.cos(beam_angle) - ds.cell_size)

    if 'echo' in var:
        bds_echo = ds.range_echo > ds.depth
        ds['echo'].values[..., bds_echo] = val
        var.remove('echo')

    for nm in var:
        a = ds[nm].values
        if 'corr' in nm:
            a[..., bds] = 0
        else:
            a[..., bds] = val
        ds[nm].values = a

    return ds


def val_exceeds_thresh(var, thresh=5, val=np.nan):
    """
    Find values of a variable that exceed a threshold value,
    and assign "val" to the velocity data where the threshold is
    exceeded.

    Parameters
    ----------
    var : xarray.DataArray
      Variable to clean
    thresh : numeric
      The maximum value of velocity to screen
    val : nan or numeric
      Specifies the value to set the bad values to (default np.nan)

    Returns
    -------
    ds : xarray.Dataset
      The adcp dataset with datapoints beyond thresh are set to `val`

    """
    var = var.copy(deep=True)

    bd = np.zeros(var.shape, dtype='bool')
    bd |= (np.abs(var.values) > thresh)

    var.values[bd] = val

    return var


def correlation_filter(ds, thresh=50, val=np.nan):
    """
    Filters out velocity data where correlation is below a 
    threshold in the beam correlation data.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean.
    thresh : numeric
      The maximum value of correlation to screen, in counts or %
    val : numeric
      Value to set masked correlation data to, default is nan

    Returns
    -------
    ds : xarray.Dataset
     Velocity data with low correlation values set to `val`

    Notes
    -----
    Does not edit correlation or amplitude data.

    """
    ds = ds.copy(deep=True)

    # 4 or 5 beam
    if hasattr(ds, 'vel_b5'):
        tag = ['', '_b5']
    else:
        tag = ['']

    # copy original ref frame
    coord_sys_orig = ds.coord_sys

    # correlation is always in beam coordinates
    rotate2(ds, 'beam', inplace=True)
    for tg in tag:
        mask = (ds['corr'+tg].values <= thresh)
        ds['vel'+tg].values[mask] = val
        ds['vel'+tg].attrs['Comments'] = 'Filtered of data with a correlation value below ' + \
            str(thresh) + ds.corr.units

    rotate2(ds, coord_sys_orig, inplace=True)

    return ds


def medfilt_orient(ds, nfilt=7):
    """
    Median filters the orientation data (heading-pitch-roll or quaternions)

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean
    nfilt : numeric
      The length of the median-filtering kernel
      *nfilt* must be odd.

    Return
    ------
    ds : xarray.Dataset
      The adcp dataset with the filtered orientation data

    See Also
    --------
    scipy.signal.medfilt()

    """
    ds = ds.copy(deep=True)

    if getattr(ds, 'has_imu'):
        q_filt = np.zeros(ds.quaternions.shape)
        for i in range(ds.quaternions.q.size):
            q_filt[i] = medfilt(ds.quaternions[i].values, nfilt)
        ds.quaternions.values = q_filt

        ds['orientmat'] = quaternion2orient(ds.quaternions)
        return ds

    else:
        # non Nortek AHRS-equipped instruments
        do_these = ['pitch', 'roll', 'heading']
        for nm in do_these:
            ds[nm].values = medfilt(ds[nm].values, nfilt)

        return ds.drop_vars('orientmat')


def fillgaps_time(var, method='cubic', max_gap=None):
    """
    Fill gaps (nan values) in var across time using the specified method

    Parameters
    ----------
    var : xarray.DataArray
      The variable to clean
    method : string
      Interpolation method to use
    max_gap : numeric
      Max number of consective NaN's to interpolate across

    Returns
    -------
    ds : xarray.Dataset
      The adcp dataset with gaps in velocity interpolated across time

    See Also
    --------
    xarray.DataArray.interpolate_na()

    """
    var = var.copy(deep=True)
    time_dim = [t for t in var.dims if 'time' in t][0]

    return var.interpolate_na(dim=time_dim, method=method,
                              use_coordinate=True,
                              max_gap=max_gap)


def fillgaps_depth(var, method='cubic', max_gap=None):
    """
    Fill gaps (nan values) in var along the depth profile using the specified method

    Parameters
    ----------
    var : xarray.DataArray
      The variable to clean
    method : string
      Interpolation method to use
    max_gap : numeric
      Max number of consective NaN's to interpolate across

    Returns
    -------
    ds : xarray.Dataset
      The adcp dataset with gaps in velocity interpolated across depth profiles

    See Also
    --------
    xarray.DataArray.interpolate_na()

    """
    var = var.copy(deep=True)
    range_dim = [t for t in var.dims if 'range' in t][0]

    return var.interpolate_na(dim=range_dim, method=method,
                              use_coordinate=False,
                              max_gap=max_gap)
