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

    r = [s for s in ds.dims if "range" in s]
    for val in r:
        ds[val] = ds[val].values + h_deploy
        ds[val].attrs["units"] = "m"

    if hasattr(ds, "h_deploy"):
        ds.attrs["h_deploy"] += h_deploy
    else:
        ds.attrs["h_deploy"] = h_deploy


def find_surface(ds, thresh=10, nfilt=None):
    """
    Find the surface (water level or seafloor) from amplitude data and
    adds the variable "depth" to the input Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
      The full adcp dataset
    thresh : int
      Specifies the threshold used in detecting the surface. Default = 10
      (The amount that amplitude must increase by near the surface for it to
      be considered a surface hit)
    nfilt : int
      Specifies the width of the median filter applied, must be odd.
      Default is None

    Returns
    -------
    None, operates "in place"
    """

    # This finds the maximum of the echo profile:
    inds = np.argmax(ds.amp.values, axis=1)
    # This finds the first point that increases (away from the profiler) in
    # the echo profile
    edf = np.diff(ds.amp.values.astype(np.int16), axis=1)
    inds2 = (
        np.max(
            (edf < 0) * np.arange(ds.vel.shape[1] - 1, dtype=np.uint8)[None, :, None],
            axis=1,
        )
        + 1
    )

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
        dfilt = medfiltnan(d, nfilt, thresh=4)
        dfilt[dfilt == 0] = np.NaN
        d = dfilt

    ds["depth"] = xr.DataArray(
        d.astype("float32"),
        dims=["time"],
        attrs={
            "units": "m",
            "long_name": "Depth",
            "standard_name": "depth",
            "positive": "down",
        },
    )


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
      Water salinity in psu. Default = 35

    Returns
    -------
    None, operates "in place" and adds the variables "water_density" and
    "depth" to the input dataset.

    Notes
    -----
    Requires that the instrument's pressure sensor was calibrated/zeroed
    before deployment to remove atmospheric pressure.

    Calculates seawater density using a linear approximation of the sea
    water equation of state:

    .. math:: \\rho - \\rho_0 = -\\alpha (T-T_0) + \\beta (S-S_0) + \\kappa P

    Where :math:`\\rho` is water density, :math:`T` is water temperature,
    :math:`P` is water pressure, :math:`S` is practical salinity,
    :math:`\\alpha` is the thermal expansion coefficient, :math:`\\beta` is
    the haline contraction coefficient, and :math:`\\kappa` is adiabatic
    compressibility.
    """

    # Density calcation
    P = ds.pressure.values
    T = ds.temp.values  # temperature, degC
    S = salinity  # practical salinity
    rho0 = 1027  # kg/m^3
    T0 = 10  # degC
    S0 = 35  # psu assumed equivalent to ppt
    a = 0.15  # thermal expansion coefficient, kg/m^3/degC
    b = 0.78  # haline contraction coefficient, kg/m^3/ppt
    k = 4.5e-3  # adiabatic compressibility, kg/m^3/dbar
    rho = rho0 - a * (T - T0) + b * (S - S0) + k * P

    # Depth = pressure (conversion from dbar to MPa) / water weight
    d = (ds.pressure * 10000) / (9.81 * rho)

    if hasattr(ds, "h_deploy"):
        d += ds.h_deploy
        description = "Depth to Seafloor"
    else:
        description = "Depth to Instrument"

    ds["water_density"] = xr.DataArray(
        rho.astype("float32"),
        dims=["time"],
        attrs={
            "units": "kg m-3",
            "long_name": "Water Density",
            "standard_name": "sea_water_density",
            "description": "Water density from linear approximation of sea water equation of state",
        },
    )
    ds["depth"] = xr.DataArray(
        d.astype("float32"),
        dims=["time"],
        attrs={
            "units": "m",
            "long_name": description,
            "standard_name": "depth",
            "positive": "down",
        },
    )


def nan_beyond_surface(ds, val=np.nan, beam_angle=None, inplace=False):
    """
    Mask the values of 3D data (vel, amp, corr, echo) that are beyond the surface.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean
    val : nan or numeric
      Specifies the value to set the bad values to. Default is `numpy.nan`
    beam_angle : int
      ADCP beam inclination angle. Default = dataset.attrs['beam_angle']
    inplace : bool
      When True the existing data object is modified. When False
      a copy is returned. Default = False

    Returns
    -------
    ds : xarray.Dataset
      Sets the adcp dataset where relevant arrays with values greater than `depth`
      set to NaN

    Notes
    -----
    Surface interference expected to happen at
    `distance > range * cos(beam angle) - cell size`
    """

    if not inplace:
        ds = ds.copy(deep=True)

    # Get all variables with 'range' coordinate
    var = [h for h in ds.keys() if any(s for s in ds[h].dims if "range" in s)]

    if beam_angle is None:
        if hasattr(ds, "beam_angle"):
            beam_angle = ds.beam_angle * (np.pi / 180)
        else:
            raise Exception(
                "'beam_angle` not found in dataset attributes. "
                "Please supply the ADCP's beam angle."
            )

    # Surface interference distance calculated from distance of transducers to surface
    if hasattr(ds, "h_deploy"):
        range_limit = (
            (ds.depth - ds.h_deploy) * np.cos(beam_angle) - ds.cell_size
        ) + ds.h_deploy
    else:
        range_limit = ds.depth * np.cos(beam_angle) - ds.cell_size

    bds = ds.range > range_limit

    # Echosounder data needs only be trimmed at water surface
    if "echo" in var:
        bds_echo = ds.range_echo > ds.depth
        ds["echo"].values[..., bds_echo] = val
        var.remove("echo")

    # Correct rest of "range" data for surface interference
    for nm in var:
        a = ds[nm].values
        try:  # float dtype
            a[..., bds] = val
        except:  # int dtype
            a[..., bds] = 0
        ds[nm].values = a

    if not inplace:
        return ds


def correlation_filter(ds, thresh=50, inplace=False):
    """
    Filters out data where correlation is below a threshold in the
    along-beam correlation data.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean.
    thresh : numeric
      The maximum value of correlation to screen, in counts or %.
      Default = 50
    inplace : bool
      When True the existing data object is modified. When False
      a copy is returned. Default = False

    Returns
    -------
    ds : xarray.Dataset
      Elements in velocity, correlation, and amplitude are removed if below the
      correlation threshold

    Notes
    -----
    Does not edit correlation or amplitude data.
    """

    if not inplace:
        ds = ds.copy(deep=True)

    # 4 or 5 beam
    if hasattr(ds, "vel_b5"):
        tag = ["", "_b5"]
    else:
        tag = [""]

    # copy original ref frame
    coord_sys_orig = ds.coord_sys

    # correlation is always in beam coordinates
    rotate2(ds, "beam", inplace=True)
    # correlation is always in beam coordinates
    for tg in tag:
        mask = ds["corr" + tg].values <= thresh

        for var in ["vel", "corr", "amp"]:
            try:
                ds[var + tg].values[mask] = np.nan
            except:
                ds[var + tg].values[mask] = 0
            ds[var + tg].attrs["Comments"] = (
                "Filtered of data with a correlation value below "
                + str(thresh)
                + ds.corr.units
            )

    rotate2(ds, coord_sys_orig, inplace=True)

    if not inplace:
        return ds


def medfilt_orient(ds, nfilt=7):
    """
    Median filters the orientation data (heading-pitch-roll or quaternions)

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean
    nfilt : numeric
      The length of the median-filtering kernel. Must be odd.
      Default = 7

    Return
    ------
    ds : xarray.Dataset
      The adcp dataset with the filtered orientation data

    See Also
    --------
    scipy.signal.medfilt()
    """

    ds = ds.copy(deep=True)

    if getattr(ds, "has_imu"):
        q_filt = np.zeros(ds.quaternions.shape)
        for i in range(ds.quaternions.q.size):
            q_filt[i] = medfilt(ds.quaternions[i].values, nfilt)
        ds.quaternions.values = q_filt

        ds["orientmat"] = quaternion2orient(ds.quaternions)
        return ds

    else:
        # non Nortek AHRS-equipped instruments
        do_these = ["pitch", "roll", "heading"]
        for nm in do_these:
            ds[nm].values = medfilt(ds[nm].values, nfilt)

        return ds.drop_vars("orientmat")


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
      The maximum value of velocity to screen. Default = 5
    val : nan or numeric
      Specifies the value to set the bad values to. Default is `numpy.nan`

    Returns
    -------
    ds : xarray.Dataset
      The adcp dataset with datapoints beyond thresh are set to `val`
    """

    var = var.copy(deep=True)

    bd = np.zeros(var.shape, dtype="bool")
    bd |= np.abs(var.values) > thresh

    var.values[bd] = val

    return var


def fillgaps_time(var, method="cubic", maxgap=None):
    """
    Fill gaps (nan values) in var across time using the specified method

    Parameters
    ----------
    var : xarray.DataArray
      The variable to clean
    method : string
      Interpolation method to use. Default is 'cubic'
    maxgap : numeric
      Maximum gap of missing data to interpolate across. Default is None

    Returns
    -------
    out : xarray.DataArray
      The input DataArray 'var' with gaps in 'var' interpolated across time

    See Also
    --------
    xarray.DataArray.interpolate_na()
    """

    time_dim = [t for t in var.dims if "time" in t][0]

    return var.interpolate_na(
        dim=time_dim, method=method, use_coordinate=True, limit=maxgap
    )


def fillgaps_depth(var, method="cubic", maxgap=None):
    """
    Fill gaps (nan values) in var along the depth profile using the specified method

    Parameters
    ----------
    var : xarray.DataArray
      The variable to clean
    method : string
      Interpolation method to use. Default is 'cubic'
    maxgap : numeric
      Maximum gap of missing data to interpolate across. Default is None

    Returns
    -------
    out : xarray.DataArray
      The input DataArray 'var' with gaps in 'var' interpolated across depth

    See Also
    --------
    xarray.DataArray.interpolate_na()
    """

    range_dim = [t for t in var.dims if "range" in t][0]

    return var.interpolate_na(
        dim=range_dim, method=method, use_coordinate=False, limit=maxgap
    )
