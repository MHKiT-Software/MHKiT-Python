"""Module containing functions to clean data."""

import warnings
from typing import Optional
import numpy as np
import xarray as xr
from scipy.signal import medfilt
from ..tools.misc import medfiltnan
from ..rotate.api import rotate2
from ..rotate.base import quaternion2orient


def __check_for_range_offset(ds) -> float:
    """
    Determines the range offset based on a variety of possible dataset attributes.
    The function first checks if specific attributes are present in the dataset (`ds`)
    and calculates the range offset accordingly. If the attribute `h_deploy`
    is found, it is renamed to `range_offset` with a deprecation warning.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing attributes used to calculate the range offset.

    Returns
    -------
    float
        The calculated or retrieved range offset. Returns 0 if no
        relevant attributes are present.

    Raises
    ------
    DeprecationWarning
        Warns that the attribute `h_deploy` is deprecated and has been
        renamed to `range_offset`.
    """

    if "bin1_dist_m" in ds.attrs:
        return ds.attrs["bin1_dist_m"] - ds.attrs["blank_dist"] - ds.attrs["cell_size"]
    elif "range_offset" in ds.attrs:
        return ds.attrs["range_offset"]
    elif "h_deploy" in ds.attrs:
        warnings.warn(
            "The attribute 'h_deploy' is no longer in use."
            "It will now be renamed to 'range_offset'.",
            DeprecationWarning,
        )
        ds.attrs["range_offset"] = ds.attrs.pop("h_deploy")
        return ds.attrs["range_offset"]
    else:
        return 0.0


def set_range_offset(ds, range_offset) -> None:
    """
    Adds an instrument's height above seafloor (for an up-facing instrument)
    or depth below water surface (for a down-facing instrument) to the range
    coordinate. Also adds an attribute to the Dataset with the current
    "range_offset" distance.

    Parameters
    ----------
    ds : xarray.Dataset
      The ADCP dataset to adjust 'range' on
    range_offset : numeric
      Deployment location in the water column, in [m]

    Returns
    -------
    ds : xarray.Dataset
      The ADCP dataset with applied range_offset

    Notes
    -----
    `Center of bin 1 = range_offset + blank_dist + cell_size`

    Nortek doesn't take `range_offset` into account, so the range that DOLfYN
    calculates distance is from the ADCP transducers. TRDI asks for `range_offset`
    input in their deployment software and is thereby known by DOLfYN.

    If the ADCP is mounted on a tripod on the seafloor, `range_offset` will be
    the height of the tripod +/- any extra distance to the transducer faces.
    If the instrument is vessel-mounted, `range_offset` is the distance between
    the surface and downward-facing ADCP's transducers.
    """

    current_offset = __check_for_range_offset(ds)

    if current_offset:
        warnings.warn(
            "The 'range_offset' is either already known or can be calculated "
            f"from 'bin1_dist_m': {current_offset} m. If you would like to "
            f"override this value with {range_offset} m, ignore this warning. "
            "If you do not want to override this value, you do not need to use "
            "this function."
        )
        # Remove offset from depth variable if exists
        if "depth" in ds.data_vars:
            ds["depth"].values -= current_offset

    # Add offset to each range coordinate
    r = [s for s in ds.dims if "range" in s]
    for coord in r:
        coord_attrs = ds[coord].attrs
        ds[coord] = ds[coord].values + range_offset
        ds[coord].attrs = coord_attrs

    # Add to depth variable if exists
    if "depth" in ds.data_vars:
        ds["depth"].values += range_offset

    # Add to dataset
    ds.attrs["range_offset"] = range_offset


def find_surface(*args, **kwargs):
    """
    Deprecated function. Use `water_depth_from_amplitude` instead.
    """
    warnings.warn(
        "The 'find_surface' function was renamed to 'water_depth_from_amplitude"
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return water_depth_from_amplitude(*args, **kwargs)


def water_depth_from_amplitude(ds, thresh=10, nfilt=None) -> None:
    """
    Find the surface (water level or seafloor) from amplitude data and
    adds the variable "depth" to the input Dataset.

    Depth is either water depth or the distance from the ADCP to
    surface/seafloor, depending on if "range_offset" has been set.

    Parameters
    ----------
    ds : xarray.Dataset
      The full adcp dataset
    thresh : int
      Specifies the threshold used in detecting the surface. Default = 10
      (The amount that amplitude must increase by near the surface for it to
      be considered a surface hit.)
    nfilt : int
      Specifies the width of the median filter applied, must be odd.
      Default is None

    Returns
    -------
    None, operates "in place" and adds the variable "depth" to the
    input dataset
    """

    if "depth" in ds.data_vars:
        raise Exception(
            "The variable 'depth' already exists. "
            "Please manually remove 'depth' if it needs to be recalculated."
        )

    # Use "avg" velocty if standard isn't available.
    # Should not matter which is used.
    tag = []
    if hasattr(ds, "vel"):
        tag += [""]
    if hasattr(ds, "vel_avg"):
        tag += ["_avg"]

    # This finds the maximum of the echo profile:
    inds = np.argmax(ds["amp" + tag[0]].values, axis=1)
    # This finds the first point that increases (away from the profiler) in
    # the echo profile
    edf = np.diff(ds["amp" + tag[0]].values.astype(np.int16), axis=1)
    inds2 = (
        np.nanmax(
            (edf < 0)
            * np.arange(ds["vel" + tag[0]].shape[1] - 1, dtype=np.uint8)[None, :, None],
            axis=1,
        )
        + 1
    )

    # Calculate the depth of these quantities
    d1 = ds["range" + tag[0]].values[inds]
    d2 = ds["range" + tag[0]].values[inds2]
    # Combine them:
    D = np.vstack((d1, d2))
    # Take the median value as the estimate of the surface:
    d = np.nanmedian(D, axis=0)

    # Throw out values that do not increase near the surface by *thresh*
    for ip in range(ds["vel" + tag[0]].shape[1]):
        itmp = np.nanmin(inds[:, ip])
        if (edf[itmp:, :, ip] < thresh).all():
            d[ip] = np.nan

    if nfilt:
        dfilt = medfiltnan(d, nfilt, thresh=4)
        dfilt[dfilt == 0] = np.nan
        d = dfilt

    range_offset = __check_for_range_offset(ds)
    if range_offset:
        d += range_offset
        long_name = "Water Depth"
    else:
        long_name = "Instrument Depth"

    ds["depth"] = xr.DataArray(
        d.astype("float32"),
        dims=["time" + tag[0]],
        attrs={"units": "m", "long_name": long_name, "standard_name": "depth"},
    )


def find_surface_from_P(*args, **kwargs):
    """
    Deprecated function. Use `water_depth_from_pressure` instead.
    """
    warnings.warn(
        "The 'find_surface_from_P' function was renamed to 'water_depth_from_pressure"
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return water_depth_from_pressure(*args, **kwargs)


def water_depth_from_pressure(ds, salinity=35) -> None:
    """
    Calculates the distance to the water surface. Temperature and salinity
    are used to calculate seawater density, which is in turn used with the
    pressure data to calculate depth.

    Depth is either water depth or the distance from the ADCP to
    surface/seafloor, depending on if "range_offset" has been set.

    Parameters
    ----------
    ds : xarray.Dataset
      The full adcp dataset
    salinity: numeric
      Water salinity in PSU. Default = 35

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

    if "depth" in ds.data_vars:
        raise Exception(
            "The variable 'depth' already exists. "
            "Please manually remove 'depth' if it needs to be recalculated."
        )
    pressure = [v for v in ds.data_vars if "pressure" in v]
    if not pressure:
        raise NameError("The variable 'pressure' does not exist.")
    else:
        for p in pressure:
            if not ds[p].sum():
                pressure.remove(p)
        if not pressure:
            raise ValueError("Pressure data not recorded.")
    temp = [
        v
        for v in ds.data_vars
        if (("temp" in v) and ("clock" not in v) and ("press" not in v))
    ]
    if not temp:
        raise NameError("The variable 'temp' does not exist.")

    # Density calcation
    P = ds[pressure[0]].values  # pressure, dbar
    T = ds[temp[0]].values  # temperature, degC
    S = salinity  # practical salinity
    rho0 = 1027  # kg/m^3
    T0 = 10  # degC
    S0 = 35  # psu assumed equivalent to ppt
    a = 0.15  # thermal expansion coefficient, kg/m^3/degC
    b = 0.78  # haline contraction coefficient, kg/m^3/ppt
    k = 4.5e-3  # adiabatic compressibility, kg/m^3/dbar
    rho = rho0 - a * (T - T0) + b * (S - S0) + k * P

    # Depth = pressure (conversion from dbar to MPa) / water weight
    d = (P * 10000) / (9.81 * rho)

    # Apply range_offset if available
    range_offset = __check_for_range_offset(ds)
    if range_offset:
        d += range_offset
        long_name = "Water Depth"
    else:
        long_name = "Instrument Depth"

    ds["water_density"] = xr.DataArray(
        rho.astype("float32"),
        dims=[ds[pressure[0]].dims[0]],
        attrs={
            "units": "kg m-3",
            "long_name": "Water Density",
            "standard_name": "sea_water_density",
            "description": "Water density from linear approximation of sea water equation of state",
        },
    )
    ds["depth"] = xr.DataArray(
        d.astype("float32"),
        dims=[ds[pressure[0]].dims[0]],
        attrs={"units": "m", "long_name": long_name, "standard_name": "depth"},
    )


def nan_beyond_surface(*args, **kwargs):
    """
    Deprecated function. Use `remove_surface_interference` instead.
    """
    warnings.warn(
        "The 'nan_beyond_surface' function was renamed to 'remove_surface_interference"
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return remove_surface_interference(*args, **kwargs)


def remove_surface_interference(
    ds, val=np.nan, beam_angle=None, cell_size=None, inplace=False
) -> Optional[xr.Dataset]:
    """
    Mask the values of 3D data (vel, amp, corr, echo) that are beyond the surface.

    Parameters
    ----------
    ds : xarray.Dataset
      The adcp dataset to clean
    val : nan or numeric
      Specifies the value to set the bad values to. Default is `numpy.nan`
    beam_angle : int
      ADCP beam inclination angle in degrees. Default = dataset.attrs['beam_angle']
    cell_size : float
      ADCP beam cellsize in meters. Default = dataset.attrs['cell_size']
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

    if "depth" not in ds.data_vars:
        raise KeyError(
            "Depth variable 'depth' does not exist in input dataset."
            "Please calculate 'depth' using the function 'water_depth_from_pressure'"
            "or 'water_depth_from_amplitude, or it can be found from the 'dist_bt'"
            "(bottom track) or 'dist_alt' (altimeter) variables, if available."
        )

    if beam_angle is None:
        if hasattr(ds, "beam_angle"):
            beam_angle = np.deg2rad(ds.attrs["beam_angle"])
        else:
            raise KeyError(
                "'beam_angle` not found in dataset attributes. "
                "Please supply the ADCP's beam angle."
            )
    else:
        beam_angle = np.deg2rad(beam_angle)

    if cell_size is None:
        # Fetch cell size
        cell_sizes = [
            a
            for a in ds.attrs
            if (
                ("cell_size" in a)
                and ("_bt" not in a)
                and ("_alt" not in a)
                and ("wave" not in a)
            )
        ]
        if cell_sizes:
            cs = cell_sizes[0]
        else:
            raise KeyError(
                "'cell_size` not found in dataset attributes. "
                "Please supply the ADCP's cell size."
            )
    else:
        cs = [cell_size]

    if not inplace:
        ds = ds.copy(deep=True)

    # Get all variables with 'range' coordinate
    profile_vars = [h for h in ds.keys() if any(s for s in ds[h].dims if "range" in s)]

    # Apply range_offset if available
    range_offset = __check_for_range_offset(ds)
    if range_offset:
        range_limit = (
            (ds["depth"] - range_offset) * np.cos(beam_angle) - ds.attrs[cs]
        ) + range_offset
    else:
        range_limit = ds["depth"] * np.cos(beam_angle) - ds.attrs[cs]

    # Echosounder data needs only be trimmed at water surface
    if "echo" in profile_vars:
        mask_echo = ds["range_echo"] > ds["depth"]
        ds["echo"].values[..., mask_echo] = val
        profile_vars.remove("echo")

    # Correct profile measurements for surface interference
    for var in profile_vars:
        # Use correct coordinate tag
        if "_" in var and ("gd" not in var):
            tag = "_" + "_".join(var.split("_")[1:])
        else:
            tag = ""
        mask = ds["range" + tag] > range_limit
        # Remove values
        a = ds[var].values
        try:  # float dtype
            a[..., mask] = val
        except:  # int dtype
            a[..., mask] = 0
        ds[var].values = a

    if not inplace:
        return ds


def correlation_filter(ds, thresh=50, inplace=False) -> Optional[xr.Dataset]:
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
    tag = []
    if hasattr(ds, "vel"):
        tag += [""]
    if hasattr(ds, "vel_b5"):
        tag += ["_b5"]
    if hasattr(ds, "vel_avg"):
        tag += ["_avg"]

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
                + ds["corr" + tg].units
            )

    rotate2(ds, coord_sys_orig, inplace=True)

    if not inplace:
        return ds


def medfilt_orient(ds, nfilt=7) -> xr.Dataset:
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


def val_exceeds_thresh(var, thresh=5, val=np.nan) -> xr.DataArray:
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
    ds : xarray.DataArray
      The adcp dataarray with datapoints beyond thresh are set to `val`
    """

    var = var.copy(deep=True)

    bd = np.zeros(var.shape, dtype="bool")
    bd |= np.abs(var.values) > thresh

    var.values[bd] = val

    return var


def fillgaps_time(var, method="cubic", maxgap=None) -> xr.DataArray:
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


def fillgaps_depth(var, method="cubic", maxgap=None) -> xr.DataArray:
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
