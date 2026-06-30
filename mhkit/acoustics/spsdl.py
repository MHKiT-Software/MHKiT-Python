"""
1. **Spectral Density Level Calculation**:

   - `sound_pressure_spectral_density_level`: Converts mean square spectral density values to
     sound pressure spectral density levels in dB.

2. **Spectral Density Aggregation**:

   - `band_aggregate`: Aggregates spectral density levels into fractional octave bands using
     specified statistical methods (e.g., median, mean).

   - `time_aggregate`: Aggregates spectral density levels into specified time windows using
     similar statistical methods.
"""

import warnings
from typing import Union, Dict, Tuple, Optional
import numpy as np
import xarray as xr

from mhkit.dolfyn.time import epoch2dt64, dt642epoch
from .analysis import _check_numeric, _fmax_warning, create_frequency_bands


def sound_pressure_spectral_density_level(spsd: xr.DataArray) -> xr.DataArray:
    """
    Calculates the sound pressure spectral density level from
    the mean square sound pressure spectral density.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in Pa^2/Hz

    Returns
    -------
    spsdl: xarray.DataArray (time, freq)
        Sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time and frequency
    """

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2 to 1 uPa^2

    # Sound pressure spectral density level from mean square values
    lpf = 10 * np.log10(spsd.values / reference)

    spsdl = xr.DataArray(
        lpf.astype(np.float32),
        coords=spsd.coords,
        attrs={
            "units": "dB re 1 uPa^2/Hz",
            "long_name": "Sound Pressure Spectral Density Level",
            "time_resolution": spsd.attrs["bin_length"],
        },
    )

    return spsdl


def _validate_method(
    method: Union[str, Dict[str, Union[float, int]]],
) -> Tuple[str, Optional[Union[float, int]]]:
    """
    Validates the 'method' parameter and returns the method name and its argument (if any)
    for an xarray.core.groupby.DataArrayGroupBy method.

    Parameters
    ----------
    method : str or dict
        The aggregation method to validate. It can be either:
          - A string representing one of the supported methods without additional arguments,
            e.g., 'mean', 'sum'.
          - A dictionary with a single key-value pair where the key is the method name and
            the value is its argument, e.g., {'quantile': 0.25}.

        Supported methods are:
          - 'all'
          - 'any'
          - 'assign_coords' (requires coordinate argument)
          - 'count'
          - 'cumprod'
          - 'fillna'
          - 'first'
          - 'last'
          - 'map' (requires custom function argument)
          - 'max'
          - 'mean'
          - 'median'
          - 'min'
          - 'prod'
          - 'quantile' (requires a quantile between 0 and 1)
          - 'reduce' (requires custom function argument)
          - 'std'
          - 'sum'
          - 'var'
          - 'where' (requires condition argument)

    Returns
    -------
    method_name : str
        The validated method name in lowercase.
    method_arg : float, int, or None
        The argument associated with the method, if applicableotherwise, None.

    Raises
    ------
    ValueError
        - If the method name is not supported.
        - If the 'quantile' method is provided without an argument or with an invalid argument.
        - If the 'method' dictionary does not contain exactly one key-value pair.
        - If 'method' is of an unsupported type.
    TypeError
        - If the key in the 'method' dictionary is not a string.

    Examples
    --------
    >>> _validate_method('mean')
    ('mean', None)

    >>> _validate_method({'quantile': 0.75})
    ('quantile', 0.75)

    >>> _validate_method('quantile')
    ValueError: The 'quantile' method must be provided as a dictionary with the quantile value,
        e.g., {'quantile': 0.25}.

    >>> _validate_method({'quantile': 1.5})
    ValueError: The 'quantile' method must have a float between 0 and 1 as an argument.

    >>> _validate_method({'unsupported_method': None})
    ValueError: Method 'unsupported_method' is not supported.
        Supported methods are:
        ['median', 'mean', 'min', 'max', 'sum', 'quantile', 'std', 'var', 'count']
    """

    allowed_methods = [
        "all",
        "any",
        "assign_coords",
        "count",
        "cumsum",
        "fillna",
        "first",
        "last",
        "map",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "reduce",
        "sum",
        "std",
        "sum",
        "var",
        "where",
    ]
    if not isinstance(method, (str, dict)):
        raise TypeError("'method' must be a string or a dictionary.")
    if isinstance(method, str):
        method_name = method.lower()
        if method_name not in allowed_methods:
            raise ValueError(
                f"Method '{method}' is not supported. Supported methods are: {allowed_methods}"
            )
        if method_name == "quantile":
            raise ValueError(
                "The 'quantile' method must be provided as a dictionary with "
                "the quantile value, e.g., {'quantile': 0.25}."
            )
        method_arg = None
    elif isinstance(method, dict):
        if len(method) != 1:
            raise ValueError(
                "'method' dictionary must contain exactly one key-value pair."
            )
        method_name, method_arg = list(method.items())[0]
        if not isinstance(method_name, str):
            raise TypeError("Key in 'method' dictionary must be a string.")
        method_name = method_name.lower()
        if method_name not in allowed_methods:
            raise ValueError(
                f"Method '{method_name}' is not supported. Supported methods are: {allowed_methods}"
            )
        if method_name == "quantile":
            if not isinstance(method_arg, (float, int)) or not 0 <= method_arg <= 1:
                raise ValueError(
                    "The 'quantile' method must have a float between 0 and 1 as an argument."
                )
    else:
        raise ValueError(
            f"Unsupported method type: {type(method)}. Must be a string or dictionary."
        )
    return method_name, method_arg


def band_aggregate(
    spsdl: xr.DataArray,
    octave: Tuple[int, int] = None,
    fmin: int = 10,
    fmax: int = 100000,
    method: Union[str, Dict[str, Union[float, int]]] = "median",
) -> xr.DataArray:
    """
    Deprecated. Reorganizes spectral density level frequency tensor into
    fractional octave bands and applies a function to them.

    Parameters
    ----------
    spsdl: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density level in dB rel 1 uPa^2/Hz
    octave: [int, int]
        Octave and octave base to subdivide spectral density level by. Set to
        octave base to 2 for the true octave band; set to base 10 for
        the decidecade octave band.
        Default = [3, 2] (true third octave)
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz
    method: str or dict
        Method to run on the binned data. Can be a string (e.g., "median") or a dict
        where the key is the method and the value is its argument (e.g., {"quantile": 0.25}).
        Options: [median, mean, min, max, sum, quantile, std, var, count]

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Frequency band-averaged sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time and frequency
    """
    warnings.warn(
        "The 'band_aggregate' function is deprecated and will be removed in a future release. "
        "Please use one of the following alternatives instead to convert the SPSD to the "
        "appropriate band before calculating the SPSDL using "
        "'sound_pressure_spectral_density_level':\n"
        "- For third octaves, use 'mhkit.acoustics.convert_to_third_octave' function.\n"
        "- For decidecades, use 'mhkit.acoustics.convert_to_decidecade' function.\n"
        "- For millidecades, use 'mhkit.acoustics.convert_to_millidecade' function.\n"
        "- For custom band aggregation, use 'mhkit.acoustics.convert_to_custom_bands.'",
        DeprecationWarning,
        stacklevel=2,
    )

    # Type checks
    if not isinstance(spsdl, xr.DataArray):
        raise TypeError("'spsdl' must be an xarray.DataArray.")
    if octave is None:
        octave = [3, 2]
    if not isinstance(octave, list) and not isinstance(octave, tuple):
        raise TypeError("'octave' must be a list or tuple of two integers.")
    for val in octave:
        if not isinstance(val, int) or (val <= 0):
            raise TypeError("'octave' must contain positive integers.")
    _check_numeric(fmin, "fmin")
    _check_numeric(fmax, "fmax")
    if fmax <= fmin:  # also checks that fmax is positive
        raise ValueError("'fmax' must be greater than 'fmin'.")

    # Value checks
    if ("freq" not in spsdl.dims) or ("time" not in spsdl.dims):
        raise ValueError("'spsdl' must have 'time' and 'freq' as dimensions.")

    # Validate method and get method_name and method_arg
    method_name, method_arg = _validate_method(method)

    # Check fmax
    fn = spsdl["freq"].max().values
    fmax = _fmax_warning(fn, fmax)

    octave_bins, band = create_frequency_bands(octave[0], octave[1], fmin, fmax)

    # Use xarray binning methods
    spsdl_group = spsdl.groupby_bins("freq", octave_bins, labels=band["center_freq"])

    # Handle method being a string or a dict
    if isinstance(method, str):
        func = getattr(spsdl_group, method.lower())
        out = func()
    else:
        method_name, method_arg = list(method.items())[0]
        func = getattr(spsdl_group, method_name.lower())
        if isinstance(method_arg, (list, tuple)):
            out = func(*method_arg)
        else:
            out = func(method_arg)

    # Update attributes
    out.attrs["units"] = spsdl.units

    return out


def time_aggregate(
    spsdl: xr.DataArray,
    window: int = 60,
    method: Union[str, Dict[str, Union[float, int]]] = "median",
) -> xr.DataArray:
    """
    Deprecated. Reorganizes spectral density level frequency tensor into
    time windows and applies a function to them.

    If the window length is equivalent to the size of spsdl["time"],
    this function is equivalent to spsdl.<method>("time")

    Parameters
    ----------
    spsdl: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density level in dB rel 1 uPa^2/Hz
    window: int
        Time in seconds to subdivide spectral density level into. Default: 60 s.
    method: str or dict
        Method to run on the binned data. Can be a string (e.g., "median") or a dict
        where the key is the method and the value is its argument (e.g., {"quantile": 0.25}).
        Options: [median, mean, min, max, sum, quantile, std, var, count]

    Returns
    -------
    out: xarray.DataArray (time_bins, freq)
        Time-averaged sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time and frequency
    """
    warnings.warn(
        "The 'time_aggregate' function is deprecated and will be removed in a future release. "
        "Please use one of the following alternatives instead to convert the SPSD to the "
        "appropriate time-aggregated form before calculating the SPSDL using "
        "'sound_pressure_spectral_density_level':\n"
        "- For time-averaged SPSDLs, use the 'mhkit.acoustics.time_average' function.\n"
        "- For time-summed SPSDLs, use the 'mhkit.acoustics.time_summation' function.\n"
        "If you are using this function for a different purpose, please reach out to the MHKiT "
        "developers to discuss how we can support your use case with a more specific function.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Type checks
    if not isinstance(spsdl, xr.DataArray):
        raise TypeError("'spsdl' must be an xarray.DataArray.")
    if not isinstance(window, int):
        raise TypeError("'window' must be an integer.")
    if not isinstance(method, (str, dict)):
        raise TypeError("'method' must be a string or dictionary.")
    if "time" not in spsdl.dims:
        raise ValueError("'spsdl' must have 'time' dimension.")

    # Value checks
    if window <= 0:
        raise ValueError("'window' must be a positive integer.")

    # Ensure 'time' coordinate is of datetime64 dtype
    if not np.issubdtype(spsdl["time"].dtype, np.datetime64):
        raise TypeError("'spsdl['time']' must be of dtype 'datetime64'.")

    # Validate method and get method_name and method_arg
    method_name, method_arg = _validate_method(method)

    window = np.timedelta64(window, "s")
    time_bins_lower = np.arange(
        spsdl["time"][0].values, spsdl["time"][-1].values, window
    )
    time_bins_upper = time_bins_lower + window
    time_bins = np.append(time_bins_lower, time_bins_upper[-1])
    center_time = epoch2dt64(
        0.5 * (dt642epoch(time_bins_lower) + dt642epoch(time_bins_upper))
    )

    # Use xarray binning methods
    spsdl_group = spsdl.groupby_bins("time", time_bins, labels=center_time)

    # Handle method being a string or a dict
    if isinstance(method, str):
        func = getattr(spsdl_group, method.lower())
        out = func()
    else:
        method_name, method_arg = list(method.items())[0]
        func = getattr(spsdl_group, method_name.lower())
        if isinstance(method_arg, (list, tuple)):
            out = func(*method_arg)
        else:
            out = func(method_arg)

    # Update attributes
    out.attrs["units"] = spsdl.units

    # Remove 'quantile' coordinate if present
    if method == "quantile":
        out = out.drop_vars("quantile")

    return out


def time_average(spsdl, window):
    """
    Reorganizes spectral density level frequency tensor into time windows and takes the
    spectral average of all of the inputs along the time dimension. This is effectively
    time-averaging the original SPSDs.
    Note: 'window' must be larger than the original 'bin_length' of the SPSD

    Parameters
    ----------
    spsdl: xarray.DataArray
        Sound pressure spectral density level with dimensions (time, freq)
    window: int
        Time in seconds to group spectral density level into.

    Returns
    -------
    xarray.DataArray
        Time-averaged sound pressure spectral density level [dB re 1 uPa^2/Hz] indexed
        by time and frequency
    """

    def spectral_average(x):
        # Convert value in decibels to absolute magnitude, still relevant to original units
        magnitude = 10 ** (x / 10)
        # Sum energy in each time bin
        summed_magnitude = magnitude.sum("time")
        # Take average
        average_magnitude = summed_magnitude / magnitude["time"].size
        # Convert back to decibels
        result = 10 * np.log10(average_magnitude)

        return result

    return time_aggregate(spsdl, window, method={"map": spectral_average})


def time_summation(spsdl, window):
    """
    Reorganizes spectral density level frequency tensor into time windows and takes the
    spectral sum of each window. This is the equivalent of recalculating the SPSD using
    `mhkit.acoustics.sound_presssure_spectral_density` with 'bin_length=window' instead
    of the original 'bin_length'.
    Note: 'window' must be larger than the original 'bin_length' of the SPSD

    Parameters
    ----------
    spsdl: xarray.DataArray
        Sound pressure spectral density level with dimensions (time, freq)
    window: int
        Time in seconds to group spectral density level into.

    Returns
    -------
    xarray.DataArray
        Time-summed sound pressure spectral density level [dB re 1 uPa^2/Hz] indexed
        by time and frequency
    """

    def spectral_sum(x):
        # Convert value in decibels to absolute magnitude, still relevant to original units
        magnitude = 10 ** (x / 10)
        # Sum energy in each time bin
        summed_magnitude = magnitude.sum("time")
        # Convert back to decibels
        result = 10 * np.log10(summed_magnitude)

        return result

    return time_aggregate(spsdl, window, method={"map": spectral_sum})
