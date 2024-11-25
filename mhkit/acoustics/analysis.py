"""
This module contains key functions for passive acoustics analysis, designed to process
and analyze sound pressure data from .wav files in the frequency and time domains.
The functions herein build on each other, with a structured flow that facilitates the
calculation of sound pressure levels, spectral densities, and banded averages, based on
input audio data.

The following functionality is provided:

1. **Frequency Validation and Warning**:

   - `_fmax_warning`: Ensures specified maximum frequency does not exceed the Nyquist frequency,
     adjusting if necessary to avoid aliasing.

2. **Shallow Water Cutoff Frequency**:

   - `minimum_frequency`: Calculates the minimum frequency cutoff based on water depth and the
     speed of sound in water and seabed materials.

3. **Spectral Density Calculations**:

   - `sound_pressure_spectral_density`: Computes the mean square sound pressure spectral density
     using FFT binning with Hanning windowing and 50% overlap.

4. **Calibration**:

   - `apply_calibration`: Applies calibration adjustments to the spectral density data using
     a sensitivity curve, filling missing values as specified.

5. **Spectral Density Level Calculation**:

   - `sound_pressure_spectral_density_level`: Converts mean square spectral density values to
     sound pressure spectral density levels in dB.

6. **Spectral Density Aggregation**:

   - `band_aggregate`: Aggregates spectral density data into fractional octave bands using
     specified statistical methods (e.g., median, mean).

   - `time_aggregate`: Aggregates spectral density data into specified time windows using
     similar statistical methods.

7. **Sound Pressure Level Calculation**:

   - `sound_pressure_level`: Computes the overall sound pressure level within a frequency band
     from mean square spectral density.

8. **Frequency-Banded Sound Pressure Level**:

   - `_band_sound_pressure_level`: Helper function for calculating sound pressure levels
     over specified frequency bandwidths.

   - `third_octave_sound_pressure_level` and `decidecade_sound_pressure_level`:
     Compute sound pressure levels across third-octave and decidecade bands, respectively.
"""

from typing import Union, Dict, Tuple, Optional
import warnings
import numpy as np
import xarray as xr

from mhkit.dolfyn import VelBinner
from mhkit.dolfyn.time import epoch2dt64, dt642epoch


def _fmax_warning(
    fn: Union[int, float, np.ndarray], fmax: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    """
    Checks that the maximum frequency limit isn't greater than the Nyquist frequency.

    Parameters
    ----------
    fn: int, float, or numpy.ndarray
        The Nyquist frequency in Hz.
    fmax: float
        The maximum frequency limit in Hz.

    Returns
    -------
    fmax: float
        The adjusted maximum frequency limit, ensuring it does not exceed the Nyquist frequency.
    """

    if not isinstance(fn, (int, float, np.ndarray)):
        raise TypeError("'fn' must be a numeric type (int or float).")
    if not isinstance(fmax, (int, float, np.ndarray)):
        raise TypeError("'fmax' must be a numeric type (int or float).")

    if fmax > fn:
        warnings.warn(
            f"`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            f"fmax = {fn}"
        )
        fmax = fn

    return fmax


def minimum_frequency(
    water_depth: Union[int, float, np.ndarray, list],
    c: Union[int, float] = 1500,
    c_seabed: Union[int, float] = 1700,
) -> Union[float, np.ndarray]:
    """
    Estimate the shallow water cutoff frequency based on the speed of
    sound in the water column and the speed of sound in the seabed
    material (generally ranges from 1450 - 1800 m/s)

    Parameters
    ----------
    water_depth: int, float or array-like
        Depth of the water column in meters.
    c: float, optional
        Speed of sound in the water column in meters per second. Default is 1500 m/s.
    c_seabed: float, optional
        Speed of sound in the seabed material in meters per second. Default is 1700 m/s.

    Returns
    -------
    f_min: float or numpy.ndarray
        The minimum cutoff frequency in Hz.

    Reference
    ---------
    Jennings 2011 - Computational Ocean Acoustics, 2nd ed.
    """

    # Convert water_depth to a NumPy array for vectorized operations
    water_depth = np.asarray(water_depth)

    # Validate water_depth
    if not np.issubdtype(water_depth.dtype, np.number):
        raise TypeError("'water_depth' must be a numeric type or array of numerics.")

    if not isinstance(c, (int, float)):
        raise TypeError("'c' must be a numeric type (int or float).")
    if not isinstance(c_seabed, (int, float)):
        raise TypeError("'c_seabed' must be a numeric type (int or float).")

    if np.any(water_depth <= 0):
        raise ValueError("All elements of 'water_depth' must be positive numbers.")
    if c <= 0:
        raise ValueError("'c' must be a positive number.")
    if c_seabed <= 0:
        raise ValueError("'c_seabed' must be a positive number.")
    if c_seabed <= c:
        raise ValueError("'c_seabed' must be greater than 'c'.")

    fmin = c / (4 * water_depth * np.sqrt(1 - (c / c_seabed) ** 2))

    return fmin


def sound_pressure_spectral_density(
    pressure: xr.DataArray, fs: Union[int, float], bin_length: Union[int, float] = 1
) -> xr.DataArray:
    """
    Calculates the mean square sound pressure spectral density from audio
    samples split into FFTs with a specified bin length in seconds, using Hanning
    windowing with 50% overlap. The amplitude of the PSD is adjusted
    according to Parseval's theorem.

    Parameters
    ----------
    pressure: xarray.DataArray (time)
        Sound pressure in [Pa] or voltage [V]
    fs: int or float
        Data collection sampling rate [Hz]
    bin_length: int or float
        Length of time in seconds to create FFTs. Default: 1.

    Returns
    -------
    spsd: xarray.DataArray (time, freq)
        Spectral density [Pa^2/Hz] indexed by time and frequency
    """

    # Type checks
    if not isinstance(pressure, xr.DataArray):
        raise TypeError("'pressure' must be an xarray.DataArray.")
    if not isinstance(fs, (int, float)):
        raise TypeError("'fs' must be a numeric type (int or float).")
    if not isinstance(bin_length, (int, float)):
        raise TypeError("'bin_length' must be a numeric type (int or float).")

    # Ensure that 'pressure' has a 'time' coordinate
    if "time" not in pressure.dims:
        raise ValueError("'pressure' must be indexed by 'time' dimension.")

    # window length of each time series
    nbin = bin_length * fs

    # Use dolfyn PSD
    binner = VelBinner(n_bin=nbin, fs=fs, n_fft=nbin)
    # Always 50% overlap if numbers reshape perfectly
    # Mean square sound pressure
    psd = binner.power_spectral_density(pressure, freq_units="Hz")
    samples = binner.reshape(pressure.values) - binner.mean(pressure.values)[:, None]
    # Power in time domain
    t_power = np.sum(samples**2, axis=1) / nbin
    # Power in frequency domain
    f_power = psd.sum("freq") * (fs / nbin)
    # Adjust the amplitude of PSD according to Parseval's theorem
    psd_adj = psd * t_power[:, None] / f_power

    out = xr.DataArray(
        psd_adj,
        coords={"time": psd_adj["time"], "freq": psd_adj["freq"]},
        attrs={
            "units": pressure.units + "^2/Hz",
            "long_name": "Mean Square Sound Pressure Spectral Density",
            "fs": fs,
            "nbin": str(bin_length) + " s",
            "overlap": "50%",
            "nfft": nbin,
        },
    )

    return out


def apply_calibration(
    spsd: xr.DataArray,
    sensitivity_curve: xr.DataArray,
    fill_value: Union[float, int, np.ndarray],
) -> xr.DataArray:
    """
    Applies custom calibration to spectral density values.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in V^2/Hz.
    sensitivity_curve: xarray.DataArray (freq)
        Calibrated sensitivity curve in units of dB rel 1 V^2/uPa^2.
        First column should be frequency, second column should be calibration values.
    fill_value: float or int
        Value with which to fill missing values from the calibration curve,
        in units of dB rel 1 V^2/uPa^2.

    Returns
    -------
    spsd_calibrated: xarray.DataArray (time, freq)
        Spectral density in Pa^2/Hz, indexed by time and frequency.
    """

    if not isinstance(spsd, xr.DataArray):
        raise TypeError("'spsd' must be an xarray.DataArray.")
    if not isinstance(sensitivity_curve, xr.DataArray):
        raise TypeError("'sensitivity_curve' must be an xarray.DataArray.")
    if not isinstance(fill_value, (int, float, np.ndarray)):
        raise TypeError("'fill_value' must be a numeric type (int or float).")

    # Ensure 'freq' dimension exists in 'spsd'
    if "freq" not in spsd.dims:
        if len(spsd.dims) > 1:
            # Issue a warning and assign the 2nd dimension as 'freq'
            warnings.warn(
                f"'spsd' does not have 'freq' as a dimension and has multiple dimensions. "
                f"Using the second dimension '{spsd.dims[1]}' as 'freq'."
            )
        # Assign the 2nd dimension as 'freq'
        spsd = spsd.rename({spsd.dims[1]: "freq"})

    # Ensure 'freq' dimension exists in 'sensitivity_curve'
    if "freq" not in sensitivity_curve.dims:
        if len(sensitivity_curve.dims) > 1:
            # Issue a warning and assign the 1st dimension as 'freq'
            warnings.warn(
                f"'sensitivity_curve' does not have 'freq' as a dimension \
                      and has multiple dimensions. "
                f"Using the first dimension '{sensitivity_curve.dims[0]}' as 'freq'."
            )
        # Assign the 0th dimension as 'freq'
        sensitivity_curve = sensitivity_curve.rename(
            {sensitivity_curve.dims[0]: "freq"}
        )

    # Create a copy of spsd to avoid in-place modification
    spsd_calibrated = spsd.copy(deep=True)
    attrs = spsd.attrs  # recover attrs

    # Read calibration curve
    freq = sensitivity_curve.dims[0]
    # Interpolate calibration curve to desired value
    calibration = sensitivity_curve.interp(
        {freq: spsd_calibrated["freq"]}, method="linear"
    )
    # Fill missing with provided value
    calibration = calibration.fillna(fill_value)

    # Subtract from sound pressure spectral density
    sensitivity_ratio = 10 ** (calibration / 10)  # V^2/uPa^2
    spsd_calibrated = spsd_calibrated / sensitivity_ratio / 1e12  # Pa^2/Hz
    attrs.update(
        {"long_name": "Calibrated Sound Pressure Spectral Density", "units": "Pa^2/Hz"}
    )
    spsd_calibrated.attrs = attrs

    return spsd_calibrated


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
        coords={"time": spsd["time"], "freq": spsd["freq"]},
        attrs={
            "units": "dB re 1 uPa^2/Hz",
            "long_name": "Sound Pressure Spectral Density Level",
        },
    )

    return spsdl


def _validate_method(
    method: Union[str, Dict[str, Union[float, int]]]
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
        The argument associated with the method, if applicable; otherwise, None.

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
    octave: int = 3,
    fmin: int = 10,
    fmax: int = 100000,
    method: Union[str, Dict[str, Union[float, int]]] = "median",
) -> xr.DataArray:
    """
    Reorganizes spectral density level frequency tensor into
    fractional octave bands and applies a function to them.

    Parameters
    ----------
    spsdl: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density level in dB rel 1 uPa^2/Hz
    octave: int
        Octave to subdivide spectral density level by. Default = 3 (third octave)
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

    # Type checks
    if not isinstance(spsdl, xr.DataArray):
        raise TypeError("'spsdl' must be an xarray.DataArray.")
    if not isinstance(octave, int) or (octave <= 0):
        raise TypeError("'octave' must be a positive integer.")
    if not isinstance(fmin, int) or (fmin <= 0):
        raise TypeError("'fmin' must be a positive integer.")
    if not isinstance(fmax, int) or (fmin <= 0):
        raise TypeError("'fmax' must be a positive integer.")
    if fmax <= fmin:
        raise ValueError("'fmax' must be greater than 'fmin'.")
    if not isinstance(method, (str, dict)):
        raise TypeError("'method' must be a string or a dictionary.")

    # Value checks
    if ("freq" not in spsdl.dims) or ("time" not in spsdl.dims):
        raise ValueError("'spsdl' must have 'time' and 'freq' as dimensions.")

    # Validate method and get method_name and method_arg
    method_name, method_arg = _validate_method(method)

    # Check fmax
    fn = spsdl["freq"].max().values
    fmax = _fmax_warning(fn, fmax)

    bandwidth = 2 ** (1 / octave)
    half_bandwidth = 2 ** (1 / (octave * 2))

    band = {}
    band["center_freq"] = 10 ** np.arange(
        np.log10(fmin),
        np.log10(fmax * bandwidth),
        step=np.log10(bandwidth),
    )
    band["lower_limit"] = band["center_freq"] / half_bandwidth
    band["upper_limit"] = band["center_freq"] * half_bandwidth
    octave_bins = np.append(band["lower_limit"], band["upper_limit"][-1])

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

    # Remove 'quantile' coordinate if present
    if method == "quantile":
        out = out.drop_vars("quantile")

    return out


def time_aggregate(
    spsdl: xr.DataArray,
    window: int = 60,
    method: Union[str, Dict[str, Union[float, int]]] = "median",
) -> xr.DataArray:
    """
    Reorganizes spectral density level frequency tensor into
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


def sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in a specified frequency band
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in [Pa^2/Hz]
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz

    Returns
    -------
    spl: xarray.DataArray (time)
        Sound pressure level [dB re 1 uPa] indexed by time
    """

    # Type checks
    if not isinstance(spsd, xr.DataArray):
        raise TypeError("'spsd' must be an xarray.DataArray.")
    if not isinstance(fmin, int):
        raise TypeError("'fmin' must be an integer.")
    if not isinstance(fmax, int):
        raise TypeError("'fmax' must be an integer.")

    # Ensure 'freq' and 'time' dimensions are present
    if ("freq" not in spsd.dims) or ("time" not in spsd.dims):
        raise ValueError("'spsd' must have 'time' and 'freq' as dimensions.")

    # Check that 'fs' (sampling frequency) is available in attributes
    if "fs" not in spsd.attrs:
        raise ValueError(
            "'spsd' must have 'fs' (sampling frequency) in its attributes."
        )

    # Value checks
    if fmin <= 0:
        raise ValueError("'fmin' must be a positive integer.")
    if fmax <= fmin:
        raise ValueError("'fmax' must be greater than 'fmin'.")

    # Check fmax
    fn = spsd.attrs["fs"] // 2
    fmax = _fmax_warning(fn, fmax)

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2

    # Mean square sound pressure in a specified frequency band from mean square values
    pressure_squared = np.trapz(
        spsd.sel(freq=slice(fmin, fmax)), spsd["freq"].sel(freq=slice(fmin, fmax))
    )

    # Mean square sound pressure level
    mspl = 10 * np.log10(pressure_squared / reference)

    out = xr.DataArray(
        mspl.astype(np.float32),
        coords={"time": spsd["time"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Sound Pressure Level",
            "freq_band_min": fmin,
            "freq_band_max": fmax,
        },
    )

    return out


def _band_sound_pressure_level(
    spsd: xr.DataArray,
    bandwidth: int,
    half_bandwidth: int,
    fmin: int = 10,
    fmax: int = 100000,
) -> xr.DataArray:
    """
    Calculates band-averaged sound pressure levels

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density.
    bandwidth : int or float
        Bandwidth to average over.
    half_bandwidth : int or float
        Half-bandwidth, used to set upper and lower bandwidth limits.
    fmin : int, optional
        Lower frequency band limit (lower limit of the hydrophone). Default is 10 Hz.
    fmax : int, optional
        Upper frequency band limit (Nyquist frequency). Default is 100,000 Hz.


    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and frequency of specified bandwidth
    """

    # Type checks
    if not isinstance(spsd, xr.DataArray):
        raise TypeError("'spsd' must be an xarray.DataArray.")
    if not isinstance(bandwidth, (int, float)):
        raise TypeError("'bandwidth' must be a numeric type (int or float).")
    if not isinstance(half_bandwidth, (int, float)):
        raise TypeError("'half_bandwidth' must be a numeric type (int or float).")
    if not isinstance(fmin, int):
        raise TypeError("'fmin' must be an integer.")
    if not isinstance(fmax, int):
        raise TypeError("'fmax' must be an integer.")

    # Ensure 'freq' and 'time' dimensions are present
    if "freq" not in spsd.dims or "time" not in spsd.dims:
        raise ValueError("'spsd' must have 'time' and 'freq' as dimensions.")

    # Check that 'fs' (sampling frequency) is available in attributes
    if "fs" not in spsd.attrs:
        raise ValueError(
            "'spsd' must have 'fs' (sampling frequency) in its attributes."
        )

    # Value checks
    if fmin <= 0:
        raise ValueError("'fmin' must be a positive integer.")
    if fmax <= fmin:
        raise ValueError("'fmax' must be greater than 'fmin'.")

    # Check fmax
    fn = spsd.attrs["fs"] // 2
    fmax = _fmax_warning(fn, fmax)

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2

    band = {}
    band["center_freq"] = 10 ** np.arange(
        np.log10(fmin),
        np.log10(fmax * bandwidth),
        step=np.log10(bandwidth),
    )
    band["lower_limit"] = band["center_freq"] / half_bandwidth
    band["upper_limit"] = band["center_freq"] * half_bandwidth
    octave_bins = np.append(band["lower_limit"], band["upper_limit"][-1])

    # Manual trapezoidal rule to get Pa^2
    pressure_squared = xr.DataArray(
        coords={"time": spsd["time"], "freq_bins": band["center_freq"]},
        dims=["time", "freq_bins"],
    )
    for i, key in enumerate(band["center_freq"]):
        band_min = octave_bins[i]
        band_max = octave_bins[i + 1]
        pressure_squared.loc[{"freq_bins": key}] = np.trapz(
            spsd.sel(freq=slice(band_min, band_max)),
            spsd["freq"].sel(freq=slice(band_min, band_max)),
        )

    # Mean square sound pressure level in dB rel 1 uPa
    mspl = 10 * np.log10(pressure_squared / reference)

    return mspl


def third_octave_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in third octave bands directly
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz

    Returns
    -------
    mspl: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and third octave bands
    """

    # Third octave bin frequencies
    bandwidth = 2 ** (1 / 3)
    half_bandwidth = 2 ** (1 / 6)

    mspl = _band_sound_pressure_level(spsd, bandwidth, half_bandwidth, fmin, fmax)
    mspl.attrs = {
        "units": "dB re 1 uPa",
        "long_name": "Third Octave Sound Pressure Level",
    }

    return mspl.astype(np.float32)


def decidecade_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in decidecade bands directly
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz

    Returns
    -------
    mspl : xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and decidecade bands
    """

    # Decidecade bin frequencies
    bandwidth = 2 ** (1 / 10)
    half_bandwidth = 2 ** (1 / 20)

    mspl = _band_sound_pressure_level(spsd, bandwidth, half_bandwidth, fmin, fmax)
    mspl.attrs = {
        "units": "dB re 1 uPa",
        "long_name": "Decidecade Sound Pressure Level",
    }

    return mspl.astype(np.float32)
