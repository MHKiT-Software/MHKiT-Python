"""
This module contains key functions related to calculating sound pressure levels
from sound pressure data.

1. **Sound Pressure Level Calculation**:

   - `sound_pressure_level`: Computes the overall sound pressure level within a frequency band
     from mean square spectral density.

2. **Frequency-Banded Sound Pressure Level**:

   - `_band_sound_pressure_level`: Helper function for calculating sound pressure levels
     over specified frequency bandwidths.

   - `third_octave_sound_pressure_level` and `decidecade_sound_pressure_level`:
     Compute sound pressure levels across third-octave and decidecade bands, respectively.
"""

import numpy as np
import xarray as xr

from .analysis import _fmax_warning, _create_frequency_bands


def _argument_check(spsd, fmin, fmax):
    """
    Validates input types, values, and dimensions for SPSD data and adjusts
    fmax to the Nyquist frequency if needed.

    Parameters
    ----------
    spsd : xarray.DataArray
        Spectral data with 'time' and 'freq' dimensions and a 'fs' attribute.
    fmin : int
        Minimum frequency (Hz), must be > 0.
    fmax : int
        Maximum frequency (Hz), must be > fmin.

    Returns
    -------
    fmax : int
        Frequency limited to below the Nyquist limit.
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
    if "nfft" not in spsd.attrs:
        raise ValueError(
            "'spsd' must have 'nfft' (sampling frequency) in its attributes."
        )

    # Value checks
    if fmin <= 0:
        raise ValueError("'fmin' must be a positive integer.")
    if fmax <= fmin:
        raise ValueError("'fmax' must be greater than 'fmin'.")

    # Check fmax
    fn = spsd.attrs["fs"] // 2
    fmax = _fmax_warning(fn, fmax)

    return fmax


def sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level (SPL) in a specified frequency band
    from the mean square sound pressure spectral density (SPSD).

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

    # Argument checks
    fmax = _argument_check(spsd, fmin, fmax)

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2

    # Mean square sound pressure in a specified frequency band from mean square values
    band = spsd.sel(freq=slice(fmin, fmax))
    freqs = band["freq"]
    pressure_squared = np.trapz(band, freqs)

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
    octave: int,
    base: int = 2,
    fmin: int = 10,
    fmax: int = 100000,
) -> xr.DataArray:
    """
    Calculates band-averaged sound pressure levels from the
    mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in [Pa^2/Hz]
    octave: int
        Octave subdivision (1 = full octave, 3 = third-octave, etc.)
    base: int
        Octave base subdivision (2 = true octave, 10 = decade octave, etc.)
    fmin : int, optional
        Lower frequency band limit (lower limit of the hydrophone).
        Default is 10 Hz.
    fmax : int, optional
        Upper frequency band limit (Nyquist frequency).
        Default is 100,000 Hz.

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and frequency of specified bandwidth
    """

    # Type checks
    if not isinstance(octave, int) or (octave <= 0):
        raise TypeError("'octave' must be a positive integer.")

    # Argument checks
    fmax = _argument_check(spsd, fmin, fmax)

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2

    _, band = _create_frequency_bands(octave, base, fmin, fmax)

    # Manual trapezoidal rule to get Pa^2
    pressure_squared = xr.DataArray(
        coords={"time": spsd["time"], "freq_bins": band["center_freq"]},
        dims=["time", "freq_bins"],
    )
    for i, key in enumerate(band["center_freq"]):
        # Min and max band limits
        band_range = [band["lower_limit"][i], band["upper_limit"][i]]

        # Integrate spectral density by frequency
        x = spsd["freq"].sel(freq=slice(*band_range))
        if len(x) < 2:
            # Interpolate between band frequencies if width is narrow
            bandwidth = band_range[1] / band_range[0]
            # Use smaller set of dataset to speed up interpolation
            spsd_slc = spsd.sel(
                freq=slice(
                    None,  # Only happens at low frequency
                    band_range[1] * bandwidth * 2,
                )
            )
            spsd_slc = spsd_slc.interp(freq=band_range)
            x = band_range
        else:
            spsd_slc = spsd.sel(freq=slice(*band_range))

        pressure_squared.loc[{"freq_bins": key}] = np.trapz(spsd_slc, x)

    # Mean square sound pressure level in dB rel 1 uPa
    mspl = 10 * np.log10(pressure_squared / reference)

    return mspl


def third_octave_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in third octave bands directly
    from the mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral in [Pa^2/Hz].
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone).
        Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency).
        Default: 100000 Hz

    Returns
    -------
    mspl: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and third octave bands
    """
    octave = 3
    base = 2
    mspl = _band_sound_pressure_level(spsd, octave, base, fmin, fmax)
    mspl.attrs.update(
        {
            "units": "dB re 1 uPa",
            "long_name": "Third Octave Sound Pressure Level",
        }
    )

    return mspl.astype(np.float32)


def decidecade_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in decidecade bands directly
    from the mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in [Pa^2/Hz].
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone).
        Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency).
        Default: 100000 Hz

    Returns
    -------
    mspl : xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and decidecade bands
    """

    octave = 10
    base = 10
    mspl = _band_sound_pressure_level(spsd, octave, base, fmin, fmax)
    mspl.attrs.update(
        {
            "units": "dB re 1 uPa",
            "long_name": "Decidecade Sound Pressure Level",
        }
    )

    return mspl.astype(np.float32)
