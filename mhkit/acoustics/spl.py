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

from .analysis import (
    _check_numeric,
    _fmax_warning,
    _get_band_table,
    _band_power_spectral_density_v3,
)


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
    _check_numeric(fmin, "fmin")
    _check_numeric(fmax, "fmax")

    # Ensure 'freq' and 'time' dimensions are present
    if ("freq" not in spsd.dims) or ("time_psd" not in spsd.dims):
        raise ValueError("'spsd' must have 'time_psd' and 'freq' as dimensions.")

    # Check that 'fs' (sampling frequency) is available in attributes
    if "fs" not in spsd.attrs:
        raise ValueError(
            "'spsd' must have 'fs' (sampling frequency) in its attributes."
        )
    if "n_fft" not in spsd.attrs:
        raise ValueError(
            "'spsd' must have 'n_fft' (number of points in each FFT) in its attributes."
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
    spsd: xarray.DataArray (time_psd, freq)
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
    pressure_squared = np.trapezoid(band, freqs)

    # Mean square sound pressure level
    mspl = 10 * np.log10(pressure_squared / reference)

    out = xr.DataArray(
        mspl.astype(np.float32),
        coords={"time_psd": spsd["time_psd"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Sound Pressure Level",
            "time_resolution": spsd.attrs["bin_length"],
            "freq_band_min": fmin,
            "freq_band_max": fmax,
        },
    )

    return out


def _band_sound_pressure_level(spsd: xr.DataArray, octave: int, base: int):
    """
    Calculates band-averaged sound pressure levels from the
    mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time_psd, freq)
        Mean square sound pressure spectral density in [Pa^2/Hz]
    octave: int
        Octave subdivision (1 = full octave, 3 = third-octave, etc.)
    base: int
        Octave base subdivision (2 = true octave, 10 = decade octave, etc.)

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and frequency of specified bandwidth

    Notes
    -----
    Assumes constant spacing in FFT frequency vector.
    """

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2
    # Frequency vector
    freq = spsd["freq"].values

    # Get bands
    bands = _get_band_table(
        freq=freq,
        bands_per_division=octave,
        base=base,
        use_fft_res_at_bottom=False,
    )
    full_pts, partial_pts, weights = _band_power_spectral_density_v3(
        freq_fft=freq, freq_table=bands
    )

    input_spsd = spsd.values
    out_sp = np.zeros((input_spsd.shape[0], bands.shape[0]))

    # Integrate band-squared pressure by frequency; vectorised over the time (row) axis
    for j in range(bands.shape[0]):
        # Contribution from fully-contained FFT bins
        if len(full_pts[j]) > 0:
            out_sp[:, j] = np.trapezoid(
                input_spsd[:, full_pts[j]], freq[full_pts[j]], axis=1
            )

        # Contribution from partial FFT bins
        if len(partial_pts[j]) > 0:
            out_sp[:, j] += np.trapezoid(
                input_spsd[:, partial_pts[j]] * weights[j][np.newaxis, :],
                dx=freq[1] - freq[0],
                axis=1,
            )

    # Mean square sound pressure level in dB rel 1 uPa
    out_spl = xr.DataArray(
        10 * np.log10(out_sp / reference),
        coords={
            "time_psd": spsd["time_psd"],
            "freq_bins": bands[:, 1],
        },
        dims=["time_psd", "freq_bins"],
    )
    return out_spl


def third_octave_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in third octave bands directly
    from the mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time_psd, freq)
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

    # Argument checks
    fmax = _argument_check(spsd, fmin, fmax)

    octave = 3
    base = 2
    mspl = _band_sound_pressure_level(spsd, octave, base)
    mspl.attrs.update(
        {
            "units": "dB re 1 uPa",
            "long_name": "Third Octave Sound Pressure Level",
            "time_resolution": spsd.attrs["bin_length"],
        }
    )

    mspl = mspl.sel(freq_bins=slice(fmin, fmax))
    return mspl.astype(np.float32)


def decidecade_sound_pressure_level(
    spsd: xr.DataArray, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound pressure level in decidecade bands directly
    from the mean square sound pressure spectral density (SPSD).

    Parameters
    ----------
    spsd: xarray.DataArray (time_psd, freq)
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

    # Argument checks
    fmax = _argument_check(spsd, fmin, fmax)

    octave = 10
    base = 10
    mspl = _band_sound_pressure_level(spsd, octave, base)
    mspl.attrs.update(
        {
            "units": "dB re 1 uPa",
            "long_name": "Decidecade Sound Pressure Level",
            "time_resolution": spsd.attrs["bin_length"],
        }
    )

    mspl = mspl.sel(freq_bins=slice(fmin, fmax))
    return mspl.astype(np.float32)
