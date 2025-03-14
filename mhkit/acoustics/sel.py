"""
This module contains key functions related to calculating sound exposure levels
from sound pressure data.

1. **Sound Exposure Level Calculation**:

   - `nmfs_auditory_weighting`: Computes the auditory weighting and exposure functions
     for marine mammals based on the National Marine Fisheries Service (NMFS) guidelines.
   - `sound_exposure_level`: Computes the sound exposure level from within a
     specified time range.
"""

import numpy as np
import xarray as xr

from .analysis import _fmax_warning


def nmfs_auditory_weighting(frequency, group):
    """
    Calculates the auditory weighting and exposure functions for marine mammals
    based on the National Marine Fisheries Service (NMFS) guidelines.

    The weighting function is applied to sound exposure level to determine the
    auditory impact on marine mammals. The exposure function is the inverse of the
    weighting function and illustrates how the weighting function relates to marine
    mammal hearing thresholds.
    Both function are returned in their log10-transform, in units of dB. To transform
    back to linear units, use 10**(weighting_func/10).

    https://www.fisheries.noaa.gov/national/marine-mammal-protection/marine-mammal-acoustic-technical-guidance-other-acoustic-tools

    Parameters
    ----------
    frequency: xarray.DataArray (freq)
        Frequency vector in [Hz].
    group: str
        Marine mammal group for which the auditory weighting function is applied.
        Options: 'LF' (low frequency cetaceans), 'HF' (high frequency cetaceans),
        'VHF' (very high frequency cetaceans), 'PW' (phocid pinnepeds),
        'OW' (otariid pinnepeds)

    Returns
    -------
    weighting_func: float
        Auditory weighting function [unitless] indexed by frequency
    exposure_func: float
        Log-transformed auditory exposure function [dB] indexed by frequency
    """

    if group.lower() == "lf":
        # Low-frequency cetaceans
        a = 0.99
        b = 5
        f1 = 0.168  # kHz
        f2 = 26.6  # kHz
        C = 0.12  # dB
        K = 177  # dB
    elif group.lower() == "hf":
        # High-frequency cetaceans
        a = 1.55
        b = 5
        f1 = 1.73
        f2 = 129
        C = 0.32
        K = 181
    elif group.lower() == "vhf":
        # Very high-frequency cetaceans
        a = 2.23
        b = 5
        f1 = 5.93
        f2 = 186
        C = 0.91
        K = 160
    elif group.lower() == "pw":
        # Phocid pinnepeds
        a = 1.63
        b = 5
        f1 = 0.81
        f2 = 68.3
        C = 0.29
        K = 175
    elif group.lower() == "ow":
        # Otariid pinnepeds
        a = 1.58
        b = 5
        f1 = 2.53
        f2 = 43.8
        C = 1.37
        K = 178
    else:
        raise ValueError("Group must be LF, MF, HF, PW, or OW")

    A = frequency / f1
    B = frequency / f2
    band_filter = A ** (2 * a) / (((1 + A**2) ** a) * ((1 + B**2) ** b))

    weighting_func = C + 10 * np.log10(band_filter)  # dB
    exposure_func = K - 10 * np.log10(band_filter)  # dB

    return weighting_func, exposure_func


def sound_exposure_level(
    spsd: xr.DataArray, group: str = None, fmin: int = 10, fmax: int = 100000
) -> xr.DataArray:
    """
    Calculates the sound exposure level (SEL) across a specified frequency band
    from the sound pressure spectral density (SPSD). If a marine mammal group is
    provided, the resulting SEL is weighted according to the U.S. National Marine
    Fisheries Service (NMFS) guidelines.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Sound pressure spectral density in [Pa^2/Hz] with a bin length
        equal to the time over which sound exposure should be computed.
    group: str
        Marine mammal group for which the auditory weighting function is applied.
        Options: 'LF' (low frequency cetaceans), 'HF' (high frequency cetaceans),
        'VHF' (very high frequency cetaceans), 'PW' (phocid pinnepeds),
        'OW' (otariid pinnepeds). Default: None
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone).
        Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default:
        100000 Hz

    Returns
    -------
    sel: xarray.DataArray (time)
        Sound exposure level [dB re 1 uPa^2 s] indexed by time
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
    if "mean square" in spsd.attrs["long_name"].lower():
        raise AssertionError(
            "'spsd' should not be the mean square sound pressure spectral density."
            "Please set `rms=False` in `mhkit.acoustics.sound_pressure_spectral_density`."
        )

    # Value checks
    if fmin <= 0:
        raise ValueError("'fmin' must be a positive integer.")
    if fmax <= fmin:
        raise ValueError("'fmax' must be greater than 'fmin'.")

    # Check fmax
    fn = spsd.attrs["fs"] // 2
    fmax = _fmax_warning(fn, fmax)

    if group is not None:
        W, _ = nmfs_auditory_weighting(spsd["freq"], group)
        # convert from dB back to unitless
        W = 10 ** (W / 10)
        long_name = "Weighted Sound Exposure Level"
    else:
        W = 1
        long_name = "Sound Exposure Level"

    # Reference value of sound pressure
    reference = 1e-12  # Pa^2, = 1 uPa^2

    # Sound exposure [Pa^2 s]
    exposure = np.trapz(
        (spsd * W).sel(freq=slice(fmin, fmax)),
        spsd["freq"].sel(freq=slice(fmin, fmax)),
    )

    # Sound exposure level
    sel = 10 * np.log10(exposure / reference)

    out = xr.DataArray(
        sel.astype(np.float32),
        coords={"time": spsd["time"]},
        attrs={
            "units": "dB re 1 uPa^2 s",
            "long_name": long_name,
            "weighting_group": group,
            "integration_time": spsd.attrs["nbin"],
            "freq_band_min": fmin,
            "freq_band_max": fmax,
        },
    )

    return out
