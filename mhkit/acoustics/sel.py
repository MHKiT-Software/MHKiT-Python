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

from .spl import _argument_check


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
    weighting_func: xarray.DataArray (freq)
        Auditory weighting function [unitless] indexed by frequency
    exposure_func: xarray.DataArray (freq)
        Log-transformed auditory exposure function [dB] indexed by frequency
    """

    if group.lower() not in [
        "lf",
        "hf",
        "vhf",
        "pw",
        "ow",
    ]:
        raise ValueError("Group must be one of: LF, HF, VHF, PW, OW")

    group_params = {
        "lf": {"a": 0.99, "b": 5, "f1": 0.168, "f2": 26.6, "c": 0.12, "k": 177},
        "hf": {"a": 1.55, "b": 5, "f1": 1.73, "f2": 129, "c": 0.32, "k": 181},
        "vhf": {"a": 2.23, "b": 5, "f1": 5.93, "f2": 186, "c": 0.91, "k": 160},
        "pw": {"a": 1.63, "b": 5, "f1": 0.81, "f2": 68.3, "c": 0.29, "k": 175},
        "ow": {"a": 1.58, "b": 5, "f1": 2.53, "f2": 43.8, "c": 1.37, "k": 178},
    }

    a, b, f1, f2, c, k = group_params[group.lower()].values()

    frequency = frequency / 1000  # Convert to kHz
    ratio_a = frequency / f1
    ratio_b = frequency / f2
    band_filter = ratio_a ** (2 * a) / (
        ((1 + ratio_a**2) ** a) * ((1 + ratio_b**2) ** b)
    )

    weighting_func = c + 10 * np.log10(band_filter)  # dB
    exposure_func = k - 10 * np.log10(band_filter)  # dB

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

    # Argument checks
    fmax = _argument_check(spsd, fmin, fmax)

    if group is not None:
        w, _ = nmfs_auditory_weighting(spsd["freq"], group)
        # convert from dB back to unitless
        w = 10 ** (w / 10)
        long_name = "Weighted Sound Exposure Level"
    else:
        w = xr.ones_like(spsd["freq"])
        long_name = "Sound Exposure Level"

    # Reference value of sound pressure
    reference = 1e-12 * 1  # Pa^2 s, = 1 uPa^2 s

    # Mean square sound pressure in a specified frequency band
    # from weighted mean square values
    band = spsd.sel(freq=slice(fmin, fmax))
    w = w.sel(freq=slice(fmin, fmax))
    exposure = np.trapezoid(band * w, band["freq"])

    # Sound exposure level (L_{E,p}) = (L_{p,rms} + 10log10(t))
    sel = 10 * np.log10(exposure / reference) + 10 * np.log10(
        spsd.attrs["nfft"] / spsd.attrs["fs"]  # n_points / (n_points/s)
    )

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
