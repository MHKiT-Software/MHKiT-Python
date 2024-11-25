"""
This submodule provides essential plotting functions for visualizing passive acoustics 
data. The functions allow for customizable plotting of sound pressure spectral density 
levels across time and frequency dimensions.

Each plotting function leverages the flexibility of Matplotlib, allowing for passthrough
of Matplotlib keyword arguments via ``**kwargs``, making it easy to modify plot aspects such as
color, scale, and label formatting.

Key Functions
-------------
1. **plot_spectrogram**:

   - Generates a spectrogram plot from sound pressure spectral density level data, 
     with a logarithmic frequency scale by default for improved readability of acoustic data.

2. **plot_spectra**:

   - Produces a spectral density plot with a log-transformed x-axis, allowing for clear 
     visualization of spectral density across frequency bands.
"""

from typing import Tuple
import xarray as xr
import matplotlib.pyplot as plt

from .analysis import _fmax_warning


def plot_spectrogram(
    spsdl: xr.DataArray,
    fmin: int = 10,
    fmax: int = 100000,
    fig: plt.figure = None,
    ax: plt.Axes = None,
    **kwargs
) -> Tuple[plt.figure, plt.Axes]:
    """
    Plots the spectrogram of the sound pressure spectral density level.

    Parameters
    ----------
    spsdl: xarray DataArray (time, freq)
        Mean square sound pressure spectral density level in dB rel 1 uPa^2/Hz
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz
    fig: matplotlib.pyplot.figure
        Figure handle to plot on
    ax: matplotlib.pyplot.axis
        Figure axis containing plot objects
    kwargs: dict
        Dictionary of matplotlib function keyword arguments

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure handle of plot
    ax: matplotlib.pyplot.Axes
        Figure plot axis
    """

    if not isinstance(fmin, (int, float)) or not isinstance(fmax, (int, float)):
        raise TypeError("fmin and fmax must be numeric types.")
    if fmin >= fmax:
        raise ValueError("fmin must be less than fmax.")

    # Set dimension names
    # "time" or "time_bins" is always first
    time = spsdl.dims[0]
    # "freq" or "freq_bins" is always second
    freq = spsdl.dims[-1]

    # Check fmax
    fn = spsdl[freq].max().item()
    fmax = _fmax_warning(fn, fmax)
    # select frequency range
    spsdl = spsdl.sel({freq: slice(fmin, fmax)})

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"yscale": "log"})
        fig.subplots_adjust(left=0.1, right=0.95, top=0.97, bottom=0.11)
    h = ax.pcolormesh(
        spsdl[time].values,
        spsdl[freq].values,
        spsdl.transpose(freq, time),
        shading="nearest",
        **kwargs
    )
    fig.colorbar(h, ax=ax, label=getattr(spsdl, "units", None))
    ax.set(xlabel="Time", ylabel="Frequency [Hz]")

    return fig, ax


def plot_spectra(
    spsdl: xr.DataArray,
    fmin: int = 10,
    fmax: int = 100000,
    fig: plt.figure = None,
    ax: plt.Axes = None,
    **kwargs
) -> Tuple[plt.figure, plt.Axes]:
    """
    Plots spectral density. X axis is log-transformed.

    Parameters
    ----------
    spsdl: xarray DataArray (time, freq)
        Mean square sound pressure spectral density level in dB rel 1 uPa^2/Hz
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz
    fig: matplotlib.pyplot.figure
        Figure handle to plot on
    ax: matplotlib.pyplot.Axes
        Figure axis containing plot objects
    kwargs: dict
        Dictionary of matplotlib function keyword arguments

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure handle of plot
    ax: matplotlib.pyplot.Axes
        Figure plot axis
    """

    if not isinstance(fmin, (int, float)) or not isinstance(fmax, (int, float)):
        raise TypeError("fmin and fmax must be numeric types.")
    if fmin >= fmax:
        raise ValueError("fmin must be less than fmax.")

    # Set dimension names.
    # "freq" or "freq_bins" is always second
    freq = spsdl.dims[-1]

    # Check fmax
    fn = spsdl[freq].max().item()
    fmax = _fmax_warning(fn, fmax)
    # select frequency range
    spsdl = spsdl.sel({freq: slice(fmin, fmax)})

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"xscale": "log"})
        fig.subplots_adjust(
            left=0.1, right=0.95, top=0.85, bottom=0.2, hspace=0.3, wspace=0.15
        )
    ax.plot(spsdl[freq], spsdl.T, **kwargs)
    ax.set(
        xlim=(fmin, fmax), xlabel="Frequency [Hz]", ylabel=getattr(spsdl, "units", None)
    )

    return fig, ax
