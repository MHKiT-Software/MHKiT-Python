"""
This submodule includes the main passive acoustics plotting functions. All
functions allow passthrough of matplotlib functionality and commands
to make them fully customizable.
"""

import matplotlib.pyplot as plt
from .base import _fmax_warning


def plot_spectogram(spsdl, fmin=10, fmax=100000, fig=None, ax=None, **kwargs):
    """
    Plots the spectogram of the sound pressure spectral density level.

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
    ax: matplotlib.pyplot.axis
        Figure plot axis
    """

    # Set dimension names
    time = spsdl.dims[0]
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
        spsdl[time].values, spsdl[freq].values, spsdl.T, shading="nearest", **kwargs
    )
    fig.colorbar(h, ax=ax, label="dB re 1 uPa^2/Hz")
    ax.set(xlabel="Time", ylabel="Frequency [Hz]")

    return fig, ax


def plot_spectra(spsdl, fmin=10, fmax=100000, fig=None, ax=None, **kwargs):
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
    ax: matplotlib.pyplot.axis
        Figure axis containing plot objects
    kwargs: dict
        Dictionary of matplotlib function keyword arguments

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure handle of plot
    ax: matplotlib.pyplot.axis
        Figure plot axis
    """

    # Set dimension names
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
    ax.set(xlim=(fmin, fmax), xlabel="Frequency [Hz]", ylabel="$dB re 1 uPa^2/Hz$")

    return fig, ax
