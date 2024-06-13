import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cmocean.cm import thermal
import matplotlib.dates as mdates


def plot_spectogram(spsdl, fmin=20, fmax=512000 // 2, fig=None, ax=None, vmin=0, vmax=100):

    fn = spsdl["freq"].max()
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    spsdl = spsdl.sel(freq=slice(fmin, fmax))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"yscale": "log"})
        fig.subplots_adjust(left=0.1, right=0.95, top=0.97, bottom=0.11)
    h = ax.pcolormesh(
        spsdl["time"].values,
        spsdl["freq"].values,
        spsdl.T,
        cmap=thermal,
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(h, ax=ax, label="dB re 1 uPa^2/Hz")
    ax.set(xlabel="Time", ylabel="Frequency [Hz]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return fig, ax


def plot_spectrum(spsdl, fmin=20, fmax=512000 // 4, fig=None, ax=None):

    freq = spsdl.dims[-1]
    fn = spsdl[freq].max()
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    spsdl = spsdl.sel({freq: slice(fmin, fmax)}).mean("time")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.subplots_adjust(
            left=0.1, right=0.95, top=0.85, bottom=0.2, hspace=0.3, wspace=0.15
        )
    ax.loglog(spsdl[freq], spsdl.T)
    ax.set(xlim=(fmin, fmax), xlabel="Frequency [Hz]", ylabel="dB re 1 uPa^2/Hz")

    return fig, ax
