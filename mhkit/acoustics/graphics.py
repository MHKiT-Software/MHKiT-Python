import warnings
import matplotlib.pyplot as plt


def plot_spectogram(spsdl, fmin=10, fmax=96000, fig=None, ax=None, kwargs={}):

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
        spsdl["time"].values, spsdl["freq"].values, spsdl.T, shading="nearest", **kwargs
    )
    fig.colorbar(h, ax=ax, label="dB re 1 uPa^2/Hz")
    ax.set(xlabel="Time", ylabel="Frequency [Hz]")

    return fig, ax


def plot_spectra(spsdl, fmin=10, fmax=96000, fig=None, ax=None, kwargs={}):
    """Plots pressure spectra"""

    freq = spsdl.dims[-1]
    fn = spsdl[freq].max()
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    spsdl = spsdl.sel({freq: slice(fmin, fmax)})

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"xscale": "log"})
        fig.subplots_adjust(
            left=0.1, right=0.95, top=0.85, bottom=0.2, hspace=0.3, wspace=0.15
        )
    ax.plot(spsdl[freq], spsdl.T, **kwargs)
    ax.set(xlim=(fmin, fmax), xlabel="Frequency [Hz]", ylabel="dB re 1 uPa^2/Hz")

    return fig, ax
