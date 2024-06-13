import wave
import warnings
import numpy as np
import xarray as xr

from mhkit.dolfyn import VelBinner
from mhkit.dolfyn.time import epoch2dt64, dt642epoch


def sound_pressure_spectral_density(P, fs, window_size=1):
    """
    Calculates the mean square sound pressure spectral density from audio
    samples split into FFTs with a specified window_size in seconds and
    at least a 50% overlap.

    Parameters
    ----------
    P: xarray.DataArray (time, frequency)
        Sound pressure in Pascals.
    fs: integer
        Data collection sampling rate [Hz]
    window_size: string (optional)
        Length of time in seconds to create FFTs. Default is 1 s.

    Returns
    -------
    out: xarray.DataArray
        Spectral density [Pa^2/Hz] indexed by time[s] and frequency [Hz]

    """
    # window length of each time series
    win = window_size * fs

    # Use dolfyn PSD
    binner = VelBinner(n_bin=win, fs=fs, n_fft=win)
    # Always 50% overlap if numbers reshape perfectly
    # Mean square sound pressure
    psd = binner.power_spectral_density(P, freq_units="Hz")
    samples = binner.reshape(P.values) - binner.mean(P.values)[:, None]
    # Power in time domain
    t_power = np.sum(samples**2, axis=1) / win
    # Power in frequency domain
    f_power = psd.sum("freq") * (fs / win)
    # Adjust the amplitude of PSD according to Parseval's theorem
    psd_adj = psd * t_power[:, None] / f_power

    out = xr.DataArray(
        psd_adj,
        coords={"time": psd_adj["time"], "freq": psd_adj["freq"]},
        attrs={
            "units": "Pa^2/Hz",
            "long_name": "Mean Square Sound Pressure Spectral Density",
            "fs": fs,
            "window": str(window_size) + " s",
            "overlap": "50%",
            "nfft": win,
        },
    )

    return out


def sound_pressure_spectral_density_level(spsd):
    """
    Calculates the sound pressure spectral density level from
    the mean square sound pressure spectral density.

    Parameters
    ----------
    spsd: xarray DataArray (time, freq)
        Mean square sound pressure spectral density in uPa^2/Hz

    Returns
    -------
    out: xarray.DataArray
        Sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time [s] and frequency
    """
    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2/Hz, = 1 uPa^2/Hz

    # Sound pressure spectral density level from mean square values
    lpf = 10 * np.log10(spsd.values / P2_ref)

    out = xr.DataArray(
        lpf,
        coords={"time": spsd["time"], "freq": spsd["freq"]},
        attrs={
            "units": "dB re 1 uPa^2/Hz",
            "long_name": "Sound Pressure Spectral Density Level",
        },
    )

    return out


def band_average(
    spsdl, octave=3, min_band=10, max_band=96000, method="mean", method_arg=None
):
    """Reorganizes and averages spectral density level frequency
    vector into fractional octave bands
    """

    fn = spsdl["freq"].max().values
    if max_band > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        max_band = fn

    bandwidth = 2 ** (1 / octave)
    half_bandwidth = 2 ** (1 / (octave * 2))

    center_freq = 10 ** np.arange(
        np.log10(min_band),
        np.log10(max_band * bandwidth),
        step=np.log10(bandwidth),
    )
    lower_limit = center_freq / half_bandwidth
    upper_limit = center_freq * half_bandwidth
    octave_bins = np.append(lower_limit, upper_limit[-1])

    # Use xarray binning methods
    spsdl_group = spsdl.groupby_bins("freq", octave_bins, labels=center_freq)
    func = getattr(spsdl_group, method.lower())
    out = func(method_arg)
    out.attrs["units"] = spsdl.units
    out.attrs["comment"] = f"Third octave frequency band {method}"

    return out


def time_average(spsdl, window, method="mean", method_arg=None):

    time_bins = np.reshape(spsdl["time"].values, (window, spsdl["time"].size // window))
    time_labels = epoch2dt64(np.mean(dt642epoch(time_bins), axis=-1))

    # Use xarray binning methods
    spsdl_group = spsdl.groupby_bins("time", window, labels=time_labels)
    func = getattr(spsdl_group, method.lower())
    out = func(method_arg)
    out.attrs["units"] = spsdl.units
    out.attrs["comment"] = f"Time average {method}"

    return out


def sound_pressure_level(spsd, fmin=10, fmax=96000):
    """
    Calculates the sound pressure level in a specified frequency band
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    psd: xarray DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: integer
        Lower frequency band limit (lower limit of the hydrophone)
    fmax: integer
        Upper frequency band limit (Nyquist frequency)

    Returns
    -------
    out: xarray.DataArray
        Sound pressure level [dB re 1 uPa] indexed by time [s]
    """

    fn = spsd.attrs["fs"] // 2
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2, = 1 uPa^2

    df = spsd.attrs["fs"] / spsd.attrs["nfft"]
    nfmin = fmin // df
    nfmax = fmax // df

    # Mean square sound pressure in a specified frequency band from mean square values
    P2 = spsd.sel(freq=slice(nfmin, nfmax)).sum("freq").values * df
    # Mean square sound pressure level
    mspl = 10 * np.log10(P2 / P2_ref)

    out = xr.DataArray(
        mspl,
        coords={"time": spsd["time"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Sound Pressure Level",
            "freq_band_min": fmin,
            "freq_band_max": fmax,
        },
    )

    return out


def third_octave_sound_pressure_level(spsd, min_band=10, max_band=96000):
    """
    Calculates the sound pressure level in a specified frequency band
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    psd: xarray DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: integer
        Lower frequency band limit (lower limit of the hydrophone)
    fmax: integer
        Upper frequency band limit (Nyquist frequency)

    Returns
    -------
    out: xarray.DataArray
        Sound pressure level [dB re 1 uPa] indexed by time [s]
    """

    fn = spsd.attrs["fs"] // 2
    if max_band > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2, = 1 uPa^2

    df = spsd.attrs["fs"] / spsd.attrs["nfft"]

    # Third octave bin frequencies
    bandwidth = 2 ** (1 / 3)
    half_bandwidth = 2 ** (1 / 6)

    center_freq = 10 ** np.arange(
        np.log10(min_band),
        np.log10(max_band * bandwidth),
        step=np.log10(bandwidth),
    )
    lower_limit = center_freq / half_bandwidth
    upper_limit = center_freq * half_bandwidth
    octave_bins = np.append(lower_limit, upper_limit[-1])

    # Mean square sound pressure in a specified frequency band from mean square values
    spsd_group = spsd.groupby_bins("freq", octave_bins, labels=center_freq)
    P2 = spsd_group.sum() * df

    # Mean square sound pressure level
    mspl = 10 * np.log10(P2 / P2_ref)

    out = xr.DataArray(
        mspl,
        coords={"time": spsd["time"], "freq_bins": P2["freq_bins"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Third Octave Sound Pressure Level",
        },
    )

    return out
