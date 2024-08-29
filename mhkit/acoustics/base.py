import warnings
import numpy as np
import xarray as xr

from mhkit.dolfyn import VelBinner
from mhkit.dolfyn.time import epoch2dt64, dt642epoch


def minimum_frequency(water_depth, c=1500, c_seabed=1700):
    """
    Estimate the shallow water cutoff frequency based on the speed of
    sound in the water column and the speed of sound in the seabed
    material (generally ranges from 1450 - 1800 m/s)

    Jennings 2011 - Computational Ocean Acoustics, 2nd ed.
    """

    f_min = c / (4 * water_depth * np.sqrt(1 - (c / c_seabed) ** 2))

    return f_min


def sound_pressure_spectral_density(P, fs, window=1):
    """
    Calculates the mean square sound pressure spectral density from audio
    samples split into FFTs with a specified window_size in seconds and
    at least a 50% overlap. The amplitude of the PSD is adjusted
    according to Parseval's theorem.

    Parameters
    ----------
    P: xarray.DataArray (time)
        Sound pressure in [Pa] or volts [V]
    fs: int
        Data collection sampling rate [Hz]
    window: string (optional)
        Length of time in seconds to create FFTs. Default: 1 s.

    Returns
    -------
    spsd: xarray.DataArray (time, freq)
        Spectral density [Pa^2/Hz] indexed by time and frequency
    """

    # window length of each time series
    win = window * fs

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
            "units": P.units + "^2/Hz",
            "long_name": "Mean Square Sound Pressure Spectral Density",
            "fs": fs,
            "window": str(window) + "s",
            "overlap": "50%",
            "nfft": win,
        },
    )

    return out


def apply_calibration(spsd, sensitivity_curve, fill_value):
    """
    Applies custom calibration to spectral density values

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in V^2/Hz.
    sensitivity_curve: xarray.DataArray (freq)
        Calibrated sensitivity curve in units of dB rel 1 V^2/uPa^2.
    fill_value: numeric
        Value with which to fill values missing from calibration curve,
        in units of dB rel 1 V^2/uPa^2.

    Returns
    -------
    spsd: xarray.DataArray (time, freq)
        Spectral density [Pa^2/Hz] indexed by time and frequency
    """

    # Read calibration curve
    freq = sensitivity_curve.dims[0]
    # Interpolate calibration curve to desired value
    calibration = sensitivity_curve.interp({freq: spsd["freq"]}, method="linear").drop(
        freq
    )
    # Fill missing with provided value
    calibration = calibration.fillna(fill_value)

    # Subtract from sound pressure spectral density
    Sf_ratio = 10 ** (calibration / 10)  # V^2/uPa^2
    spsd /= Sf_ratio  # uPa^2/Hz
    spsd /= 1e12  # Pa^2/Hz
    spsd.attrs["units"] = "Pa^2/Hz"

    return spsd


def sound_pressure_spectral_density_level(spsd):
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
    P2_ref = 1e-12  # Pa^2 to 1 uPa^2

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
    spsdl, octave=3, fmin=10, fmax=100000, method="median", method_arg=None
):
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
    method: str
        xarray.DataArray method to run on the binned data. Default: "median".
        Options: [median, mean, min, max, sum, quantile, std, var, count]
    method_arg: numeric
        Optional argument for `method`. Only required for "quantile" function.

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Frequency band-averaged sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time and frequency
    """

    fn = spsdl["freq"].max().values
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    bandwidth = 2 ** (1 / octave)
    half_bandwidth = 2 ** (1 / (octave * 2))

    center_freq = 10 ** np.arange(
        np.log10(fmin),
        np.log10(fmax * bandwidth),
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


def time_average(spsdl, window=60, method="median", method_arg=None):
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
    method: str
        xarray.DataArray method to run on the binned data. Default: "median".
        Options: [median, mean, min, max, sum, quantile, std, var, count]
    method_arg: numeric
        Optional argument for `method`. Only required for "quantile" function.

    Returns
    -------
    out: xarray.DataArray (time_bins, freq)
        Time-averaged sound pressure spectral density level [dB re 1 uPa^2/Hz]
        indexed by time and frequency
    """

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
    func = getattr(spsdl_group, method.lower())
    out = func(method_arg)
    out.attrs["units"] = spsdl.units
    out.attrs["comment"] = f"Time average {method}"

    # This coordinate hangs on for some reason
    if method == "quantile":
        out = out.drop_vars("quantile")

    return out


def sound_pressure_level(spsd, fmin=10, fmax=100000):
    """
    Calculates the sound pressure level in a specified frequency band
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    psd: xarray.DataArray (time, freq)
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

    fn = spsd.attrs["fs"] // 2
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2, = 1 uPa^2

    # Mean square sound pressure in a specified frequency band from mean square values
    P2 = np.trapz(
        spsd.sel(freq=slice(fmin, fmax)), spsd["freq"].sel(freq=slice(fmin, fmax))
    )

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


def _band_sound_pressure_level(spsd, bandwidth, half_bandwidth, fmin, fmax):
    fn = spsd.attrs["fs"] // 2
    if fmax > fn:
        warnings.warn(
            "`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            "fmax = {fn}"
        )
        fmax = fn

    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2, = 1 uPa^2

    center_freq = 10 ** np.arange(
        np.log10(fmin),
        np.log10(fmax * bandwidth),
        step=np.log10(bandwidth),
    )
    lower_limit = center_freq / half_bandwidth
    upper_limit = center_freq * half_bandwidth
    octave_bins = np.append(lower_limit, upper_limit[-1])

    # Manual trapezoidal rule to get Pa^2
    P2 = xr.DataArray(
        coords={"time": spsd["time"], "freq_bins": center_freq},
        dims=["time", "freq_bins"],
    )
    for i, key in enumerate(center_freq):
        band_min = octave_bins[i]
        band_max = octave_bins[i + 1]
        P2.loc[dict(freq_bins=key)] = np.trapz(
            spsd.sel(freq=slice(band_min, band_max)),
            spsd["freq"].sel(freq=slice(band_min, band_max)),
        )

    # Mean square sound pressure level in dB rel 1 uPa
    mspl = 10 * np.log10(P2 / P2_ref)

    return mspl


def third_octave_sound_pressure_level(spsd, fmin=10, fmax=100000):
    """
    Calculates the sound pressure level in third octave bands directly
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    psd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
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

    return mspl


def decidecade_sound_pressure_level(spsd, fmin=10, fmax=100000):
    """
    Calculates the sound pressure level in decidecade bands directly
    from the mean square sound pressure spectral density.

    Parameters
    ----------
    psd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density.
    fmin: int
        Lower frequency band limit (lower limit of the hydrophone). Default: 10 Hz
    fmax: int
        Upper frequency band limit (Nyquist frequency). Default: 100000 Hz

    Returns
    -------
    out: xarray.DataArray (time, freq_bins)
        Sound pressure level [dB re 1 uPa] indexed by time and third octave bands
    """

    # Decidecade bin frequencies
    bandwidth = 2 ** (1 / 10)
    half_bandwidth = 2 ** (1 / 20)

    mspl = _band_sound_pressure_level(spsd, bandwidth, half_bandwidth, fmin, fmax)
    mspl.attrs = {
        "units": "dB re 1 uPa",
        "long_name": "Decidecade Sound Pressure Level",
    }

    return mspl
