import numpy as np
import xarray as xr
from scipy.signal import periodogram, welch


def _fast_fourier_transform(fs, x):
    # demean the time series
    x -= np.mean(x)
    # add hanning window
    x *= np.hanning(len(x))
    # Nyquist frequency
    fn = fs / 2
    # Next highest power of 2 greater than length(x).
    n_fft = 2 * 2 ** (np.ceil(np.log(len(x)) / np.log(2)))
    # Take FFT, padding with zeros
    temp_fft = np.fft.fft(x, n_fft)
    # FFT is symmetric, throw away second half
    n_unique = np.ceil((n_fft + 1) / 2)
    spec = temp_fft[:n_unique]
    # Multiply by 2 to take into account the fact that we
    # threw out second half of TempFFT above
    spec = spec * 2
    # Account for endpoint uniqueness
    spec[0] = spec(1) / 2
    # We know n_fft is even
    spec[-1] = spec(len(spec)) / 2
    # Scale the FFT so that it is not a function of the length of x.
    spec = spec / len(x)
    # Frequency vector
    f = np.arange(n_unique) * 2 * fn / n_fft

    return f, spec


# Vo2 = 1  # reference v0^2, V^2/Hz
# Po2 = 1e-12  # reference p0^2, Pa^2/Hz
# G = 0  # amplifier gain in dB
# fmin = 20  # frequency band lower limit of the hydrophone
# fmax = 192000 / 2  # Nyquist frequency


def sound_pressure_spectral_density(P, fs, window_size=1, overlap=0.5):
    """
    Calculates the mean square sound pressure spectral density from audio
    samples split into FFTs with a specified window_size in seconds and
    a specified percent overlap.

    Parameters
    ----------
    P: xarray.DataArray (time, frequency)
        Sound pressure in Pascals.
    fs: integer
        Data collection sampling rate [Hz]
    window_size: string (optional)
        Length of time in seconds to create FFTs. Default is 1 s.
    overlap: numeric
        Percent overlap between neighboring FFTs. Default is 50%.

    Returns
    -------
    out: xarray.DataArray
        Spectral density [Pa^2/Hz] indexed by time[s] and frequency [Hz]

    """
    # window length of each time series
    win = window_size * fs
    # overlap between each window
    step = int(overlap * fs)
    # number of time series samples
    ns = int(np.floor((len(P) - win) / step))
    # number of fft points
    nfft = int(2 * 2.0 ** (np.ceil(np.log(win) / np.log(2))))
    # frequency resolution
    df = fs / nfft
    # Next highest power of 2 greater than length(x).
    nfreq = int(np.ceil((nfft + 1) / 2))
    # mean-squared sound pressure spectral density
    Pf2 = np.zeros((nfreq, ns))

    # Takes too long
    for i in range(ns):
        sample = P[i * step : i * step + win]
        sample = sample - np.mean(sample)
        # mean squred sound pressure; power in time domain
        t_power = sum(sample * 2) / len(sample)
        freq, spec = _fast_fourier_transform(fs, sample)
        # PSD
        psd = spec * np.conj(spec) / df / 2
        # power in frequency domain
        f_power = np.sum(psd) * df
        # adjust the amplitude of PSD according to Parseval's theorem
        psd_adj = psd * t_power / f_power
        # mean-squared sound pressure spectral density
        Pf2[:, i] = psd_adj

    ts = np.arange(len(P)) / fs  # time label for the whole time series
    out = xr.DataArray(
        psd_adj.T,
        coords={"time": ts, "freq": freq},
        attrs={
            "units": "Pa^2/Hz",
            "long_name": "Mean Square Sound Pressure Spectral Density",
            "fs": fs,
            "window": win,
            "overlap": overlap,
            "n_samples": ns,
            "nfft": nfft,
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


def sound_pressure_level(spsd, fmin=20, fmax=192000 // 2):
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
    # Reference value of sound pressure
    P2_ref = 1e-12  # Pa^2, = 1 uPa^2

    df = spsd.attrs["fs"] / spsd.attrs["nfft"]
    nfmin = fmin // df
    nfmax = fmax // df

    # Sound pressure level in a specified frequency band from mean square values
    P2 = (
        spsd.sel(freq=slice(nfmin, nfmax)).sum("freq").values * df
    )  # mean square sound pressure
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
