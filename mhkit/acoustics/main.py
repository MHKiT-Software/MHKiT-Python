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


def spectral_density(P, fs, window_size=1, overlap=0.5, Vo2=1):
    """
    Calculate sound pressure level from the power spectral density of sound
    samples split into windows with a specific overlap.
    """
    # window length of each time series
    win = window_size * fs
    # overlap between each window
    step = overlap * fs
    # number of time series samples
    ns = np.floor((len(P) - win) / step)
    # number of fft points
    nfft = 2 * 2.0 ** (np.ceil(np.log(win) / np.log(2)))
    # frequency resolution
    df = fs / nfft
    # Next highest power of 2 greater than length(x).
    nfreq = np.ceil((nfft + 1) / 2)
    # mean-squared sound pressure spectral density
    Pf2 = np.zeros((nfreq, ns))

    # Should be able to do this with a reshape
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
            "units": "Pa^2 / Hz",
            "long_name": "Mean-Squared Sound Pressure Spectral Density",
            "fs": fs,
            "window": win,
            "overlap": overlap,
            "n_samples": ns,
            "nfft": nfft,
        },
    )

    return out


def spectral_density_level(psd, Po2=1e-12):
    """
    Mean-squared sound pressure spectral density level
    """
    # mean-squared sound pressure spectral density level
    Lpf = 10 * np.log10(psd.values / Po2)

    SPL = xr.DataArray(
        Lpf,
        coords={"time": psd["time"], "freq": psd["freq"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Mean-Squared Sound Pressure Spectral Density Level",
        },
    )

    return SPL


def sound_pressure_level(psd, Po2=1e-12, fmin=20, fmax=192000 / 2):
    """
    Sound pressure level in a band for marine energy converter
    """
    df = psd.attrs["fs"] / psd.attrs["nfft"]
    nfmin = fmin // df
    nfmax = fmax // df

    # marine energy converter sound pressure level in a band
    mspl = 10 * np.log10(np.sum(psd.sel(freq=slice(nfmin, nfmax)), axis=0) * df / Po2)

    MSPL = xr.DataArray(
        mspl,
        coords={"time": psd["time"]},
        attrs={
            "units": "dB re 1 uPa",
            "long_name": "Mean-Squared Sound Pressure Level",
        },
    )

    return MSPL
