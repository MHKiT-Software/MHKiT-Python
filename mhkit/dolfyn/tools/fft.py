import warnings
import numpy as np
import scipy.signal


def fft_frequency(nfft, fs, full=False):
    """
    Compute the frequency vector for a given `nfft` and `fs`.

    Parameters
    ----------
    fs : float
      The sampling frequency (e.g. samples/sec)
    nfft : int
      The number of samples in a window.
    full : bool
      Whether to return half frequencies (positive), or the full frequencies.
      Default = False

    Returns
    -------
    freq : numpy.ndarray
      The frequency vector, in same units as 'fs'
    """

    fs = np.float64(fs)
    f = np.fft.fftfreq(int(nfft), 1 / fs)
    if full:
        return f
    else:
        return np.abs(f[1 : int(nfft / 2.0 + 1)])


def _normalize_window(window, nfft):
    """
    Convert a dolfyn window specification to a form scipy.signal accepts.

    Parameters
    ----------
    window : {None, 1, 'hann', 'hamm', numpy.ndarray}
      Window specification used by dolfyn (None/1 = boxcar).
    nfft : int
      FFT length (used only to validate array windows).

    Returns
    -------
    window : str or numpy.ndarray
      Window specification accepted by scipy.signal.
    """
    if window is None or (isinstance(window, (int, float)) and window == 1):
        return "boxcar"
    if isinstance(window, str):
        if "hann" in window:
            return "hann"
        if "hamm" in window:
            return "hamming"
        return window  # pass other strings directly to scipy
    if isinstance(window, np.ndarray):
        if len(window) != nfft:
            raise ValueError("Custom window length must equal nfft")
    return window  # ndarray: pass through


def _step_to_noverlap(step, nfft):
    """
    Validate a dolfyn step (integer sample count) and convert to scipy noverlap.

    Parameters
    ----------
    step : int or None
      Number of samples to advance between FFT windows.  None lets scipy
      choose its default (50% overlap).
    nfft : int
      FFT window length.

    Returns
    -------
    noverlap : int or None
    """
    if step is None:
        return None
    step = int(step)
    if step < 1:
        raise ValueError(
            f"step must be a positive integer number of samples, got {step}. "
            "For exactly 50% overlap use step=n_fft//2."
        )
    if step > nfft / 2:
        warnings.warn(
            f"Specified step ({step}) is greater than nfft/2 ({nfft / 2}), "
            "resulting in less than 50% overlap between FFT windows."
        )
    return nfft - step


def cpsd_1D(a, b, nfft, fs, window="hann", step=None):
    """
    Compute the cross power spectral density (CPSD) of the signals `a` and `b`.

    Parameters
    ----------
    a : numpy.ndarray
      The first signal.
    b : numpy.ndarray
      The second signal.
    nfft : int
      The number of points in the fft.
    fs : float
      The sample rate (e.g. sample/second).
    window : {None, 1, 'hann', numpy.ndarray}
      The window to use (default: 'hann'). Valid entries are:
      - None,1               : uses a 'boxcar' or ones window.
      - 'hann'               : hanning window.
      - a length(nfft) array : use this as the window directly.
    step : int
      Use this to specify the overlap.  For example:
      - step : nfft/2 specifies a 50% overlap.
      - step : nfft specifies no overlap.
      - step=2*nfft means that half the data will be skipped.
      By default, `step` is calculated to maximize data use, have
      at least 50% overlap and minimize the number of ensembles.

    Returns
    -------
    cpsd : numpy.ndarray
      The cross-spectral density of `a` and `b`.

    See also
    --------
    :func:`psd`
    :func:`coherence_1D`
    `numpy.fft`

    Notes
    -----
    cpsd removes a linear trend from the signals.

    The two signals should be the same length, and should both be real.

    This performs:

    .. math::

        fft(a)*conj(fft(b))

    This implementation is consistent with the numpy.correlate
    definition of correlation.  (The conjugate of D.B. Chelton's
    definition of correlation.)

    The units of the spectra is the product of the units of `a` and
    `b`, divided by the units of fs.
    """

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("Velocity cannot be complex")
    noverlap = _step_to_noverlap(step, nfft)
    _, cpsd = scipy.signal.csd(
        a,
        b,
        fs=fs,
        window=_normalize_window(window, nfft),
        nperseg=nfft,
        noverlap=noverlap,
        detrend="linear",
        return_onesided=True,
        scaling="density",
    )
    # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
    return cpsd[1:]


def psd_1D(a, nfft, fs, window="hann", step=None):
    """
    Compute the power spectral density (PSD).

    This function computes the one-dimensional `n`-point PSD.

    The units of the spectra is the product of the units of `a` and
    `b`, divided by the units of fs.

    Parameters
    ----------
    a : numpy.ndarray
      The first signal, as a 1D vector
    nfft : int
      The number of points in the fft.
    fs : float
      The sample rate (e.g. sample/second).
    window : {None, 1, 'hann', numpy.ndarray}
      The window to use (default: 'hann'). Valid entries are:
      - None,1               : uses a 'boxcar' or ones window.
      - 'hann'               : hanning window.
      - a length(nfft) array : use this as the window directly.
    step : int
      Use this to specify the overlap.  For example:
      - step : nfft/2 specifies a 50% overlap.
      - step : nfft specifies no overlap.
      - step=2*nfft means that half the data will be skipped.
      By default, `step` is calculated to maximize data use, have
      at least 50% overlap and minimize the number of ensembles.

    Returns
    -------
    cpsd : numpy.ndarray
      The cross-spectral density of `a` and `b`.

    Notes
    -----
    Credit: This function's line of code was copied from JN's fast_psd.m
    routine.

    See Also
    --------
    :func:`cpsd_1D`
    :func:`coherence_1D`
    `numpy.fft`
    """

    noverlap = _step_to_noverlap(step, nfft)
    _, psd = scipy.signal.welch(
        a,
        fs=fs,
        window=_normalize_window(window, nfft),
        nperseg=nfft,
        noverlap=noverlap,
        detrend="linear",
        return_onesided=True,
        scaling="density",
    )
    # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
    return psd[1:]
