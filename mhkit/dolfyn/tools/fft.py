import numpy as np
from .misc import detrend_array

fft = np.fft.fft


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


def _getwindow(window, nfft):
    if window is None:
        window = np.ones(nfft)
    elif isinstance(window, (int, float)) and window == 1:
        window = np.ones(nfft)
    elif isinstance(window, str):
        if "hann" in window:
            window = np.hanning(nfft)
        elif "hamm" in window:
            window = np.hamming(nfft)
        else:
            raise ValueError("Unsupported window type: {}".format(window))
    elif isinstance(window, np.ndarray):
        if len(window) != nfft:
            raise ValueError("Custom window length must be equal to nfft")
    else:
        raise ValueError("Invalid window parameter")

    return window


def _stepsize(l, nfft, nens=None, step=None):
    """
    Calculates the fft-step size for a length *l* array.

    If nens is None, the step size is chosen to maximize data use,
    minimize nens and have a minimum of 50% overlap.

    If nens is specified, the step-size is computed directly.

    Parameters
    ----------
    l       : The length of the array.
    nfft    : The number of points in the fft.
    nens : The number of nens to perform (default compute this).

    Returns
    -------
    step    : The step size.
    nens    : The number of ensemble ffts to average together.
    nfft    : The number of points in the fft (set to l if nfft>l).
    """

    if l < nfft:
        nfft = l
    if nens is None and step is None:
        if l == nfft:
            return 0, 1, int(nfft)
        nens = int(2.0 * l / nfft)
        return int((l - nfft) / (nens - 1)), nens, int(nfft)
    elif nens is None:
        return int(step), int((l - nfft) / step + 1), int(nfft)
    else:
        if nens == 1:
            return 0, 1, int(nfft)
        return int((l - nfft) / (nens - 1)), int(nens), int(nfft)


def cpsd_quasisync_1D(a, b, nfft, fs, window="hann"):
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

    Returns
    -------
    cpsd : numpy.ndarray
      The cross-spectral density of `a` and `b`.

    See Also
    ---------
    :func:`psd`,
    :func:`coherence_1D`,
    :func:`cpsd_1D`,
    numpy.fft

    Notes
    -----
    `a` and `b` do not need to be 'tightly' synchronized, and can even
    be different lengths, but the first- and last-index of both series
    should be synchronized (to whatever degree you want unbiased
    phases).

    This performs:

    .. math::

        fft(a)*conj(fft(b))

    Note that this is consistent with :func:`numpy.correlate`.

    It detrends the data and uses a minimum of 50% overlap for the
    shorter of `a` and `b`. For the longer, the overlap depends on the
    difference in size.  1-(l_short/l_long) data will be underutilized
    (where l_short and l_long are the length of the shorter and longer
    series, respectively).

    The units of the spectra is the product of the units of `a` and
    `b`, divided by the units of fs.
    """

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise Exception("Velocity cannot be complex")
    l = [len(a), len(b)]
    if l[0] == l[1]:
        return cpsd_1D(a, b, nfft, fs, window=window)
    elif l[0] > l[1]:
        a, b = b, a
        l = l[::-1]
    step = [0, 0]
    step[0], nens, nfft = _stepsize(l[0], nfft)
    step[1], nens, nfft = _stepsize(l[1], nfft, nens=nens)
    fs = np.float64(fs)
    window = _getwindow(window, nfft)
    fft_inds = slice(1, int(nfft / 2.0 + 1))
    wght = 2.0 / (window**2).sum()
    pwr = fft(detrend_array(a[0:nfft]) * window)[fft_inds] * np.conj(
        fft(detrend_array(b[0:nfft]) * window)[fft_inds]
    )
    if nens - 1:
        for i1, i2 in zip(
            range(step[0], l[0] - nfft + 1, step[0]),
            range(step[1], l[1] - nfft + 1, step[1]),
        ):
            pwr += fft(detrend_array(a[i1 : (i1 + nfft)]) * window)[fft_inds] * np.conj(
                fft(detrend_array(b[i2 : (i2 + nfft)]) * window)[fft_inds]
            )
    pwr *= wght / nens / fs
    return pwr


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
        raise Exception("Velocity cannot be complex")
    auto_psd = False
    if a is b:
        auto_psd = True
    l = len(a)
    step, nens, nfft = _stepsize(l, nfft, step=step)
    fs = np.float64(fs)
    window = _getwindow(window, nfft)
    fft_inds = slice(1, int(nfft / 2.0 + 1))
    wght = 2.0 / (window**2).sum()
    s1 = fft(detrend_array(a[0:nfft]) * window)[fft_inds]
    if auto_psd:
        pwr = np.abs(s1) ** 2
    else:
        pwr = s1 * np.conj(fft(detrend_array(b[0:nfft]) * window)[fft_inds])
    if nens - 1:
        for i in range(step, l - nfft + 1, step):
            s1 = fft(detrend_array(a[i : (i + nfft)]) * window)[fft_inds]
            if auto_psd:
                pwr += np.abs(s1) ** 2
            else:
                pwr += s1 * np.conj(
                    fft(detrend_array(b[i : (i + nfft)]) * window)[fft_inds]
                )
    pwr *= wght / nens / fs
    return pwr


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

    return np.abs(cpsd_1D(a, a, nfft, fs, window=window, step=step))
