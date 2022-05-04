import numpy as np
from .misc import detrend
fft = np.fft.fft


def psd_freq(nfft, fs, full=False):
    """
    Compute the frequency for vector for a `nfft` and `fs`.

    Parameters
    ----------
    fs : float
      The sampling frequency (e.g. samples/sec)
    nfft : int
      The number of samples in a window.
    full : bool (default: False)
      Whether to return half frequencies (positive), or the full frequencies.

    Returns
    -------
    freq : |np.ndarray|
      The frequency vector, in same units as 'fs'

    """
    fs = np.float64(fs)
    f = np.fft.fftfreq(int(nfft), 1 / fs)
    if full:
        return f
    else:
        return np.abs(f[1:int(nfft / 2. + 1)])


def _getwindow(window, nfft):
    if window == 'hann':
        window = np.hanning(nfft)
    elif window == 'hamm':
        window = np.hamming(nfft)
    elif window is None or window == 1:
        window = np.ones(nfft)
    return window


def stepsize(l, nfft, nens=None, step=None):
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
        nens = int(2. * l / nfft)
        return int((l - nfft) / (nens - 1)), nens, int(nfft)
    elif nens is None:
        return int(step), int((l - nfft) / step + 1), int(nfft)
    else:
        if nens == 1:
            return 0, 1, int(nfft)
        return int((l - nfft) / (nens - 1)), int(nens), int(nfft)


def coherence(a, b, nfft, window='hann', debias=True, noise=(0, 0)):
    """
    Computes the magnitude-squared coherence of `a` and `b`.

    Parameters
    ----------
    a : |np.ndarray|
      The first array over which to compute coherence.
    b : |np.ndarray|
      The second array over which to compute coherence.
    nfft : int
      The number of points to use in the fft.
    window : string, np.ndarray (default 'hann')
      The window to use for ffts.
    debias : bool (default: True)
      Specify whether to debias the signal according to Benignus1969.
    noise : tuple(2), or float
      The `noise` keyword may be used to specify the signals'
      noise levels (std of noise in a,b). If `noise` is a two
      element tuple or list, the first and second elements specify
      the noise levels of a and b, respectively.
      default: noise=(0,0)

    Returns
    -------
    out : |np.ndarray|
      Coherence between `a` and `b`

    Notes
    -----
    Coherence is defined as:

    .. math::

      C_{ab} = \\frac{|S_{ab}|^2}{S_{aa} * S_{bb}}

    Here :math:`S_{ab}`, :math:`S_{aa}` and :math:`S_{bb}` are the cross,
    and auto spectral densities of the signal `a` and `b`.

    """
    l = [len(a), len(b)]
    cross = cpsd_quasisync
    if l[0] == l[1]:
        cross = cpsd
    elif l[0] > l[1]:
        a, b = b, a
        l = l[::-1]
    step1, nens, nfft = stepsize(l[0], nfft)
    step2, nens, nfft = stepsize(l[1], nfft, nens=nens)
    if noise.__class__ not in [list, tuple, np.ndarray]:
        noise = [noise, noise]
    elif len(noise) == 1:
        noise = [noise[0], noise[0]]
    if nens <= 2:
        raise Exception("Coherence must be computed from a set of ensembles.")
    # fs=1 is ok because it comes out in the normalization.  (noise
    # normalization depends on this)
    out = ((np.abs(cross(a, b, nfft, 1, window=window)) ** 2) /
           ((psd(a, nfft, 1, window=window, step=step1) - noise[0] ** 2 / np.pi) *
            (psd(b, nfft, 1, window=window, step=step2) - noise[1] ** 2 / np.pi))
           )
    if debias:
        # This is from Benignus1969, it seems to work (make data with different
        # nens (nfft) agree).
        return out * (1 + 1. / nens) - 1. / nens
    return out


def cpsd_quasisync(a, b, nfft, fs, window='hann'):
    """
    Compute the cross power spectral density (CPSD) of the signals `a` and `b`.

    Parameters
    ----------
    a : |np.ndarray|
      The first signal.
    b : |np.ndarray|
      The second signal.
    nfft : int
      The number of points in the fft.
    fs : float
      The sample rate (e.g. sample/second).
    window : {None, 1, 'hann', |np.ndarray|}
      The window to use (default: 'hann'). Valid entries are:
      - None,1               : uses a 'boxcar' or ones window.
      - 'hann'               : hanning window.
      - a length(nfft) array : use this as the window directly.

    Returns
    -------
    cpsd : |np.ndarray|
      The cross-spectral density of `a` and `b`.

    See Also
    ---------
    :func:`psd`,
    :func:`coherence`,
    :func:`cpsd`,
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
        return cpsd(a, b, nfft, fs, window=window)
    elif l[0] > l[1]:
        a, b = b, a
        l = l[::-1]
    step = [0, 0]
    step[0], nens, nfft = stepsize(l[0], nfft)
    step[1], nens, nfft = stepsize(l[1], nfft, nens=nens)
    fs = np.float64(fs)
    window = _getwindow(window, nfft)
    fft_inds = slice(1, int(nfft / 2. + 1))
    wght = 2. / (window ** 2).sum()
    pwr = fft(detrend(a[0:nfft]) * window)[fft_inds] * \
        np.conj(fft(detrend(b[0:nfft]) * window)[fft_inds])
    if nens - 1:
        for i1, i2 in zip(range(step[0], l[0] - nfft + 1, step[0]),
                          range(step[1], l[1] - nfft + 1, step[1])):
            pwr += fft(detrend(a[i1:(i1 + nfft)]) * window)[fft_inds] * \
                np.conj(fft(detrend(b[i2:(i2 + nfft)]) * window)[fft_inds])
    pwr *= wght / nens / fs
    return pwr


def cpsd(a, b, nfft, fs, window='hann', step=None):
    """
    Compute the cross power spectral density (CPSD) of the signals `a` and `b`.

    Parameters
    ----------
    a : |np.ndarray|
      The first signal.
    b : |np.ndarray|
      The second signal.
    nfft : int
      The number of points in the fft.
    fs : float
      The sample rate (e.g. sample/second).
    window : {None, 1, 'hann', |np.ndarray|}
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
    cpsd : |np.ndarray|
      The cross-spectral density of `a` and `b`.

    See also
    --------
    :func:`psd`
    :func:`coherence`
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
    step, nens, nfft = stepsize(l, nfft, step=step)
    fs = np.float64(fs)
    window = _getwindow(window, nfft)
    fft_inds = slice(1, int(nfft / 2. + 1))
    wght = 2. / (window ** 2).sum()
    s1 = fft(detrend(a[0:nfft]) * window)[fft_inds]
    if auto_psd:
        pwr = np.abs(s1) ** 2
    else:
        pwr = s1 * np.conj(fft(detrend(b[0:nfft]) * window)[fft_inds])
    if nens - 1:
        for i in range(step, l - nfft + 1, step):
            s1 = fft(detrend(a[i:(i + nfft)]) * window)[fft_inds]
            if auto_psd:
                pwr += np.abs(s1) ** 2
            else:
                pwr += s1 * \
                    np.conj(fft(detrend(b[i:(i + nfft)]) * window)[fft_inds])
    pwr *= wght / nens / fs
    return pwr


def psd(a, nfft, fs, window='hann', step=None):
    """
    Compute the power spectral density (PSD).

    This function computes the one-dimensional `n`-point PSD.

    The units of the spectra is the product of the units of `a` and
    `b`, divided by the units of fs.

    Parameters
    ----------
    a : |np.ndarray|
      The first signal, as a 1D vector
    nfft : int
      The number of points in the fft.
    fs : float
      The sample rate (e.g. sample/second).
    window : {None, 1, 'hann', |np.ndarray|}
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
    cpsd : |np.ndarray|
      The cross-spectral density of `a` and `b`.

    Notes
    -----
    Credit: This function's line of code was copied from JN's fast_psd.m 
    routine.

    See Also
    --------
    :func:`cpsd`
    :func:`coherence`
    `numpy.fft`

    """
    return np.abs(cpsd(a, a, nfft, fs, window=window, step=step))


def phase_angle(a, b, nfft, window='hann', step=None):
    """
    Compute the phase difference between signals `a` and `b`. This is the
    complimentary function to coherence and cpsd.

    Positive angles means that `b` leads `a`, i.e. this does,
    essentially:

          angle(b) - angle(a)

    This function computes one-dimensional `n`-point PSD.

    The angles are output as magnitude = 1 complex numbers (to
    simplify averaging). Therefore, use `numpy.angle` to actually
    output the angle.


    Parameters
    ----------
    a      : 1d-array_like, the signal. Currently only supports vectors.
    nfft   : The number of points in the fft.
    window : The window to use (default: 'hann'). Valid entries are:
                 None,1               : uses a 'boxcar' or ones window.
                 'hann'               : hanning window.
                 a length(nfft) array : use this as the window directly.
    step   : Use this to specify the overlap.  For example:
             -  step=nfft/2 specifies a 50% overlap.
             -  step=nfft specifies no overlap.
             -  step=2*nfft means that half the data will be skipped.

             By default, `step` is calculated to maximize data use, have
             at least 50% overlap and minimize the number of ensembles.

    Returns
    -------
    ang    : complex |np.ndarray| (unit magnitude values)

    See Also
    --------
    `numpy.fft`
    :func:`coherence`
    :func:`cpsd`

    """
    window = _getwindow(window, nfft)
    fft_inds = slice(1, int(nfft / 2. + 1))
    s1 = fft(detrend(a[0:nfft]) * window)[fft_inds]
    s2 = fft(detrend(b[0:nfft]) * window)[fft_inds]
    s1 /= np.abs(s1)
    s2 /= np.abs(s2)
    ang = s2 / s1
    l = len(a)
    step, nens, nfft = stepsize(l, nfft, step=step)
    if nens - 1:
        for i in range(step, l - nfft + 1, step):
            s1 = fft(detrend(a[i:(i + nfft)]) * window)[fft_inds]
            s1 /= np.abs(s1)
            s2 = fft(detrend(a[i:(i + nfft)]) * window)[fft_inds]
            s2 /= np.abs(s2)
            ang += s2 / s1
    ang /= nens
    return ang
