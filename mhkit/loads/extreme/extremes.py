import numpy as np
from scipy import stats

import mhkit.loads.extreme as extreme


def ste_peaks(peaks_distribution, npeaks):
    """
    Estimate the short-term extreme distribution from the peaks
    distribution.

    Parameters
    ----------
    peaks_distribution: scipy.stats.rv_frozen
        Probability distribution of the peaks.
    npeaks : float
        Number of peaks in short term period.

    Returns
    -------
    ste: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not callable(peaks_distribution.cdf):
        raise TypeError("peaks_distribution must be a scipy.stat distribution.")
    if not isinstance(npeaks, float):
        raise TypeError(f"npeaks must be of type float. Got: {type(npeaks)}")

    class _ShortTermExtreme(stats.rv_continuous):
        def __init__(self, *args, **kwargs):
            self.peaks = kwargs.pop("peaks_distribution")
            self.npeaks = kwargs.pop("npeaks")
            super().__init__(*args, **kwargs)

        def _cdf(self, x):
            peaks_cdf = np.array(self.peaks.cdf(x))
            peaks_cdf[np.isnan(peaks_cdf)] = 0.0
            if len(peaks_cdf) == 1:
                peaks_cdf = peaks_cdf[0]
            return peaks_cdf**self.npeaks

    ste = _ShortTermExtreme(
        name="short_term_extreme", peaks_distribution=peaks_distribution, npeaks=npeaks
    )
    return ste


def block_maxima(t, x, t_st):
    """
    Find the block maxima of a time-series.

    The timeseries (t,x) is divided into blocks of length t_st, and the
    maxima of each bloock is returned.

    Parameters
    ----------
    t : np.array
        Time array.
    x : np.array
        global peaks timeseries.
    t_st : float
        Short-term period.

    Returns
    -------
    block_maxima: np.array
        Block maxima (i.e. largest peak in each block).
    """
    if not isinstance(t, np.ndarray):
        raise TypeError(f"t must be of type np.ndarray. Got: {type(t)}")
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")
    if not isinstance(t_st, float):
        raise TypeError(f"t_st must be of type float. Got: {type(t_st)}")

    nblock = int(t[-1] / t_st)
    block_maxima = np.zeros(int(nblock))
    for iblock in range(nblock):
        ix = x[(t >= iblock * t_st) & (t < (iblock + 1) * t_st)]
        block_maxima[iblock] = np.max(ix)
    return block_maxima


def ste_block_maxima_gev(block_maxima):
    """
    Approximate the short-term extreme distribution using the block
    maxima method and the Generalized Extreme Value distribution.

    Parameters
    ----------
    block_maxima: np.array
        Block maxima (i.e. largest peak in each block).

    Returns
    -------
    ste: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(block_maxima, np.ndarray):
        raise TypeError(
            f"block_maxima must be of type np.ndarray. Got: {type(block_maxima)}"
        )

    ste_params = stats.genextreme.fit(block_maxima)
    param_names = ["c", "loc", "scale"]
    ste_params = {k: v for k, v in zip(param_names, ste_params)}
    ste = stats.genextreme(**ste_params)
    ste.params = ste_params
    return ste


def ste_block_maxima_gumbel(block_maxima):
    """
    Approximate the short-term extreme distribution using the block
    maxima method and the Gumbel (right) distribution.

    Parameters
    ----------
    block_maxima: np.array
        Block maxima (i.e. largest peak in each block).

    Returns
    -------
    ste: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(block_maxima, np.ndarray):
        raise TypeError(
            f"block_maxima must be of type np.ndarray. Got: {type(block_maxima)}"
        )

    ste_params = stats.gumbel_r.fit(block_maxima)
    param_names = ["loc", "scale"]
    ste_params = {k: v for k, v in zip(param_names, ste_params)}
    ste = stats.gumbel_r(**ste_params)
    ste.params = ste_params
    return ste


def ste(t, data, t_st, method):
    """
    Alias for `short_term_extreme`.
    """
    ste = short_term_extreme(t, data, t_st, method)
    return ste


def short_term_extreme(t, data, t_st, method):
    """
    Approximate the short-term  extreme distribution from a
    timeseries of the response using chosen method.

    The availabe methods are: 'peaks_weibull', 'peaks_weibull_tail_fit',
    'peaks_over_threshold', 'block_maxima_gev', and 'block_maxima_gumbel'.
    For the block maxima methods the timeseries needs to be many times
    longer than the short-term period. For the peak-fitting methods the
    timeseries can be of arbitrary length.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Response timeseries.
    t_st: float
        Short-term period.
    method : string
        Method for estimating the short-term extreme distribution.

    Returns
    -------
    ste: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(t, np.ndarray):
        raise TypeError(f"t must be of type np.ndarray. Got: {type(t)}")
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")
    if not isinstance(t_st, float):
        raise TypeError(f"t_st must be of type float. Got: {type(t_st)}")
    if not isinstance(method, str):
        raise TypeError(f"method must be of type string. Got: {type(method)}")

    peaks_methods = {
        "peaks_weibull": extreme.peaks_distribution_weibull,
        "peaks_weibull_tail_fit": extreme.peaks_distribution_weibull_tail_fit,
        "peaks_over_threshold": extreme.peaks_distribution_peaks_over_threshold,
    }
    blockmaxima_methods = {
        "block_maxima_gev": ste_block_maxima_gev,
        "block_maxima_gumbel": ste_block_maxima_gumbel,
    }

    if method in peaks_methods.keys():
        fit_peaks = peaks_methods[method]
        _, peaks = extreme.global_peaks(t, data)
        npeaks = len(peaks)
        time = t[-1] - t[0]
        nst = extreme.number_of_short_term_peaks(npeaks, time, t_st)
        peaks_dist = fit_peaks(peaks)
        ste = ste_peaks(peaks_dist, nst)
    elif method in blockmaxima_methods.keys():
        fit_maxima = blockmaxima_methods[method]
        maxima = block_maxima(t, data, t_st)
        ste = fit_maxima(maxima)
    else:
        print("Passed `method` not found.")
    return ste


def full_seastate_long_term_extreme(ste, weights):
    """
    Return the long-term extreme distribution of a response of
    interest using the full sea state approach.

    Parameters
    ----------
    ste: list[scipy.stats.rv_frozen]
        Short-term extreme distribution of the quantity of interest for
        each sample sea state.
    weights: list, np.ndarray
        The weights from the full sea state sampling

    Returns
    -------
    ste: scipy.stats.rv_frozen
        Short-term extreme distribution.
    """
    if not isinstance(ste, list):
        raise TypeError(
            f"ste must be of type list[scipy.stats.rv_frozen]. Got: {type(ste)}"
        )
    if not isinstance(weights, (list, np.ndarray)):
        raise TypeError(
            f"weights must be of type list or np.ndarray. Got: {type(weights)}"
        )

    class _LongTermExtreme(stats.rv_continuous):
        def __init__(self, *args, **kwargs):
            weights = kwargs.pop("weights")
            # make sure weights add to 1.0
            self.weights = weights / np.sum(weights)
            self.ste = kwargs.pop("ste")
            self.n = len(self.weights)
            super().__init__(*args, **kwargs)

        def _cdf(self, x):
            f = 0.0
            for w_i, ste_i in zip(self.weights, self.ste):
                f += w_i * ste_i.cdf(x)
            return f

    return _LongTermExtreme(name="long_term_extreme", weights=weights, ste=ste)
