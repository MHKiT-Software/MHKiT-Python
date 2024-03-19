"""
This module provides functionality for estimating the short-term and
long-term extreme distributions of responses in a time series. It 
includes methods for analyzing peaks, block maxima, and applying 
statistical distributions to model extreme events. The module supports 
various methods for short-term extreme estimation, including peaks 
fitting with Weibull, tail fitting, peaks over threshold, and block 
maxima methods with GEV (Generalized Extreme Value) and Gumbel 
distributions. Additionally, it offers functionality to approximate 
the long-term extreme distribution by weighting short-term extremes 
across different sea states.

Functions:
- ste_peaks: Estimates the short-term extreme distribution from peaks 
    distribution using specified statistical methods.
- block_maxima: Finds the block maxima in a time-series data to be used
    in block maxima methods.
- ste_block_maxima_gev: Approximates the short-term extreme distribution 
    using the block maxima method with the GEV distribution.
- ste_block_maxima_gumbel: Approximates the short-term extreme 
    distribution using the block maxima method with the Gumbel distribution.
- ste: Alias for `short_term_extreme`, facilitating easier access to the 
    primary functionality of estimating short-term extremes.
- short_term_extreme: Core function to approximate the short-term extreme 
    distribution from a time series using chosen methods.
- full_seastate_long_term_extreme: Combines short-term extreme 
    distributions using weights to estimate the long-term extreme distribution.
"""

from typing import Union

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous

import mhkit.loads.extreme.peaks as peaks_distributions


def ste_peaks(peaks_distribution: rv_continuous, npeaks: float) -> rv_continuous:
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
    short_term_extreme: scipy.stats.rv_frozen
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

        def _cdf(self, x, *args, **kwargs):
            peaks_cdf = np.array(self.peaks.cdf(x, *args, **kwargs))
            peaks_cdf[np.isnan(peaks_cdf)] = 0.0
            if len(peaks_cdf) == 1:
                peaks_cdf = peaks_cdf[0]
            return peaks_cdf**self.npeaks

    short_term_extreme_peaks = _ShortTermExtreme(
        name="short_term_extreme", peaks_distribution=peaks_distribution, npeaks=npeaks
    )
    return short_term_extreme_peaks


def block_maxima(
    time: np.ndarray, global_peaks_data: np.ndarray, time_st: float
) -> np.ndarray:
    """
    Find the block maxima of a time-series.

    The timeseries (time, global_peaks) is divided into blocks of length t_st, and the
    maxima of each bloock is returned.

    Parameters
    ----------
    time : np.array
        Time array.
    global_peaks_data : np.array
        global peaks timeseries.
    time_st : float
        Short-term period.

    Returns
    -------
    block_max: np.array
        Block maxima (i.e. largest peak in each block).
    """
    if not isinstance(time, np.ndarray):
        raise TypeError(f"time must be of type np.ndarray. Got: {type(time)}")
    if not isinstance(global_peaks_data, np.ndarray):
        raise TypeError(
            f"global_peaks_data must be of type np.ndarray. Got: {type(global_peaks_data)}"
        )
    if not isinstance(time_st, float):
        raise TypeError(f"time_st must be of type float. Got: {type(time_st)}")

    nblock = int(time[-1] / time_st)
    block_max = np.zeros(int(nblock))
    for iblock in range(nblock):
        i_x = global_peaks_data[
            (time >= iblock * time_st) & (time < (iblock + 1) * time_st)
        ]
        block_max[iblock] = np.max(i_x)
    return block_max


def ste_block_maxima_gev(block_max):
    """
    Approximate the short-term extreme distribution using the block
    maxima method and the Generalized Extreme Value distribution.

    Parameters
    ----------
    block_max: np.array
        Block maxima (i.e. largest peak in each block).

    Returns
    -------
    short_term_extreme_rv: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(block_max, np.ndarray):
        raise TypeError(f"block_max must be of type np.ndarray. Got: {type(block_max)}")

    ste_params = stats.genextreme.fit(block_max)
    param_names = ["c", "loc", "scale"]
    ste_params = dict(zip(param_names, ste_params))
    short_term_extreme_rv = stats.genextreme(**ste_params)
    short_term_extreme_rv.params = ste_params
    return short_term_extreme_rv


def ste_block_maxima_gumbel(block_max):
    """
    Approximate the short-term extreme distribution using the block
    maxima method and the Gumbel (right) distribution.

    Parameters
    ----------
    block_max: np.array
        Block maxima (i.e. largest peak in each block).

    Returns
    -------
    ste: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(block_max, np.ndarray):
        raise TypeError(f"block_max must be of type np.ndarray. Got: {type(block_max)}")

    ste_params = stats.gumbel_r.fit(block_max)
    param_names = ["loc", "scale"]
    ste_params = dict(zip(param_names, ste_params))
    short_term_extreme_rv = stats.gumbel_r(**ste_params)
    short_term_extreme_rv.params = ste_params
    return short_term_extreme_rv


def ste(time: np.ndarray, data: np.ndarray, t_st: float, method: str) -> rv_continuous:
    """
    Alias for `short_term_extreme`.
    """
    ste_dist = short_term_extreme(time, data, t_st, method)
    return ste_dist


def short_term_extreme(
    time: np.ndarray, data: np.ndarray, t_st: float, method: str
) -> Union[rv_continuous, None]:
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
    time: np.array
        Time array.
    data: np.array
        Response timeseries.
    t_st: float
        Short-term period.
    method : string
        Method for estimating the short-term extreme distribution.

    Returns
    -------
    short_term_extreme_dist: scipy.stats.rv_frozen
            Short-term extreme distribution.
    """
    if not isinstance(time, np.ndarray):
        raise TypeError(f"time must be of type np.ndarray. Got: {type(time)}")
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")
    if not isinstance(t_st, float):
        raise TypeError(f"t_st must be of type float. Got: {type(t_st)}")
    if not isinstance(method, str):
        raise TypeError(f"method must be of type string. Got: {type(method)}")

    peaks_methods = {
        "peaks_weibull": peaks_distributions.peaks_distribution_weibull,
        "peaks_weibull_tail_fit": peaks_distributions.peaks_distribution_weibull_tail_fit,
        "peaks_over_threshold": peaks_distributions.peaks_distribution_peaks_over_threshold,
    }
    blockmaxima_methods = {
        "block_maxima_gev": ste_block_maxima_gev,
        "block_maxima_gumbel": ste_block_maxima_gumbel,
    }

    if method in peaks_methods:
        fit_peaks = peaks_methods[method]
        _, peaks = peaks_distributions.global_peaks(time, data)
        npeaks = len(peaks)
        time = time[-1] - time[0]
        nst = peaks_distributions.number_of_short_term_peaks(npeaks, time, t_st)
        peaks_dist = fit_peaks(peaks)
        short_term_extreme_dist = ste_peaks(peaks_dist, nst)
    elif method in blockmaxima_methods:
        fit_maxima = blockmaxima_methods[method]
        maxima = block_maxima(time, data, t_st)
        short_term_extreme_dist = fit_maxima(maxima)
    else:
        print("Passed `method` not found.")
    return short_term_extreme_dist


def full_seastate_long_term_extreme(short_term_extreme_dist, weights):
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
    if not isinstance(short_term_extreme_dist, list):
        raise TypeError(
            "short_term_extreme_dist must be of type list[scipy.stats.rv_frozen]."
            + f"Got: {type(short_term_extreme_dist)}"
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
            # Disabled bc not sure where/ how n is applied
            self.n = len(self.weights)  # pylint: disable=invalid-name
            super().__init__(*args, **kwargs)

        def _cdf(self, x, *args, **kwargs):
            weighted_cdf = 0.0
            for w_i, ste_i in zip(self.weights, self.ste):
                weighted_cdf += w_i * ste_i.cdf(x, *args, **kwargs)
            return weighted_cdf

    return _LongTermExtreme(
        name="long_term_extreme", weights=weights, ste=short_term_extreme_dist
    )
