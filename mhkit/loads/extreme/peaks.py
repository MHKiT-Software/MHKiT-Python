"""
This module provides utilities for analyzing wave data, specifically
for identifying significant wave heights and estimating wave peak
distributions using statistical methods. 

Functions:
- _calculate_window_size: Calculates the window size for peak 
  independence using the auto-correlation function of wave peaks.
- _peaks_over_threshold: Identifies peaks over a specified 
  threshold and returns independent storm peak values adjusted by
  the threshold.
- global_peaks: Identifies global peaks in a zero-centered 
  response time-series based on consecutive zero up-crossings.
- number_of_short_term_peaks: Estimates the number of peaks within a
 specified short-term period.
- peaks_distribution_weibull: Estimates the peaks distribution by
 fitting a Weibull distribution to the peaks of the response.
- peaks_distribution_weibull_tail_fit: Estimates the peaks distribution
 using the Weibull tail fit method.
- automatic_hs_threshold: Determines the best significant wave height
 threshold for the peaks-over-threshold method.
- peaks_distribution_peaks_over_threshold: Estimates the peaks
 distribution using the peaks over threshold method by fitting a 
 generalized Pareto distribution.

References:
- Neary, V. S., S. Ahn, B. E. Seng, M. N. Allahdadi, T. Wang, Z. Yang, 
 and R. He (2020). "Characterization of Extreme Wave Conditions for 
 Wave Energy Converter Design and Project Risk Assessment.” J. Mar. 
 Sci. Eng. 2020, 8(4), 289; https://doi.org/10.3390/jmse8040289.

"""

from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats, optimize, signal
from scipy.stats import rv_continuous

from mhkit.utils import upcrossing


def _calculate_window_size(peaks: NDArray[np.float64], sampling_rate: float) -> float:
    """
    Calculate the window size for independence based on the auto-correlation function.

    Parameters
    ----------
    peaks : np.ndarray
        A NumPy array of peak values from a time series.
    sampling_rate : float
        The sampling rate of the time series in Hz (samples per second).

    Returns
    -------
    float
        The window size determined by the auto-correlation function.
    """
    nlags = int(14 * 24 / sampling_rate)
    x = peaks - np.mean(peaks)
    acf = signal.correlate(x, x, mode="full")
    lag = signal.correlation_lags(len(x), len(x), mode="full")
    idx_zero = np.argmax(lag == 0)
    positive_lag = lag[idx_zero : idx_zero + nlags + 1]
    acf_positive = acf[idx_zero : idx_zero + nlags + 1] / acf[idx_zero]

    window_size = sampling_rate * positive_lag[acf_positive < 0.5][0]
    return window_size / sampling_rate


def _peaks_over_threshold(
    peaks: NDArray[np.float64], threshold: float, sampling_rate: float
) -> List[float]:
    """
    Identifies peaks in a time series that are over a specified threshold and
    returns a list of independent storm peak values adjusted by the threshold.
    Independence is determined by a window size calculated from the auto-correlation
    function to ensure that peaks are separated by at least the duration
    corresponding to the first significant drop in auto-correlation.

    Parameters
    ----------
    peaks : np.ndarray
        A NumPy array of peak values from a time series.
    threshold : float
        The percentile threshold (0-1) to identify significant peaks.
        For example, 0.95 for the 95th percentile.
    sampling_rate : float
        The sampling rate of the time series in Hz (samples per second).

    Returns
    -------
    List[float]
        A list of peak values exceeding the specified threshold, adjusted
        for independence based on the calculated window size.

    Notes
    -----
    This function requires the global_peaks function to identify the
    maxima between consecutive zero up-crossings and uses the signal processing
    capabilities from scipy.signal for calculating the auto-correlation function.
    """
    threshold_unit = np.percentile(peaks, 100 * threshold, method="hazen")
    idx_peaks = np.arange(len(peaks))
    idx_storm_peaks, storm_peaks = global_peaks(idx_peaks, peaks - threshold_unit)
    idx_storm_peaks = idx_storm_peaks.astype(int)

    independent_storm_peaks = [storm_peaks[0]]
    idx_independent_storm_peaks = [idx_storm_peaks[0]]

    window = _calculate_window_size(peaks, sampling_rate)

    for idx in idx_storm_peaks[1:]:
        if (idx - idx_independent_storm_peaks[-1]) > window:
            idx_independent_storm_peaks.append(idx)
            independent_storm_peaks.append(peaks[idx] - threshold_unit)
        elif peaks[idx] > independent_storm_peaks[-1]:
            idx_independent_storm_peaks[-1] = idx
            independent_storm_peaks[-1] = peaks[idx] - threshold_unit

    return independent_storm_peaks


def global_peaks(t: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the global peaks of a zero-centered response time-series.

    The global peaks are the maxima between consecutive zero
    up-crossings.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Response time-series.

    Returns
    -------
    t_peaks: np.array
        Time array for peaks
    peaks: np.array
        Peak values of the response time-series
    """
    if not isinstance(t, np.ndarray):
        raise TypeError(f"t must be of type np.ndarray. Got: {type(t)}")
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # Find zero up-crossings
    inds = upcrossing(t, data)

    # We also include the final point in the dataset
    inds = np.append(inds, len(data) - 1)

    # As we want to return both the time and peak
    # values, look for the index at the peak.
    # The call to argmax gives us the index within the
    # upcrossing period. Therefore to get the index in the
    # original array we need to add on the index that
    # starts the zero crossing period, ind1.
    def find_peak_index(ind1, ind2):
        return np.argmax(data[ind1:ind2]) + ind1

    peak_inds = np.array(
        [find_peak_index(ind1, inds[i + 1]) for i, ind1 in enumerate(inds[:-1])],
        dtype=int,
    )

    return t[peak_inds], data[peak_inds]


def number_of_short_term_peaks(n: int, t: float, t_st: float) -> float:
    """
    Estimate the number of peaks in a specified period.

    Parameters
    ----------
    n : int
        Number of peaks in analyzed timeseries.
    t : float
        Length of time of analyzed timeseries.
    t_st: float
        Short-term period for which to estimate the number of peaks.

    Returns
    -------
    n_st : float
        Number of peaks in short term period.
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be of type int. Got: {type(n)}")
    if not isinstance(t, float):
        raise TypeError(f"t must be of type float. Got: {type(t)}")
    if not isinstance(t_st, float):
        raise TypeError(f"t_st must be of type float. Got: {type(t_st)}")

    return n * t_st / t


def peaks_distribution_weibull(x: NDArray[np.float_]) -> rv_continuous:
    """
    Estimate the peaks distribution by fitting a Weibull
    distribution to the peaks of the response.

    The fitted parameters can be accessed through the `params` field of
    the returned distribution.

    Parameters
    ----------
    x : NDArray[np.float_]
        Global peaks.

    Returns
    -------
    peaks: scipy.stats.rv_frozen
        Probability distribution of the peaks.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")

    # peaks distribution
    peaks_params = stats.exponweib.fit(x, f0=1, floc=0)
    param_names = ["a", "c", "loc", "scale"]
    peaks_params = {k: v for k, v in zip(param_names, peaks_params)}
    peaks = stats.exponweib(**peaks_params)
    # save the parameter info
    peaks.params = peaks_params
    return peaks


def peaks_distribution_weibull_tail_fit(x: NDArray[np.float_]) -> rv_continuous:
    """
    Estimate the peaks distribution using the Weibull tail fit
    method.

    The fitted parameters can be accessed through the `params` field of
    the returned distribution.

    Parameters
    ----------
    x : np.array
        Global peaks.

    Returns
    -------
    peaks: scipy.stats.rv_frozen
        Probability distribution of the peaks.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")

    # Initial guess for Weibull parameters
    p0 = stats.exponweib.fit(x, f0=1, floc=0)
    p0 = np.array([p0[1], p0[3]])
    # Approximate CDF
    x = np.sort(x)
    npeaks = len(x)
    F = np.zeros(npeaks)
    for i in range(npeaks):
        F[i] = i / (npeaks + 1.0)
    # Divide into seven sets & fit Weibull
    subset_shape_params = np.zeros(7)
    subset_scale_params = np.zeros(7)
    set_lim = np.arange(0.60, 0.90, 0.05)

    def weibull_cdf(x, c, s):
        return stats.exponweib(a=1, c=c, loc=0, scale=s).cdf(x)

    for local_set in range(7):
        x_set = x[(F > set_lim[local_set])]
        f_set = F[(F > set_lim[local_set])]
        # pylint: disable=W0632
        popt, _ = optimize.curve_fit(weibull_cdf, x_set, f_set, p0=p0)
        subset_shape_params[local_set] = popt[0]
        subset_scale_params[local_set] = popt[1]
    # peaks distribution
    peaks_params = [1, np.mean(subset_shape_params), 0, np.mean(subset_scale_params)]
    param_names = ["a", "c", "loc", "scale"]
    peaks_params = {k: v for k, v in zip(param_names, peaks_params)}
    peaks = stats.exponweib(**peaks_params)
    # save the parameter info
    peaks.params = peaks_params
    peaks.subset_shape_params = subset_shape_params
    peaks.subset_scale_params = subset_scale_params
    return peaks


def automatic_hs_threshold(
    peaks: NDArray[np.float_],
    sampling_rate: float,
    initial_threshold_range: Tuple[float, float, float] = (0.990, 0.995, 0.001),
    max_refinement: int = 5,
) -> Tuple[float, float]:
    """
    Find the best significant wave height threshold for the
    peaks-over-threshold method.

    This method was developed by:

    > Neary, V. S., S. Ahn, B. E. Seng, M. N. Allahdadi, T. Wang, Z. Yang and R. He (2020).
    > "Characterization of Extreme Wave Conditions for Wave Energy Converter Design and Project Risk Assessment.”
    > J. Mar. Sci. Eng. 2020, 8(4), 289; https://doi.org/10.3390/jmse8040289.

    Please cite this paper if using this method.

    After all thresholds in the initial range are evaluated, the search
    range is refined around the optimal point until either (i) there
    is minimal change from the previous refinement results, (ii) the
    number of data points become smaller than about 1 per year, or (iii)
    the maximum number of iterations is reached.

    Parameters
    ----------
    peaks: NDArray[np.float_]
        Peak values of the response time-series.
    sampling_rate: float
        Sampling rate in hours.
    initial_threshold_range: Tuple[float, float, float]
        Initial range of thresholds to search. Described as
        (min, max, step).
    max_refinement: int
        Maximum number of times to refine the search range.

    Returns
    -------
    Tuple[float, float]
        The best threshold and its corresponding unit.

    """
    if not isinstance(sampling_rate, (float, int)):
        raise TypeError(
            f"sampling_rate must be of type float or int. Got: {type(sampling_rate)}"
        )
    if not isinstance(peaks, np.ndarray):
        raise TypeError(f"peaks must be of type np.ndarray. Got: {type(peaks)}")
    if not len(initial_threshold_range) == 3:
        raise ValueError(
            f"initial_threshold_range must be length 3. Got: {len(initial_threshold_range)}"
        )
    if not isinstance(max_refinement, int):
        raise TypeError(
            f"max_refinement must be of type int. Got: {type(max_refinement)}"
        )

    range_min, range_max, range_step = initial_threshold_range
    best_threshold = -1
    years = len(peaks) / (365.25 * 24 / sampling_rate)

    for i in range(max_refinement):
        thresholds = np.arange(range_min, range_max, range_step)
        correlations = []

        for threshold in thresholds:
            distribution = stats.genpareto
            over_threshold = _peaks_over_threshold(peaks, threshold, sampling_rate)
            rate_per_year = len(over_threshold) / years
            if rate_per_year < 2:
                break
            distributions_parameters = distribution.fit(over_threshold, floc=0.0)
            _, (_, _, correlation) = stats.probplot(
                peaks, distributions_parameters, distribution, fit=True
            )
            correlations.append(correlation)

        max_i = np.argmax(correlations)
        minimal_change = np.abs(best_threshold - thresholds[max_i]) < 0.0005
        best_threshold = thresholds[max_i]
        if minimal_change and i < max_refinement - 1:
            break
        range_step /= 10
        if max_i == len(thresholds) - 1:
            range_min = thresholds[max_i - 1]
            range_max = thresholds[max_i] + 5 * range_step
        elif max_i == 0:
            range_min = thresholds[max_i] - 9 * range_step
            range_max = thresholds[max_i + 1]
        else:
            range_min = thresholds[max_i - 1]
            range_max = thresholds[max_i + 1]

    best_threshold_unit = np.percentile(peaks, 100 * best_threshold, method="hazen")
    return best_threshold, best_threshold_unit


def peaks_distribution_peaks_over_threshold(
    x: NDArray[np.float_], threshold: Optional[float] = None
) -> rv_continuous:
    """
    Estimate the peaks distribution using the peaks over threshold
    method.

    This fits a generalized Pareto distribution to all the peaks above
    the specified threshold. The distribution is only defined for values
    above the threshold and therefore cannot be used to obtain integral
    metrics such as the expected value. A typical choice of threshold is
    1.4 standard deviations above the mean. The peaks over threshold
    distribution can be accessed through the `pot` field of the returned
    peaks distribution.

    Parameters
    ----------
    x : NDArray[np.float_]
        Global peaks.
    threshold : Optional[float]
        Threshold value. Only peaks above this value will be used.
        Default value calculated as: `np.mean(x) + 1.4 * np.std(x)`

    Returns
    -------
    peaks: rv_continuous
        Probability distribution of the peaks.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")
    if threshold is None:
        threshold = np.mean(x) + 1.4 * np.std(x)
    if threshold is not None and not isinstance(threshold, float):
        raise TypeError(
            f"If specified, threshold must be of type float. Got: {type(threshold)}"
        )

    # peaks over threshold
    x = np.sort(x)
    pot = x[x > threshold] - threshold
    npeaks = len(x)
    npot = len(pot)
    # Fit a generalized Pareto
    pot_params = stats.genpareto.fit(pot, floc=0.0)
    param_names = ["c", "loc", "scale"]
    pot_params = {k: v for k, v in zip(param_names, pot_params)}
    pot = stats.genpareto(**pot_params)
    # save the parameter info
    pot.params = pot_params

    # peaks
    class _Peaks(rv_continuous):
        def __init__(
            self, pot_distribution: rv_continuous, threshold: float, *args, **kwargs
        ):
            self.pot = pot_distribution
            self.threshold = threshold
            super().__init__(*args, **kwargs)

        def _cdf(self, x: NDArray[np.float_], *args, **kwargs) -> NDArray[np.float_]:
            x = np.atleast_1d(x)
            out = np.zeros_like(x)
            out[x < self.threshold] = np.NaN
            xt = x[x >= self.threshold]
            if xt.size != 0:
                pot_ccdf = 1.0 - self.pot.cdf(xt - self.threshold, *args, **kwargs)
                prop_pot = npot / npeaks
                out[x >= self.threshold] = 1.0 - (prop_pot * pot_ccdf)
            return out

    peaks = _Peaks(name="peaks", pot_distribution=pot, threshold=threshold)
    peaks.pot = pot
    return peaks
