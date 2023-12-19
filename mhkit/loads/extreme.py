import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats, optimize, signal
from mhkit.wave.resource import frequency_moment
from mhkit.utils import upcrossing, custom

def _peaks_over_threshold(peaks, threshold, sampling_rate):
    threshold_unit = np.percentile(peaks, 100*threshold, method='hazen')
    idx_peaks = np.arange(len(peaks))
    idx_storm_peaks, storm_peaks = global_peaks(
        idx_peaks, peaks-threshold_unit)
    idx_storm_peaks = idx_storm_peaks.astype(int)

    # Two storms that are close enough (within specified window) are
    # considered the same storm, to ensure independence.
    independent_storm_peaks = [storm_peaks[0],]
    idx_independent_storm_peaks = [idx_storm_peaks[0],]
    # check first 14 days to determine window size
    nlags = int(14 * 24 / sampling_rate)
    x = peaks - np.mean(peaks)
    acf = signal.correlate(x, x, mode="full")
    lag = signal.correlation_lags(len(x), len(x), mode="full")
    idx_zero = np.argmax(lag==0)
    positive_lag = lag[(idx_zero):(idx_zero+nlags+1)]
    acf_positive = acf[(idx_zero):(idx_zero+nlags+1)] / acf[idx_zero]

    window_size = sampling_rate * positive_lag[acf_positive<0.5][0]
    # window size in "observations" instead of "hours" between peaks.
    window = window_size / sampling_rate
    # keep only independent storm peaks
    for idx in idx_storm_peaks[1:]:
        if (idx - idx_independent_storm_peaks[-1]) > window:
            idx_independent_storm_peaks.append(idx)
            independent_storm_peaks.append(peaks[idx]-threshold_unit)
        elif peaks[idx] > independent_storm_peaks[-1]:
            idx_independent_storm_peaks[-1] = idx
            independent_storm_peaks[-1] = peaks[idx]-threshold_unit

    return independent_storm_peaks


def global_peaks(t, data):
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
        raise TypeError(f't must be of type np.ndarray. Got: {type(t)}')
    if not isinstance(data, np.ndarray):
        raise TypeError(f'data must be of type np.ndarray. Got: {type(data)}')

    # Find zero up-crossings
    inds = upcrossing(t, data)

    # We also include the final point in the dataset
    inds = np.append(inds, len(data)-1)

    # As we want to return both the time and peak
    # values, look for the index at the peak.
    # The call to argmax gives us the index within the
    # upcrossing period. Therefore to get the index in the
    # original array we need to add on the index that
    # starts the zero crossing period, ind1.
    func = lambda ind1, ind2: np.argmax(data[ind1:ind2]) + ind1

    peak_inds = np.array(custom(t, data, func, inds), dtype=int)
    
    return t[peak_inds], data[peak_inds]


def number_of_short_term_peaks(n, t, t_st):
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
        raise TypeError(f'n must be of type int. Got: {type(n)}')
    if not isinstance(t, float):
        raise TypeError(f't must be of type float. Got: {type(t)}')
    if not isinstance(t_st, float):
        raise TypeError(f't_st must be of type float. Got: {type(t_st)}')

    return n * t_st / t


def peaks_distribution_weibull(x):
    """
    Estimate the peaks distribution by fitting a Weibull
    distribution to the peaks of the response.

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
        raise TypeError(f'x must be of type np.ndarray. Got: {type(x)}')

    # peaks distribution
    peaks_params = stats.exponweib.fit(x, f0=1, floc=0)
    param_names = ['a', 'c', 'loc', 'scale']
    peaks_params = {k: v for k, v in zip(param_names, peaks_params)}
    peaks = stats.exponweib(**peaks_params)
    # save the parameter info
    peaks.params = peaks_params
    return peaks


def peaks_distribution_weibull_tail_fit(x):
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
        raise TypeError(f'x must be of type np.ndarray. Got: {type(x)}')

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
    setLim = np.arange(0.60, 0.90, 0.05)
    func = lambda x, c, s: stats.exponweib(a=1, c=c, loc=0, scale=s).cdf(x)
    for set in range(7):
        xset = x[(F > setLim[set])]
        Fset = F[(F > setLim[set])]
        popt, _ = optimize.curve_fit(func, xset, Fset, p0=p0)
        subset_shape_params[set] = popt[0]
        subset_scale_params[set] = popt[1]
    # peaks distribution
    peaks_params = [1, np.mean(subset_shape_params), 0,
                    np.mean(subset_scale_params)]
    param_names = ['a', 'c', 'loc', 'scale']
    peaks_params = {k: v for k, v in zip(param_names, peaks_params)}
    peaks = stats.exponweib(**peaks_params)
    # save the parameter info
    peaks.params = peaks_params
    peaks.subset_shape_params = subset_shape_params
    peaks.subset_scale_params = subset_scale_params
    return peaks


def automatic_hs_threshold(
    peaks,
    sampling_rate,
    initial_threshold_range = (0.990, 0.995, 0.001),
    max_refinement=5
):
    """
    Find the best significant wave height threshold for the
    peaks-over-threshold method.

    This method was developed by:

    > Neary, V. S., S. Ahn, B. E. Seng, M. N. Allahdadi, T. Wang, Z. Yang and R. He (2020).
    > "Characterization of Extreme Wave Conditions for Wave Energy Converter Design and Project Risk Assessment.”
    > J. Mar. Sci. Eng. 2020, 8(4), 289; https://doi.org/10.3390/jmse8040289.

    please cite this paper if using this method.

    After all thresholds in the initial range are evaluated, the search
    range is refined around the optimal point until either (i) there
    is minimal change from the previous refinement results, (ii) the
    number of data points become smaller than about 1 per year, or (iii)
    the maximum number of iterations is reached.

    Parameters
    ----------
    peaks: np.array
        Peak values of the response time-series
    sampling_rate: float
        Sampling rate in hours.
    initial_threshold_range: tuple
        Initial range of thresholds to search. Described as
        (min, max, step).
    max_refinement: int
        Maximum number of times to refine the search range.

    Returns
    -------
    best_threshold: float
        Threshold that results in the best correlation.
    """
    if not isinstance(sampling_rate, (float, int)):
        raise TypeError(
            f'sampling_rate must be of type float or int. Got: {type(sampling_rate)}')
    if not isinstance(peaks, np.ndarray):
        raise TypeError(
            f'peaks must be of type np.ndarray. Got: {type(peaks)}')
    if not len(initial_threshold_range) == 3:
        raise ValueError(
            f'initial_threshold_range must be length 3. Got: {len(initial_threshold_range)}')
    if not isinstance(max_refinement, int):
        raise TypeError(
            f'max_refinement must be of type int. Got: {type(max_refinement)}')

    range_min, range_max, range_step = initial_threshold_range
    best_threshold = -1
    years = len(peaks)/(365.25*24/sampling_rate)

    for i in range(max_refinement):
        thresholds = np.arange(range_min, range_max, range_step)
        correlations = []

        for threshold in thresholds:
            distribution = stats.genpareto
            over_threshold = _peaks_over_threshold(
                peaks, threshold, sampling_rate)
            rate_per_year = len(over_threshold) / years
            if rate_per_year < 2:
                break
            distributions_parameters = distribution.fit(
                over_threshold, floc=0.)
            _, (_, _, correlation) = stats.probplot(
                peaks, distributions_parameters, distribution, fit=True)
            correlations.append(correlation)

        max_i = np.argmax(correlations)
        minimal_change = np.abs(best_threshold - thresholds[max_i]) < 0.0005
        best_threshold = thresholds[max_i]
        if minimal_change and i<max_refinement-1:
            break
        range_step /= 10
        if max_i == len(thresholds)-1:
            range_min = thresholds[max_i - 1]
            range_max = thresholds[max_i] + 5*range_step
        elif max_i == 0:
            range_min = thresholds[max_i] - 9*range_step
            range_max = thresholds[max_i + 1]
        else:
            range_min = thresholds[max_i-1]
            range_max = thresholds[max_i+1]

    best_threshold_unit = np.percentile(peaks, 100*best_threshold, method='hazen')
    return best_threshold, best_threshold_unit


def peaks_distribution_peaks_over_threshold(x, threshold=None):
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
    x : np.array
        Global peaks.
    threshold : float
        Threshold value. Only peaks above this value will be used.
        Default value calculated as: `np.mean(x) + 1.4 * np.std(x)`

    Returns
    -------
    peaks: scipy.stats.rv_frozen
        Probability distribution of the peaks.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f'x must be of type np.ndarray. Got: {type(x)}')
    if threshold is None:
        threshold = np.mean(x) + 1.4 * np.std(x)
    if not isinstance(threshold, float):
        raise TypeError(
            f'If specified, threshold must be of type float. Got: {type(threshold)}')

    # peaks over threshold
    x = np.sort(x)
    pot = x[(x > threshold)] - threshold
    npeaks = len(x)
    npot = len(pot)
    # Fit a generalized Pareto
    pot_params = stats.genpareto.fit(pot, floc=0.)
    param_names = ['c', 'loc', 'scale']
    pot_params = {k: v for k, v in zip(param_names, pot_params)}
    pot = stats.genpareto(**pot_params)
    # save the parameter info
    pot.params = pot_params

    # peaks
    class _Peaks(stats.rv_continuous):

        def __init__(self, *args, **kwargs):
            self.pot = kwargs.pop('pot_distribution')
            self.threshold = kwargs.pop('threshold')
            super().__init__(*args, **kwargs)

        def _cdf(self, x):
            x = np.atleast_1d(np.array(x))
            out = np.zeros(x.shape)
            out[x < self.threshold] = np.NaN
            xt = x[x >= self.threshold]
            if xt.size != 0:
                pot_ccdf = 1. - self.pot.cdf(xt-self.threshold)
                prop_pot = npot/npeaks
                out[x >= self.threshold] = 1. - (prop_pot * pot_ccdf)
            return out

    peaks = _Peaks(name="peaks", pot_distribution=pot, threshold=threshold)
    # save the peaks over threshold distribution
    peaks.pot = pot
    return peaks


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
        raise TypeError('peaks_distribution must be a scipy.stat distribution.')
    if not isinstance(npeaks, float):
        raise TypeError(f'npeaks must be of type float. Got: {type(npeaks)}')

    class _ShortTermExtreme(stats.rv_continuous):

        def __init__(self, *args, **kwargs):
            self.peaks = kwargs.pop('peaks_distribution')
            self.npeaks = kwargs.pop('npeaks')
            super().__init__(*args, **kwargs)

        def _cdf(self, x):
            peaks_cdf = np.array(self.peaks.cdf(x))
            peaks_cdf[np.isnan(peaks_cdf)] = 0.0
            if len(peaks_cdf) == 1:
                peaks_cdf = peaks_cdf[0]
            return peaks_cdf ** self.npeaks

    ste = _ShortTermExtreme(name="short_term_extreme",
                            peaks_distribution=peaks_distribution,
                            npeaks=npeaks)
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
        raise TypeError(f't must be of type np.ndarray. Got: {type(t)}')
    if not isinstance(x, np.ndarray):
        raise TypeError(f'x must be of type np.ndarray. Got: {type(x)}')
    if not isinstance(t_st, float):
        raise TypeError(f't_st must be of type float. Got: {type(t_st)}')

    nblock = int(t[-1] / t_st)
    block_maxima = np.zeros(int(nblock))
    for iblock in range(nblock):
        ix = x[(t >= iblock * t_st) & (t < (iblock+1)*t_st)]
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
            f'block_maxima must be of type np.ndarray. Got: {type(block_maxima)}')

    ste_params = stats.genextreme.fit(block_maxima)
    param_names = ['c', 'loc', 'scale']
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
            f'block_maxima must be of type np.ndarray. Got: {type(block_maxima)}')

    ste_params = stats.gumbel_r.fit(block_maxima)
    param_names = ['loc', 'scale']
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
        raise TypeError(f't must be of type np.ndarray. Got: {type(t)}')
    if not isinstance(data, np.ndarray):
        raise TypeError(f'data must be of type np.ndarray. Got: {type(data)}')
    if not isinstance(t_st, float):
        raise TypeError(f't_st must be of type float. Got: {type(t_st)}')
    if not isinstance(method, str):
        raise TypeError(f'method must be of type string. Got: {type(method)}')

    peaks_methods = {
        'peaks_weibull': peaks_distribution_weibull,
        'peaks_weibull_tail_fit': peaks_distribution_weibull_tail_fit,
        'peaks_over_threshold': peaks_distribution_peaks_over_threshold}
    blockmaxima_methods = {
        'block_maxima_gev': ste_block_maxima_gev,
        'block_maxima_gumbel': ste_block_maxima_gumbel,
    }

    if method in peaks_methods.keys():
        fit_peaks = peaks_methods[method]
        _, peaks = global_peaks(t, data)
        npeaks = len(peaks)
        time = t[-1]-t[0]
        nst = number_of_short_term_peaks(npeaks, time, t_st)
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
            f'ste must be of type list[scipy.stats.rv_frozen]. Got: {type(ste)}')
    if not isinstance(weights, (list, np.ndarray)):
        raise TypeError(
            f'weights must be of type list or np.ndarray. Got: {type(weights)}')

    class _LongTermExtreme(stats.rv_continuous):

        def __init__(self, *args, **kwargs):
            weights = kwargs.pop('weights')
            # make sure weights add to 1.0
            self.weights = weights / np.sum(weights)
            self.ste = kwargs.pop('ste')
            self.n = len(self.weights)
            super().__init__(*args, **kwargs)

        def _cdf(self, x):
            f = 0.0
            for w_i, ste_i in zip(self.weights, self.ste):
                f += w_i * ste_i.cdf(x)
            return f

    return _LongTermExtreme(name="long_term_extreme", weights=weights, ste=ste)


def mler_coefficients(rao, wave_spectrum, response_desired, time_dimension="", to_pandas=True):
    """
    Calculate MLER (most likely extreme response) coefficients from a
    sea state spectrum and a response RAO.

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    wave_spectrum: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave spectral density [m^2/Hz] indexed by frequency [Hz]. 
        DataFrame and Dataset inputs should only have one data variable
    response_desired: int or float
        Desired response, units should correspond to a motion RAO or
        units of force for a force RAO.
    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler: pandas DataFrame or xarray Dataset
        DataFrame containing conditioned wave spectral amplitude
        coefficient [m^2-s], and Phase [rad] indexed by freq [Hz].
    """
    try:
        rao = np.array(rao)
    except:
        pass
    
    if not isinstance(rao, np.ndarray):
        raise TypeError(
            f'rao must be of type np.ndarray. Got: {type(rao)}')
    if not isinstance(wave_spectrum, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            f'wave_spectrum must be of type pd.Series, pd.DataFrame, xr.DataArray, or xr.Dataset. Got: {type(wave_spectrum)}')
    if not isinstance(response_desired, (int, float)):
        raise TypeError(
            f'response_desired must be of type int or float. Got: {type(response_desired)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    # Convert input to xarray DataArray
    if isinstance(wave_spectrum, (pd.Series, pd.DataFrame)):
        wave_spectrum = wave_spectrum.squeeze().to_xarray()
    
    if isinstance(wave_spectrum, xr.Dataset):
        if len(wave_spectrum.data_vars) > 1:
            raise ValueError(
                f'wave_spectrum can only contain one variable. Got {list(wave_spectrum.data_vars)}.')
        wave_spectrum = wave_spectrum.to_array()

    if time_dimension == "":
        time_dimension = list(wave_spectrum.coords)[0]

    # convert from Hz to rad/s
    freq_hz = wave_spectrum.coords[time_dimension].values
    freq = freq_hz * (2*np.pi)
    wave_spectrum = wave_spectrum.to_numpy() / (2*np.pi)

    # get frequency step
    dw = 2.0*np.pi / (len(freq)-1)

    # Note: waves.A is "S" in Quon2016; 'waves' naming convention
    # matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    spectrum_r = np.abs(rao)**2 * (2*wave_spectrum)

    # calculate spectral moments and other important spectral values.
    m0 = (frequency_moment(pd.Series(spectrum_r, index=freq), 0)).iloc[0, 0]
    m1 = (frequency_moment(pd.Series(spectrum_r, index=freq), 1)).iloc[0, 0]
    m2 = (frequency_moment(pd.Series(spectrum_r, index=freq), 2)).iloc[0, 0]
    wBar = m1 / m0

    # calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
    # Drummen version.  Dietz has negative of this.
    _coeff_a_rn = np.abs(rao) * np.sqrt(2*wave_spectrum*dw) * \
        ((m2 - freq*m1) + wBar*(freq*m0 - m1)) / (m0*m2 - m1**2)

    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase = -np.unwrap(np.angle(rao))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift, then flip sign on
    # negative Amplitudes
    _phase[_coeff_a_rn < 0] -= np.pi
    _coeff_a_rn[_coeff_a_rn < 0] *= -1

    # calculate the conditioned spectrum [m^2-s/rad]
    _s = wave_spectrum * _coeff_a_rn**2 * response_desired**2
    _a = 2*wave_spectrum * _coeff_a_rn**2 * response_desired**2

    # if the response amplitude we ask for is negative, we will add
    # a pi phase shift to the phase information.  This is because
    # the sign of self.desiredRespAmp is lost in the squaring above.
    # Ordinarily this would be put into the final equation, but we
    # are shaping the wave information so that it is buried in the
    # new spectral information, S. (AP)
    if response_desired < 0:
        _phase += np.pi

    mler = xr.Dataset(data_vars = {'WaveSpectrum': (['frequency'], _s),
                                  'Phase': (['frequency'], _phase)},
                      coords = {'frequency': freq_hz}
                      )
    mler.fillna(0)

    if to_pandas:
        mler = mler.to_pandas()

    return mler


def mler_simulation(parameters=None):
    """
    Define the simulation parameters that are used in various MLER
    functionalities.

    See `extreme_response_contour_example.ipynb` example for how this is
    useful. If no input is given, then default values are returned.

    Parameters
    ----------
    parameters: dict (optional)
        Simulation parameters.
        Keys:
        -----
        'startTime': starting time [s]
        'endTime': ending time [s]
        'dT': time-step size [s]
        'T0': time of maximum event [s]
        'startx': start of simulation space [m]
        'endX': end of simulation space [m]
        'dX': horizontal spacing [m]
        'X': position of maximum event [m]

    Returns
    -------
    sim: dict
        Simulation parameters including spatial and time calculated
        arrays.
    """
    if not isinstance(parameters, (type(None), dict)):
        raise TypeError(
            f'If specified, parameters must be of type dict. Got: {type(parameters)}')

    sim = {}

    if parameters == None:
        sim['startTime'] = -150.0  # [s] Starting time
        sim['endTime'] = 150.0  # [s] Ending time
        sim['dT'] = 1.0  # [s] Time-step size
        sim['T0'] = 0.0  # [s] Time of maximum event

        sim['startX'] = -300.0  # [m] Start of simulation space
        sim['endX'] = 300.0  # [m] End of simulation space
        sim['dX'] = 1.0  # [m] Horiontal spacing
        sim['X0'] = 0.0  # [m] Position of maximum event
    else:
        sim = parameters

    # maximum timestep index
    sim['maxIT'] = int(
        np.ceil((sim['endTime'] - sim['startTime'])/sim['dT'] + 1))
    sim['T'] = np.linspace(sim['startTime'], sim['endTime'], sim['maxIT'])

    sim['maxIX'] = int(np.ceil((sim['endX'] - sim['startX'])/sim['dX'] + 1))
    sim['X'] = np.linspace(sim['startX'], sim['endX'], sim['maxIX'])

    return sim


def mler_wave_amp_normalize(wave_amp, mler, sim, k, time_dimension="", to_pandas=True):
    """
    Function that renormalizes the incoming amplitude of the MLER wave
    to the desired peak height (peak to MSL).

    Parameters
    ----------
    wave_amp: float
        Desired wave amplitude (peak to MSL).
    mler: pandas DataFrame or xarray Dataset
        MLER coefficients generated by 'mler_coefficients' function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number
    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler_norm : pandas DataFrame or xarray Dataset
        MLER coefficients
    """
    try:
        k = np.array(k)
    except:
        pass
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f'mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}')
    if not isinstance(wave_amp, (int, float)):
        raise TypeError(
            f'wave_amp must be of type int or float. Got: {type(wave_amp)}')
    if not isinstance(sim, dict):
        raise TypeError(
            f'sim must be of type dict. Got: {type(sim)}')
    if not isinstance(k, np.ndarray):
        raise TypeError(
            f'k must be of type ndarray. Got: {type(k)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    # If input is pandas, convert to xarray
    if isinstance(mler,pd.DataFrame):
        mler = mler.to_xarray()

    if time_dimension == "":
        time_dimension = list(mler.coords)[0]
    freq = mler.coords[time_dimension].values * 2*np.pi
    dw = (max(freq) - min(freq)) / (len(freq)-1)  # get delta

    wave_amp_time = np.zeros((sim['maxIX'], sim['maxIT']))
    for ix, x in enumerate(sim['X']):
        for it, t in enumerate(sim['T']):
            # conditioned wave
            wave_amp_time[ix, it] = np.sum(
                np.sqrt(2*mler['WaveSpectrum']*dw) *
                np.cos(freq*(t-sim['T0']) - k*(x-sim['X0']) + mler['Phase'])
            )

    tmp_max_amp = np.max(np.abs(wave_amp_time))

    # renormalization of wave amplitudes
    rescale_fact = np.abs(wave_amp) / np.abs(tmp_max_amp)

    # rescale the wave spectral amplitude coefficients
    mler_norm = mler['WaveSpectrum'] * rescale_fact**2
    mler_norm = mler_norm.to_dataset()
    mler_norm = mler_norm.assign({'Phase': (time_dimension, mler['Phase'].data)})

    if to_pandas:
        mler_norm = mler_norm.to_pandas()

    return mler_norm


def mler_export_time_series(rao, mler, sim, k, time_dimension="", to_pandas=True):
    """
    Generate the wave amplitude time series at X0 from the calculated
    MLER coefficients

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    mler: pandas DataFrame or xarray Dataset
        MLER coefficients dataframe generated from an MLER function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number.
    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler_ts: pandas DataFrame or xarray Dataset
        Time series of wave height [m] and linear response [*] indexed
        by time [s].

    """
    try:
        rao = np.array(rao)
    except:
        pass
    try:
        k = np.array(k)
    except:
        pass
    if not isinstance(rao, np.ndarray):
        raise TypeError(
            f'rao must be of type ndarray. Got: {type(rao)}')
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f'mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}')
    if not isinstance(sim, dict):
        raise TypeError(
            f'sim must be of type dict. Got: {type(sim)}')
    if not isinstance(k, np.ndarray):
        raise TypeError(
            f'k must be of type ndarray. Got: {type(k)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    # If input is pandas, convert to xarray
    if isinstance(mler,pd.DataFrame):
        mler = mler.to_xarray()

    if time_dimension == "":
        time_dimension = list(mler.coords)[0]
    freq = mler.coords[time_dimension].values * 2*np.pi
    dw = (max(freq) - min(freq)) / (len(freq)-1)  # get delta

    # calculate the series
    wave_amp_time = np.zeros((sim['maxIT'], 2))
    xi = sim['X0']
    for i, ti in enumerate(sim['T']):
        # conditioned wave
        wave_amp_time[i, 0] = np.sum(
            np.sqrt(2*mler['WaveSpectrum']*dw) *
            np.cos(freq*(ti-sim['T0']) + mler['Phase'] - k*(xi-sim['X0']))
        )
        # Response calculation
        wave_amp_time[i, 1] = np.sum(
            np.sqrt(2*mler['WaveSpectrum']*dw) * np.abs(rao) *
            np.cos(freq*(ti-sim['T0']) - k*(xi-sim['X0']))
        )

    mler_ts = xr.Dataset(data_vars = {'WaveHeight': (['time'], wave_amp_time[:,0]),
                                      'LinearResponse': (['time'], wave_amp_time[:,1])},
                         coords = {'time': sim['T']})

    if to_pandas:
        mler_ts = mler_ts.to_pandas()

    return mler_ts


def return_year_value(ppf, return_year, short_term_period_hr):
    """
    Calculate the value from a given distribution corresponding to a particular
    return year.

    Parameters
    ----------
    ppf: callable function of 1 argument
        Percentage Point Function (inverse CDF) of short term distribution.
    return_year: int, float
        Return period in years.
    short_term_period_hr: int, float
        Short term period the distribution is created from in hours.

    Returns
    -------
    value: float
        The value corresponding to the return period from the distribution.
    """
    if not callable(ppf):
        raise TypeError('ppf must be a callable Percentage Point Function')
    if not isinstance(return_year, (float, int)):
        raise TypeError(
            f'return_year must be of type float or int. Got: {type(return_year)}')
    if not isinstance(short_term_period_hr, (float, int)):
        raise TypeError(
            f'short_term_period_hr must be of type float or int. Got: {type(short_term_period_hr)}')

    p = 1 / (return_year * 365.25 * 24 / short_term_period_hr)

    return ppf(1 - p)
