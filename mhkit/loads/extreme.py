import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
from mhkit.wave.resource import frequency_moment


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
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    # eliminate zeros
    zeroMask = (data == 0)
    data[zeroMask] = 0.5 * np.min(np.abs(data))
    # zero up-crossings
    diff = np.diff(np.sign(data))
    zeroUpCrossings_mask = (diff == 2) | (diff == 1)
    zeroUpCrossings_index = np.where(zeroUpCrossings_mask)[0]
    zeroUpCrossings_index = np.append(zeroUpCrossings_index, len(data) - 1)
    # global peaks
    npeaks = len(zeroUpCrossings_index)
    peaks = np.array([])
    t_peaks = np.array([])
    for i in range(npeaks - 1):
        peak_index = np.argmax(
            data[zeroUpCrossings_index[i]:zeroUpCrossings_index[i + 1]])
        t_peaks = np.append(t_peaks, t[zeroUpCrossings_index[i] + peak_index])
        peaks = np.append(peaks, data[zeroUpCrossings_index[i] + peak_index])
    return t_peaks, peaks


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
    assert isinstance(n, int), 'n must be of type int'
    assert isinstance(t, float), 't must be of type float'
    assert isinstance(t_st, float), 't_st must be of type float'

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
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'

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
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'

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
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'
    if threshold is None:
        threshold = np.mean(x) + 1.4 * np.std(x)
    assert isinstance(threshold, float
                      ), 'threshold must be of type float'

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
    assert callable(peaks_distribution.cdf
                    ), 'peaks_distribution must be a scipy.stat distribution.'
    assert isinstance(npeaks, float), 'npeaks must be of type float'

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
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'
    assert isinstance(t_st, float), 't_st must be of type float'

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
    assert isinstance(
        block_maxima, np.ndarray), 'block_maxima must be of type np.ndarray'

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
    assert isinstance(
        block_maxima, np.ndarray), 'block_maxima must be of type np.ndarray'

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
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'x must be of type np.ndarray'
    assert isinstance(t_st, float), 't_st must be of type float'
    assert isinstance(method, str), 'method must be of type string'

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
    weights: list[floats]
        The weights from the full sea state sampling

    Returns
    -------
    ste: scipy.stats.rv_frozen
        Short-term extreme distribution.
    """
    assert isinstance(
        ste, list), 'ste must be of type list[scipy.stats.rv_frozen]'
    assert isinstance(weights, (list, np.ndarray)
                      ), 'weights must be of type list[floats]'

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


def mler_coefficients(rao, wave_spectrum, response_desired):
    """
    Calculate MLER (most likely extreme response) coefficients from a
    sea state spectrum and a response RAO.

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    wave_spectrum: pd.DataFrame
        Wave spectral density [m^2/Hz] indexed by frequency [Hz].
    response_desired: int or float
        Desired response, units should correspond to a motion RAO or
        units of force for a force RAO.

    Returns
    -------
    mler: pd.DataFrame
        DataFrame containing conditioned wave spectral amplitude
        coefficient [m^2-s], and Phase [rad] indexed by freq [Hz].
    """
    try:
        rao = np.array(rao)
    except:
        pass
    assert isinstance(rao, np.ndarray), 'rao must be of type np.ndarray'
    assert isinstance(wave_spectrum, pd.DataFrame
                      ), 'wave_spectrum must be of type pd.DataFrame'
    assert isinstance(response_desired, (int, float)
                      ), 'response_desired must be of type int or float'

    freq_hz = wave_spectrum.index.values
    # convert from Hz to rad/s
    freq = freq_hz * (2*np.pi)
    # change from Hz to rad/s
    wave_spectrum = wave_spectrum.iloc[:, 0].values / (2*np.pi)
    # get delta
    dw = (2*np.pi - 0.) / (len(freq)-1)

    spectrum_r = np.zeros(len(freq))  # [(response units)^2-s/rad]
    _s = np.zeros(len(freq))  # [m^2-s/rad]
    _a = np.zeros(len(freq))  # [m^2-s/rad]
    _coeff_a_rn = np.zeros(len(freq))  # [1/(response units)]
    _phase = np.zeros(len(freq))

    # Note: waves.A is "S" in Quon2016; 'waves' naming convention
    # matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    spectrum_r[:] = np.abs(rao)**2 * (2*wave_spectrum)

    # calculate spectral moments and other important spectral values.
    m0 = (frequency_moment(pd.Series(spectrum_r, index=freq), 0)).iloc[0, 0]
    m1 = (frequency_moment(pd.Series(spectrum_r, index=freq), 1)).iloc[0, 0]
    m2 = (frequency_moment(pd.Series(spectrum_r, index=freq), 2)).iloc[0, 0]
    wBar = m1 / m0

    # calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
    # Drummen version.  Dietz has negative of this.
    _coeff_a_rn[:] = np.abs(rao) * np.sqrt(2*wave_spectrum*dw) * \
        ((m2 - freq*m1) + wBar*(freq*m0 - m1)) / (m0*m2 - m1**2)

    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase[:] = -np.unwrap(np.angle(rao))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift, then flip sign on
    # negative Amplitudes
    _phase[_coeff_a_rn < 0] -= np.pi
    _coeff_a_rn[_coeff_a_rn < 0] *= -1

    # calculate the conditioned spectrum [m^2-s/rad]
    _s[:] = wave_spectrum * _coeff_a_rn[:]**2 * response_desired**2
    _a[:] = 2*wave_spectrum * _coeff_a_rn[:]**2 * \
        response_desired**2

    # if the response amplitude we ask for is negative, we will add
    # a pi phase shift to the phase information.  This is because
    # the sign of self.desiredRespAmp is lost in the squaring above.
    # Ordinarily this would be put into the final equation, but we
    # are shaping the wave information so that it is buried in the
    # new spectral information, S. (AP)
    if response_desired < 0:
        _phase += np.pi

    mler = pd.DataFrame(
        data={'WaveSpectrum': _s, 'Phase': _phase}, index=freq_hz)
    mler = mler.fillna(0)
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
    if not parameters == None:
        assert isinstance(parameters, dict), 'parameters must be of type dict'

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


def mler_wave_amp_normalize(wave_amp, mler, sim, k):
    """
    Function that renormalizes the incoming amplitude of the MLER wave
    to the desired peak height (peak to MSL).

    Parameters
    ----------
    wave_amp: float
        Desired wave amplitude (peak to MSL).
    mler: pd.DataFrame
        MLER coefficients generated by 'mler_coefficients' function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number.

    Returns
    -------
    mler_norm : pd.DataFrame
        MLER coefficients
    """
    try:
        k = np.array(k)
    except:
        pass
    assert isinstance(mler, pd.DataFrame), 'mler must be of type pd.DataFrame'
    assert isinstance(wave_amp, (int, float)
                      ), 'wave_amp must be of type int or float'
    assert isinstance(sim, dict), 'sim must be of type dict'
    assert isinstance(k, np.ndarray), 'k must be of type ndarray'

    freq = mler.index.values * 2*np.pi
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
    spectrum = mler['WaveSpectrum'] * rescale_fact**2

    mler_norm = pd.DataFrame(index=mler.index)
    mler_norm['WaveSpectrum'] = spectrum
    mler_norm['Phase'] = mler['Phase']

    return mler_norm


def mler_export_time_series(rao, mler, sim, k):
    """
    Generate the wave amplitude time series at X0 from the calculated
    MLER coefficients

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    mler: pd.DataFrame
        MLER coefficients dataframe generated from an MLER function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number.

    Returns
    -------
    mler_ts: pd.DataFrame
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
    assert isinstance(rao, np.ndarray), 'rao must be of type ndarray'
    assert isinstance(mler, pd.DataFrame), 'mler must be of type pd.DataFrame'
    assert isinstance(sim, dict), 'sim must be of type dict'
    assert isinstance(k, np.ndarray), 'k must be of type ndarray'

    freq = mler.index.values * 2*np.pi  # convert Hz to rad/s
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

    mler_ts = pd.DataFrame(wave_amp_time, index=sim['T'])
    mler_ts = mler_ts.rename(columns={0: 'WaveHeight', 1: 'LinearResponse'})

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
    assert callable(ppf)
    assert isinstance(return_year, (float, int))
    assert isinstance(short_term_period_hr, (float, int))

    p = 1 / (return_year * 365.25 * 24 / short_term_period_hr)

    return ppf(1 - p)
