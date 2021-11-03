import numpy as np
from scipy import stats
from scipy import optimize


def global_peaks(t, data):
    # eliminate zeros
    zeroMask = (data == 0)
    data[zeroMask] = 0.5 * np.min(np.abs(data))
    # zero up-crossings
    diff = np.diff(np.sign(data))
    zeroUpCrossings_mask = (diff == 2) | (diff == 1)
    zeroUpCrossings_index = np.where(zeroUpCrossings_mask)[0]
    zeroUpCrossings_index = np.append(zeroUpCrossings_index, len(data) - 1)
    # global peaks
    N = len(zeroUpCrossings_index)
    peaks = np.array([])
    t_peaks = np.array([])
    for i in range(N - 1):
        peak_index = np.argmax(
            data[zeroUpCrossings_index[i]:zeroUpCrossings_index[i + 1]])
        t_peaks = np.append(t_peaks, t[zeroUpCrossings_index[i] + peak_index])
        peaks = np.append(peaks, data[zeroUpCrossings_index[i] + peak_index])
    # return
    return t_peaks, peaks


def npeaks_st(n, t, t_st):
    # Number of peaks in short-term period
    return n * t_st / t


def peaks_distribution_Weibull(x):
    # peaks distribution
    peaks_params = stats.exponweib.fit(x, f0=1, floc=0)
    param_names = ['a', 'c', 'loc', 'scale']
    peaks_params = {k: v for k,v in zip(param_names, peaks_params)}
    peaks = stats.exponweib(**peaks_params)
    # save the parameter info
    peaks.params = peaks_params
    return peaks


def peaks_distribution_WeibullTailFit(x):
    # Initial guess for Weibull parameters
    p0 = stats.exponweib.fit(x, f0=1, floc=0)
    p0 = np.array([p0[1], p0[3]])
    # Approximate CDF
    x = np.sort(x)
    N = len(x)
    F = np.zeros(N)
    for i in range(N):
        F[i] = i / (N + 1.0)
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


def peaks_distribution_peaksOverThreshold(x, threshold):
    # peaks over threshold
    x = np.sort(x)
    pot = x[(x > threshold)] - threshold
    N = len(x)
    Npot = len(pot)
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
                prop_pot = Npot/N
                out[x >= self.threshold] = 1. - (prop_pot * pot_ccdf)
            return out

    peaks = _Peaks(name="peaks", pot_distribution=pot, threshold=threshold)
    # save the peaks over threshold distribution
    peaks.pot = pot
    return peaks


def peaks_to_ste(peaks_distribution, npeaks):

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


def blockMaxima(t, x, t_st):
    nblock = int(t[-1] / t_st)
    block_maxima = np.zeros(int(nblock))
    for iblock in range(nblock):
        ix = x[(t >= iblock * t_st) & (t < (iblock+1)*t_st)]
        block_maxima[iblock] = np.max(ix)
    return block_maxima


def ste_block_maxima_GEV(block_maxima):
    ste_params = stats.genextreme.fit(block_maxima)
    param_names = ['c', 'loc', 'scale']
    ste_params = {k: v for k, v in zip(param_names, ste_params)}
    ste = stats.genextreme(**ste_params)
    ste.params = ste_params
    return ste


def ste_block_maxima_Gumbel(block_maxima):
    ste_params = stats.gumbel_r.fit(block_maxima)
    param_names = ['loc', 'scale']
    ste_params = {k: v for k, v in zip(param_names, ste_params)}
    ste = stats.gumbel_r(**ste_params)
    ste.params = ste_params
    return ste


def short_term_extreme(t, data, t_st, method):
    peaks_methods = {
        'peaksWeibull': peaks_distribution_Weibull,
        'peaksWeibullTailFit': peaks_distribution_WeibullTailFit,
        'peaksOverThreshold': peaks_distribution_peaksOverThreshold}
    blockmaxima_methods = {
        'blockMaximaGEV': ste_block_maxima_GEV,
        'blockMaximaGumbel': ste_block_maxima_Gumbel,
    }

    if method in peaks_methods.keys():
        fit_peaks = peaks_methods[method]
        _, peaks = global_peaks(t, data)
        npeaks = len(peaks)
        time = t[-1]-t[0]
        nst = npeaks_st(npeaks, time, t_st)
        peaks_dist = fit_peaks(peaks)
        ste = peaks_to_ste(peaks_dist, nst)
    elif method in blockmaxima_methods.keys():
        fit_maxima = blockmaxima_methods[method]
        maxima = blockMaxima(t, data, t_st)
        ste = fit_maxima(maxima)
    return ste
