import numpy as np
import pandas as pd
import xarray 
import types
from scipy.stats import binned_statistic_2d as _binned_statistic_2d
from mhkit import wave
import matplotlib.pylab as plt
from os.path import join

def capture_length(P, J):
    """
    Calculates the capture length (often called capture width).

    Parameters
    ------------
    P: numpy array or pandas Series
        Power [W]
    J: numpy array or pandas Series
        Omnidirectional wave energy flux [W/m]

    Returns
    ---------
    L: numpy array or pandas Series
        Capture length [m]
    """
    assert isinstance(P, (np.ndarray, pd.Series)), 'P must be of type np.ndarray or pd.Series'
    assert isinstance(J, (np.ndarray, pd.Series)), 'J must be of type np.ndarray or pd.Series'

    L = P/J

    return L


def statistics(X):
    """
    Calculates statistics, including count, mean, standard
    deviation (std), min, percentiles (25%, 50%, 75%), and max.

    Note that std uses a degree of freedom of 1 in accordance with
    IEC/TS 62600-100.

    Parameters
    ------------
    X: numpy array or pandas Series
        Data

    Returns
    ---------
    stats: pandas Series
        Statistics
    """
    assert isinstance(X, (np.ndarray, pd.Series)), 'X must be of type np.ndarray or pd.Series'

    stats = pd.Series(X).describe()
    stats['std'] = _std_ddof1(X)

    return stats


def _std_ddof1(a):
    # Standard deviation with degree of freedom equal to 1
    if len(a) == 0:
        return np.nan
    elif len(a) == 1:
        return 0
    else:
        return np.std(a, ddof=1)


def _performance_matrix(X, Y, Z, statistic, x_centers, y_centers):
    # General performance matrix function

    # Convert bin centers to edges
    xi = [np.mean([x_centers[i], x_centers[i+1]]) for i in range(len(x_centers)-1)]
    xi.insert(0,-np.inf)
    xi.append(np.inf)

    yi = [np.mean([y_centers[i], y_centers[i+1]]) for i in range(len(y_centers)-1)]
    yi.insert(0,-np.inf)
    yi.append(np.inf)

    # Override standard deviation with degree of freedom equal to 1
    if statistic == 'std':
        statistic = _std_ddof1

    # Provide function to compute frequency
    def _frequency(a):
        return len(a)/len(Z)
    if statistic == 'frequency':
        statistic = _frequency

    zi, x_edge, y_edge, binnumber = _binned_statistic_2d(X, Y, Z, statistic,
                        bins=[xi,yi], expand_binnumbers=False)

    M = pd.DataFrame(zi, index=x_centers, columns=y_centers)

    return M


def capture_length_matrix(Hm0, Te, L, statistic, Hm0_bins, Te_bins):
    """
    Generates a capture length matrix for a given statistic

    Note that IEC/TS 62600-100 requires capture length matrices for
    the mean, std, count, min, and max.

    Parameters
    ------------
    Hm0: numpy array or pandas Series
        Significant wave height from spectra [m]
    Te: numpy array or pandas Series
        Energy period from spectra [s]
    L : numpy array or pandas Series
        Capture length [m]
    statistic: string
        Statistic for each bin, options include: 'mean', 'std', 'median',
        'count', 'sum', 'min', 'max', and 'frequency'.  Note that 'std' uses
        a degree of freedom of 1 in accordance with IEC/TS 62600-100.
    Hm0_bins: numpy array
        Bin centers for Hm0 [m]
    Te_bins: numpy array
        Bin centers for Te [s]

    Returns
    ---------
    LM: pandas DataFrames
        Capture length matrix with index equal to Hm0_bins and columns
        equal to Te_bins

    """
    assert isinstance(Hm0, (np.ndarray, pd.Series)), 'Hm0 must be of type np.ndarray or pd.Series'
    assert isinstance(Te, (np.ndarray, pd.Series)), 'Te must be of type np.ndarray or pd.Series'
    assert isinstance(L, (np.ndarray, pd.Series)), 'L must be of type np.ndarray or pd.Series'
    assert isinstance(statistic, (str, types.FunctionType)), 'statistic must be of type str or callable'
    assert isinstance(Hm0_bins, np.ndarray), 'Hm0_bins must be of type np.ndarray'
    assert isinstance(Te_bins, np.ndarray), 'Te_bins must be of type np.ndarray'

    LM = _performance_matrix(Hm0, Te, L, statistic, Hm0_bins, Te_bins)

    return LM


def wave_energy_flux_matrix(Hm0, Te, J, statistic, Hm0_bins, Te_bins):
    """
    Generates a wave energy flux matrix for a given statistic

    Parameters
    ------------
    Hm0: numpy array or pandas Series
        Significant wave height from spectra [m]
    Te: numpy array or pandas Series
        Energy period from spectra [s]
    J : numpy array or pandas Series
        Wave energy flux from spectra [W/m]
    statistic: string
        Statistic for each bin, options include: 'mean', 'std', 'median',
        'count', 'sum', 'min', 'max', and 'frequency'.  Note that 'std' uses a degree of freedom
        of 1 in accordance of IEC/TS 62600-100.
    Hm0_bins: numpy array
        Bin centers for Hm0 [m]
    Te_bins: numpy array
        Bin centers for Te [s]

    Returns
    ---------
    JM: pandas DataFrames
        Wave energy flux matrix with index equal to Hm0_bins and columns
        equal to Te_bins

    """
    assert isinstance(Hm0, (np.ndarray, pd.Series)), 'Hm0 must be of type np.ndarray or pd.Series'
    assert isinstance(Te, (np.ndarray, pd.Series)), 'Te must be of type np.ndarray or pd.Series'
    assert isinstance(J, (np.ndarray, pd.Series)), 'J must be of type np.ndarray or pd.Series'
    assert isinstance(statistic, (str, callable)), 'statistic must be of type str or callable'
    assert isinstance(Hm0_bins, np.ndarray), 'Hm0_bins must be of type np.ndarray'
    assert isinstance(Te_bins, np.ndarray), 'Te_bins must be of type np.ndarray'

    JM = _performance_matrix(Hm0, Te, J, statistic, Hm0_bins, Te_bins)

    return JM

def power_matrix(LM, JM):
    """
    Generates a power matrix from a capture length matrix and wave energy
    flux matrix

    Parameters
    ------------
    LM: pandas DataFrame
        Capture length matrix
    JM: pandas DataFrame
        Wave energy flux matrix

    Returns
    ---------
    PM: pandas DataFrames
        Power matrix

    """
    assert isinstance(LM, pd.DataFrame), 'LM must be of type pd.DataFrame'
    assert isinstance(JM, pd.DataFrame), 'JM must be of type pd.DataFrame'

    PM = LM*JM

    return PM

def mean_annual_energy_production_timeseries(L, J):
    """
    Calculates mean annual energy production (MAEP) from time-series

    Parameters
    ------------
    L: numpy array or pandas Series
        Capture length
    J: numpy array or pandas Series
        Wave energy flux

    Returns
    ---------
    maep: float
        Mean annual energy production

    """
    assert isinstance(L, (np.ndarray, pd.Series)), 'L must be of type np.ndarray or pd.Series'
    assert isinstance(J, (np.ndarray, pd.Series)), 'J must be of type np.ndarray or pd.Series'

    T = 8766 # Average length of a year (h)
    n = len(L)

    maep = T/n * np.sum(L * J)

    return maep

def mean_annual_energy_production_matrix(LM, JM, frequency):
    """
    Calculates mean annual energy production (MAEP) from matrix data
    along with data frequency in each bin

    Parameters
    ------------
    LM: pandas DataFrame
        Capture length
    JM: pandas DataFrame
        Wave energy flux
    frequency: pandas DataFrame
        Data frequency for each bin

    Returns
    ---------
    maep: float
        Mean annual energy production

    """
    assert isinstance(LM, pd.DataFrame), 'LM must be of type pd.DataFrame'
    assert isinstance(JM, pd.DataFrame), 'JM must be of type pd.DataFrame'
    assert isinstance(frequency, pd.DataFrame), 'frequency must be of type pd.DataFrame'
    assert LM.shape == JM.shape == frequency.shape, 'LM, JM, and frequency must be of the same size'
    #assert frequency.sum().sum() == 1

    T = 8766 # Average length of a year (h)
    maep = T * np.nansum(LM * JM * frequency)

    return maep

def power_performance_workflow(S, h, P, statistic, frequency_bins=None, deep=False, rho=1205, g=9.80665, ratio=2, show_values=False, savepath=""):
    """
    High-level function to compute power performance quantities of
    interest following IEC TS 62600-100 for given wave spectra.

    Parameters
    ------------
    S: pandas DataFrame or Series
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    h: float
        Water depth [m]
    P: numpy array or pandas Series
        Power [W]
    statistic: string or list of strings
        Statistics for plotting capture length matrices,
        options include: "mean", "std", "median",
        "count", "sum", "min", "max", and "frequency".
        Note that "std" uses a degree of freedom of 1 in accordance with IEC/TS 62600-100.
        To output capture length matrices for multiple binning parameters,
        define as a list of strings: statistic = ["", "", ""]
    frequency_bins: numpy array or pandas Series (optional)
       Bin widths for frequency of S. Required for unevenly sized bins
    deep: bool (optional)
        If True use the deep water approximation. Default False. When
        False a depth check is run to check for shallow water. The ratio
        of the shallow water regime can be changed using the ratio
        keyword.
    rho: float (optional)
        Water density [kg/m^3]. Default = 1025 kg/m^3
    g: float (optional)
        Gravitational acceleration [m/s^2]. Default = 9.80665 m/s^2
    ratio: float or int (optional)
        Only applied if depth=False. If h/l > ratio,
        water depth will be set to deep. Default ratio = 2.
    show_values : bool (optional)
        Show values on the scatter diagram. Default = False.
    savepath: string (optional)
        Path to save figure. Terminate with '\'. Default="".

    Returns
    ---------
    LM: xarray dataset
        Capture length matrices

    maep_matrix: float
        Mean annual energy production
    """
    assert isinstance(S, (pd.DataFrame,pd.Series)), 'S must be of type pd.DataFrame or pd.Series'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(P, (np.ndarray, pd.Series)), 'P must be of type np.ndarray or pd.Series'
    assert isinstance(deep, bool), 'deep must be of type bool'
    assert isinstance(rho, (int,float)), 'rho must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'
    assert isinstance(ratio, (int,float)), 'ratio must be of type int or float'

    # Compute the enegy periods from the spectra data
    Te = wave.resource.energy_period(S, frequency_bins=frequency_bins)
    Te = Te['Te']

    # Compute the significant wave height from the NDBC spectra data
    Hm0 = wave.resource.significant_wave_height(S, frequency_bins=frequency_bins)
    Hm0 = Hm0['Hm0']

    # Compute the energy flux from spectra data and water depth
    J = wave.resource.energy_flux(S, h, deep=deep, rho=rho, g=g, ratio=ratio)
    J = J['J']

    # Calculate capture length from power and energy flux
    L = wave.performance.capture_length(P,J)

    # Generate bins for Hm0 and Te, input format (start, stop, step_size)
    Hm0_bins = np.arange(0, Hm0.values.max() + .5, .5)
    Te_bins = np.arange(0, Te.values.max() + 1, 1)

    # Create capture length matrices for each statistic based on IEC/TS 62600-100
    # Median, sum, frequency additionally provided
    LM = xarray.Dataset()
    LM['mean'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'mean', Hm0_bins, Te_bins)
    LM['std'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'std', Hm0_bins, Te_bins)
    LM['median'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'median', Hm0_bins, Te_bins)
    LM['count'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'count', Hm0_bins, Te_bins)
    LM['sum'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'sum', Hm0_bins, Te_bins)
    LM['min'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'min', Hm0_bins, Te_bins)
    LM['max'] = wave.performance.capture_length_matrix(Hm0, Te, L, 'max', Hm0_bins, Te_bins)
    LM['freq'] = wave.performance.capture_length_matrix(Hm0, Te, L,'frequency', Hm0_bins, Te_bins)

    # Create wave energy flux matrix using mean
    JM = wave.performance.wave_energy_flux_matrix(Hm0, Te, J, 'mean', Hm0_bins, Te_bins)

    # Calculate maep from matrix
    maep_matrix = wave.performance.mean_annual_energy_production_matrix(LM['mean'].to_pandas(), JM, LM['freq'].to_pandas())

    # Plot capture length matrices using statistic
    for str in statistic:
        if str not in list(LM.data_vars):
            print('ERROR: Invalid Statistics passed')
            continue
        plt.figure(figsize=(12,12), num='Capture Length Matrix ' + str)
        ax = plt.gca()
        wave.graphics.plot_matrix(LM[str].to_pandas(), xlabel='Te (s)', ylabel='Hm0 (m)', zlabel= str + ' of Capture Length', show_values=show_values, ax=ax)
        plt.savefig(join(savepath,'Capture Length Matrix ' + str + '.png'))

    return LM, maep_matrix
