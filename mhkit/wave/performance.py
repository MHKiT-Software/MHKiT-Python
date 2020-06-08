import numpy as np
import pandas as pd
import types
from scipy.stats import binned_statistic_2d as _binned_statistic_2d


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





