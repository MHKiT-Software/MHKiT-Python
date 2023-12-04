import pandas as pd
import numpy as np
from scipy import fftpack
import xarray as xr

# This group of functions are to be used for power quality assessments

def harmonics(x, freq, grid_freq, to_pandas=True):
    """
    Calculates the harmonics from time series of voltage or current based on IEC 61000-4-7. 

    Parameters
    -----------
    x: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Time-series of voltage [V] or current [A]

    freq: float or Int
        Frequency of the time-series data [Hz]

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60
        
    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    harmonics: pandas DataFrame  or xarray Dataset
        Amplitude of the time-series data harmonics indexed by the harmonic 
        frequency with signal name columns
    """
    if not isinstance(x, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise ValueError(
            'Provided voltage or current must be of type pd.DataFrame or pd.Series')

    if not isinstance(freq, (float, int)):
        raise ValueError('freq must be of type float or integer')

    if grid_freq not in [50, 60]:
        raise ValueError('grid_freq must be either 50 or 60')

    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_xarray()
        
    # Check if x is a DataFrame
    if isinstance(x, (pd.DataFrame)):
        cols = x.columns

    x = x.to_numpy()
    sample_spacing = 1./freq
    frequency_bin_centers = fftpack.fftfreq(len(x), d=sample_spacing)

    harmonics_amplitude = np.abs(np.fft.fft(x, axis=0))

    harmonics = pd.DataFrame(harmonics_amplitude, index=frequency_bin_centers)
    harmonics = harmonics.sort_index()

    # Keep the signal name as the column name
    if 'cols' in locals():
        harmonics.columns = cols

    if grid_freq == 60:
        hz = np.arange(0, 3060, 5)
    elif grid_freq == 50:
        hz = np.arange(0, 2570, 5)

    harmonics = harmonics.reindex(hz, method='nearest')
    harmonics = harmonics/len(x)*2
    
    if to_pandas:
        harmonics = harmonics.to_pandas()

    return harmonics


def harmonic_subgroups(harmonics, grid_freq, to_pandas=True):
    """
    Calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonics: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency 
        
    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60
        
    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    harmonic_subgroups: pandas DataFrame or xarray Dataset
        Harmonic subgroups indexed by harmonic frequency 
        with signal name columns
    """
    if not isinstance(harmonics, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise ValueError('harmonics must be of type pd.DataFrame or pd.Series')

    if grid_freq not in [50, 60]:
        raise ValueError('grid_freq must be either 50 or 60')

    # Check if harmonics is a DataFrame
    if isinstance(harmonics, pd.DataFrame):
        cols = harmonics.columns

    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    else:
        hz = np.arange(0, 2550, 50)

    j = 0
    i = 0
    cols = harmonics.columns
    harmonic_subgroups = np.ones((np.size(hz), np.size(cols)))
    for n in hz:

        harmonics = harmonics.sort_index(axis=0)
        ind = pd.Index(harmonics.index)

        indn = ind.get_indexer([n], method='nearest')[0]

        for col in cols:
            harmonic_subgroups[i, j] = np.sqrt(np.sum(
                [harmonics[col].iloc[indn-1]**2, harmonics[col].iloc[indn]**2, harmonics[col].iloc[indn+1]**2]))
            j = j+1
        j = 0
        i = i+1

    harmonic_subgroups = pd.DataFrame(harmonic_subgroups, index=hz)

    # Keep the signal name as the column name
    if 'cols' in locals():
        harmonic_subgroups.columns = cols
    
    if to_pandas:
        harmonic_subgroups = harmonic_subgroups.to_pandas()

    return harmonic_subgroups


def total_harmonic_current_distortion(harmonics_subgroup, rated_current, to_pandas=True):
    """
    Calculates the total harmonic current distortion (THC) based on IEC/TS 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Subgrouped current harmonics indexed by harmonic frequency

    rated_current: float
        Rated current of the energy device in Amps
        
    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    THCD: pd.DataFrame or xarray Dataset
        Total harmonic current distortion indexed by signal name with THCD column 
    """
    if not isinstance(harmonics_subgroup, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise ValueError(
            'harmonic_subgroups must be of type pd.DataFrame or pd.Series')

    if not isinstance(rated_current, float):
        raise ValueError('rated_current must be a float')

    harmonics_sq = harmonics_subgroup.iloc[2:50]**2

    harmonics_sum = harmonics_sq.sum()

    THCD = (np.sqrt(harmonics_sum)/harmonics_subgroup.iloc[1])*100
    THCD = pd.DataFrame(THCD)  # converting to dataframe for Matlab
    THCD.columns = ['THCD']
    THCD = THCD.T
    
    if to_pandas:
        THCD = THCD.to_pandas()

    return THCD


def interharmonics(harmonics, grid_freq, to_pandas=True):
    """
    Calculates the interharmonics from the harmonics of current

    Parameters
    -----------
    harmonics: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency 

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60
        
    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    -------
    interharmonics: pandas DataFrame or xarray Dataset
        Interharmonics groups
    """
    if not isinstance(harmonics, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise ValueError('harmonics must be of type pd.DataFrame or pd.Series')

    if grid_freq not in [50, 60]:
        raise ValueError('grid_freq must be either 50 or 60')

    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    elif grid_freq == 50:
        hz = np.arange(0, 2550, 50)

    j = 0
    i = 0
    cols = harmonics.columns
    interharmonics = np.ones((np.size(hz), np.size(cols)))
    for n in hz:
        harmonics = harmonics.sort_index(axis=0)
        ind = pd.Index(harmonics.index)

        indn = ind.get_indexer([n], method='nearest')[0]

        for col in cols:
            if grid_freq == 60:
                subset = harmonics[col].iloc[indn+1:indn+11]**2
                subset = subset.squeeze()
            else:
                subset = harmonics[col].iloc[indn+1:indn+7]**2
                subset = subset.squeeze()

            interharmonics[i, j] = np.sqrt(np.sum(subset))
            j = j+1
        j = 0
        i = i+1

    interharmonics = pd.DataFrame(interharmonics, index=hz)
    
    if to_pandas:
        interharmonics = interharmonics.to_pandas()

    return interharmonics
