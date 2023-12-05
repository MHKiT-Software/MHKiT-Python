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
    harmonics: pandas DataFrame or xarray Dataset
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
    
    sample_spacing = 1./freq
    
    # Handle multi variable input
    if isinstance(x, (xr.Dataset)):
        harmonics = xr.Dataset()
        cols = list(x.data_vars)
        for var in x.data_vars:
            dataarray = x[var]
            dataarray = dataarray.to_numpy()
            
            frequency_bin_centers = fftpack.fftfreq(len(dataarray), d=sample_spacing)
            harmonics_amplitude = np.abs(np.fft.fft(dataarray, axis=0))
            
            harmonics = harmonics.assign({var: (['frequency'], harmonics_amplitude)})
            harmonics = harmonics.assign_coords({'frequency': frequency_bin_centers})
    else:
        cols = x.name
        x = x.to_numpy()
        frequency_bin_centers = fftpack.fftfreq(len(x), d=sample_spacing)
    
        harmonics_amplitude = np.abs(np.fft.fft(x, axis=0))
        harmonics = xr.DataArray(data=harmonics_amplitude,
                                 dims='frequency',
                                 coords={'frequency': frequency_bin_centers},
                                 name = cols)
    
    harmonics = harmonics.sortby('frequency')

    if grid_freq == 60:
        hz = np.arange(0, 3060, 5)
    elif grid_freq == 50:
        hz = np.arange(0, 2570, 5)

    harmonics = harmonics.reindex({'frequency': hz}, method='nearest')
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

    if isinstance(harmonics, (pd.DataFrame, pd.Series)):
        harmonics = harmonics.to_xarray()

    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    else:
        hz = np.arange(0, 2550, 50)
    
    # Sort input data index
    dim = list(harmonics.dims)[0]
    harmonics = harmonics.sortby(dim)
    
    # Handle multi variable input
    if isinstance(harmonics, xr.Dataset):
        harmonic_subgroups = xr.Dataset()
        
        for var in harmonics.data_vars:
            dataarray = harmonics[var]
            subgroup = np.zeros(np.size(hz))
            
            for ihz in np.arange(0,len(hz)):
                n = hz[ihz] 
                ind = dataarray.indexes[dim].get_loc(n)
                
                data_subset = dataarray.isel({dim:[ind-1, ind, ind+1]})
                subgroup[ihz] = (data_subset**2).sum()**0.5
                
            harmonic_subgroups = harmonic_subgroups.assign({var: (['frequency'], subgroup)})
            harmonic_subgroups = harmonic_subgroups.assign_coords({'frequency': hz})
    else:
        subgroup = np.zeros(np.size(hz))
        
        for ihz in np.arange(0,len(hz)):
            n = hz[ihz] 
            ind = harmonics.indexes[dim].get_loc(n)
            
            data_subset = harmonics.isel({dim:[ind-1, ind, ind+1]})
            subgroup[ihz] = (data_subset**2).sum()**0.5
            
        harmonic_subgroups = xr.DataArray(data = subgroup,
                                          dims = 'frequency',
                                          coords = {'frequency': hz},
                                          name = harmonics.name)

    if to_pandas:
        harmonic_subgroups = harmonic_subgroups.to_pandas()

    return harmonic_subgroups


def total_harmonic_current_distortion(harmonics_subgroup, to_pandas=True):
    """
    Calculates the total harmonic current distortion (THC) based on IEC/TS 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Subgrouped current harmonics indexed by harmonic frequency
        
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

    if isinstance(harmonics_subgroup, (pd.DataFrame, pd.Series)):
        harmonics_subgroup = harmonics_subgroup.to_xarray()
    
    dim = list(harmonics_subgroup.dims)[0]
    harmonics_sq = harmonics_subgroup.isel({dim: slice(2,50)})**2
    harmonics_sum = harmonics_sq.sum()

    THCD = (np.sqrt(harmonics_sum)/harmonics_subgroup.isel({dim: 1}))*100
    
    if isinstance(THCD, xr.DataArray):
        THCD.name = ['THCD']
    
    THCD = THCD.transpose()
    
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

    if isinstance(harmonics, (pd.DataFrame, pd.Series)):
        harmonics = harmonics.to_xarray()
    
    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    elif grid_freq == 50:
        hz = np.arange(0, 2550, 50)
        
    # Sort input data index
    dim = list(harmonics.dims)[0]
    harmonics = harmonics.sortby(dim)
    
    # Handle multi variable input
    if isinstance(harmonics, xr.Dataset):
        interharmonics = xr.Dataset()
        
        for var in harmonics.data_vars:
            dataarray = harmonics[var]
            subset = np.zeros(np.size(hz))
            
            for ihz in np.arange(0,len(hz)):
                n = hz[ihz] 
                ind = dataarray.indexes[dim].get_loc(n)
                
                if grid_freq == 60:
                    data = dataarray.isel({dim:slice(ind+1,ind+11)})
                    subset[ihz] = (data**2).sum()**0.5
                else:
                    data = dataarray.isel({dim:slice(ind+1,ind+7)})
                    subset[ihz] = (data**2).sum()**0.5
                
            interharmonics = interharmonics.assign({var: (['frequency'], subset)})
            interharmonics = interharmonics.assign_coords({'frequency': hz})
    else:
        subset = np.zeros(np.size(hz))
        
        for ihz in np.arange(0,len(hz)):
            n = hz[ihz] 
            ind = harmonics.indexes[dim].get_loc(n)    
            
            if grid_freq == 60:
                data = harmonics.isel({dim:slice(ind+1,ind+11)})
                subset[ihz] = (data**2).sum()**0.5
            else:
                data = harmonics.isel({dim:slice(ind+1,ind+7)})
                subset[ihz] = (data**2).sum()**0.5
                
        interharmonics = xr.DataArray(data = subset,
                                          dims = 'frequency',
                                          coords = {'frequency': hz},
                                          name = harmonics.name)
    
    if to_pandas:
        interharmonics = interharmonics.to_pandas()

    return interharmonics
