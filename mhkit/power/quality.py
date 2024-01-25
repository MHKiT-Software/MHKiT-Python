import pandas as pd
import numpy as np
from scipy import fftpack
import xarray as xr
from .characteristics import _convert_to_dataset

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
        raise TypeError('x must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(x)}')

    if not isinstance(freq, (float, int)):
        raise TypeError(f'freq must be of type float or integer. Got {type(freq)}')

    if grid_freq not in [50, 60]:
        raise ValueError(f'grid_freq must be either 50 or 60. Got {grid_freq}')

    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got {type(to_pandas)}')

    # Convert input to xr.Dataset
    x = _convert_to_dataset(x, 'data')

    sample_spacing = 1./freq
    
    # Loop through all variables in x
    harmonics = xr.Dataset()
    for var in x.data_vars:
        dataarray = x[var]
        dataarray = dataarray.to_numpy()
        
        frequency_bin_centers = fftpack.fftfreq(len(dataarray), d=sample_spacing)
        harmonics_amplitude = np.abs(np.fft.fft(dataarray, axis=0))
        
        harmonics = harmonics.assign({var: (['frequency'], harmonics_amplitude)})
        harmonics = harmonics.assign_coords({'frequency': frequency_bin_centers})    
    harmonics = harmonics.sortby('frequency')

    if grid_freq == 60:
        hz = np.arange(0, 3060, 5)
    elif grid_freq == 50:
        hz = np.arange(0, 2570, 5)

    harmonics = harmonics.reindex({'frequency': hz}, method='nearest')
    harmonics = harmonics/len(x[var])*2
    
    if to_pandas:
        harmonics = harmonics.to_pandas()

    return harmonics


def harmonic_subgroups(harmonics, grid_freq, frequency_dimension="", to_pandas=True):
    """
    Calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonics: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency 

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    harmonic_subgroups: pandas DataFrame or xarray Dataset
        Harmonic subgroups indexed by harmonic frequency 
        with signal name columns
    """
    if not isinstance(harmonics, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('harmonics must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(harmonics)}')
    
    if grid_freq not in [50, 60]:
        raise ValueError(f'grid_freq must be either 50 or 60. Got {grid_freq}')

    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    if not isinstance(frequency_dimension, str):
        raise TypeError(
            f'frequency_dimension must be of type bool. Got: {type(frequency_dimension)}')

    # Convert input to xr.Dataset
    harmonics = _convert_to_dataset(harmonics, 'harmonics')
    
    if frequency_dimension != '' and frequency_dimension not in harmonics.coords:
        raise ValueError('frequency_dimension was supplied but is not a dimension '
                         + f'of harmonics. Got {frequency_dimension}')

    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    else:
        hz = np.arange(0, 2550, 50)
    
    # Sort input data index
    if frequency_dimension == "":
        frequency_dimension = list(harmonics.dims)[0]
    harmonics = harmonics.sortby(frequency_dimension)
    
    # Loop through all variables in harmonics
    harmonic_subgroups = xr.Dataset()
    for var in harmonics.data_vars:
        dataarray = harmonics[var]
        subgroup = np.zeros(np.size(hz))
        
        for ihz in np.arange(0,len(hz)):
            n = hz[ihz] 
            ind = dataarray.indexes[frequency_dimension].get_loc(n)
            
            data_subset = dataarray.isel({frequency_dimension:[ind-1, ind, ind+1]})
            subgroup[ihz] = (data_subset**2).sum()**0.5
            
        harmonic_subgroups = harmonic_subgroups.assign({var: (['frequency'], subgroup)})
        harmonic_subgroups = harmonic_subgroups.assign_coords({'frequency': hz})

    if to_pandas:
        harmonic_subgroups = harmonic_subgroups.to_pandas()

    return harmonic_subgroups


def total_harmonic_current_distortion(harmonics_subgroup, frequency_dimension="", to_pandas=True):
    """
    Calculates the total harmonic current distortion (THC) based on IEC/TS 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Subgrouped current harmonics indexed by harmonic frequency

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    THCD: pd.DataFrame or xarray Dataset
        Total harmonic current distortion indexed by signal name with THCD column 
    """
    if not isinstance(harmonics_subgroup, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('harmonics_subgroup must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(harmonics_subgroup)}')

    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    if not isinstance(frequency_dimension, str):
        raise TypeError(
            f'frequency_dimension must be of type bool. Got: {type(frequency_dimension)}')

    # Convert input to xr.Dataset
    harmonics_subgroup = _convert_to_dataset(harmonics_subgroup, 'harmonics')

    if frequency_dimension != '' and frequency_dimension not in harmonics.coords:
        raise ValueError('frequency_dimension was supplied but is not a dimension '
                         + f'of harmonics. Got {frequency_dimension}')
    
    if frequency_dimension == "":
        frequency_dimension = list(harmonics_subgroup.dims)[0]
    harmonics_sq = harmonics_subgroup.isel({frequency_dimension: slice(2,50)})**2
    harmonics_sum = harmonics_sq.sum()

    THCD = (np.sqrt(harmonics_sum)/harmonics_subgroup.isel({frequency_dimension: 1}))*100
    
    if isinstance(THCD, xr.DataArray):
        THCD.name = ['THCD']    
    
    if to_pandas:
        THCD = THCD.to_pandas()

    return THCD


def interharmonics(harmonics, grid_freq, frequency_dimension="", to_pandas=True):
    """
    Calculates the interharmonics from the harmonics of current

    Parameters
    -----------
    harmonics: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency 

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied, 
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    -------
    interharmonics: pandas DataFrame or xarray Dataset
        Interharmonics groups
    """
    if not isinstance(harmonics, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('harmonics must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(harmonics)}')

    if grid_freq not in [50, 60]:
        raise ValueError(f'grid_freq must be either 50 or 60. Got {grid_freq}')

    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    if isinstance(harmonics, (pd.DataFrame, pd.Series)):
        harmonics = harmonics.to_xarray()

    if grid_freq == 60:
        hz = np.arange(0, 3060, 60)
    elif grid_freq == 50:
        hz = np.arange(0, 2550, 50)

    # Sort input data index
    if frequency_dimension == "":
        frequency_dimension = list(harmonics.dims)[0]
    harmonics = harmonics.sortby(frequency_dimension)

    # Loop through all variables in harmonics
    interharmonics = xr.Dataset()
    for var in harmonics.data_vars:
        dataarray = harmonics[var]
        subset = np.zeros(np.size(hz))

        for ihz in np.arange(0,len(hz)):
            n = hz[ihz]
            ind = dataarray.indexes[frequency_dimension].get_loc(n)

            if grid_freq == 60:
                data = dataarray.isel({frequency_dimension:slice(ind+1,ind+11)})
                subset[ihz] = (data**2).sum()**0.5
            else:
                data = dataarray.isel({frequency_dimension:slice(ind+1,ind+7)})
                subset[ihz] = (data**2).sum()**0.5

        interharmonics = interharmonics.assign({var: (['frequency'], subset)})
        interharmonics = interharmonics.assign_coords({'frequency': hz})

    if to_pandas:
        interharmonics = interharmonics.to_pandas()

    return interharmonics
