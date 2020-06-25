import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.signal import hilbert
from scipy import signal, fft, fftpack


#This group of functions are to be used for power quality assessments 

def harmonics(x,freq,grid_freq):
    """
    Calculates the harmonics from time series of voltage or current based on IEC 61000-4-7. 

    Parameters
    -----------
    x: pandas Series of DataFrame
        timeseries of voltage [V] or current [A]
    
    freq: float or Int
        frequency of the timeseries data [Hz]
    
    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60.
   
    
    Returns
    --------
    harmonics: pandas DataFrame 
        amplitude of the harmonics of the time series data indexed by the harmonic frequency
    """
    assert isinstance(x, (pd.Series, pd.DataFrame)), 'Provided voltage or current must be of type pd.DataFrame or pd.Series'
    assert isinstance(freq, (float, int)), 'freq must be of type float or integer'
    assert (grid_freq == 50 or grid_freq == 60), 'grid_freq must be either 50 or 60'

    x = x.to_numpy()
    sample_spacing = 1./freq 
    frequency_bin_centers = fftpack.fftfreq(len(x), d=sample_spacing)

    harmonics_amplitude = np.abs(np.fft.fft(x, axis=0))

    harmonics = pd.DataFrame(harmonics_amplitude, index=frequency_bin_centers)
    harmonics = harmonics.sort_index()

    if grid_freq == 60:    
        hz = np.arange(0,3060,5)
    elif grid_freq == 50: 
        hz = np.arange(0,2570,5)


    harmonics_index = pd.Index(harmonics.index)
    frequency_index_loc = [None]*np.size(hz)

    index_0 = harmonics_index.get_loc(hz[0], method='nearest')
    frequency_index_loc[0] = index_0
    sorted_index_greater_than_hz0 = harmonics_index[index_0:]

    i = 1
    for val in hz[1:]:
        frequency_index_loc[i] = sorted_index_greater_than_hz0.get_loc(val, method='nearest')
        i += 1
    frequency_index_loc[1:] = (np.array(frequency_index_loc[1:])+index_0).tolist()

    harmonics = harmonics.iloc[frequency_index_loc]
    harmonics.index = hz
    harmonics = harmonics.loc[~harmonics.index.duplicated(keep='first')]
    harmonics = harmonics/len(x)*2


#    harmonics = harmonics.iloc[frequency_index_loc]
#    harmonics.index = hz
#    harmonics = harmonics.loc[~harmonics.index.duplicated(keep='first')]
#    harmonics = harmonics/len(x)*2
#   import ipdb; ipdb.set_trace()
    
    return harmonics


def harmonic_subgroups(harmonics, grid_freq): 
    """
    calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonics: pandas Series or DataFrame 
        harmonic amplitude indexed by the harmonic frequency 
    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    --------
    harmonic_subgroups: pandas DataFrame
        harmonic subgroups indexed by harmonic frequency
    """        
    assert isinstance(harmonics, (pd.Series, pd.DataFrame)), 'harmonics must be of type pd.DataFrame or pd.Series'
    assert (grid_freq == 50 or grid_freq == 60), 'grid_freq must be either 50 or 60'

    if grid_freq == 60:
        
        hz = np.arange(0,3060,60)
    elif grid_freq == 50: 
        
        hz = np.arange(0,2550,50)
    
    j=0
    i=0
    cols=harmonics.columns
    harmonic_subgroups=np.ones((np.size(hz),np.size(cols)))
    for n in hz:

        harmonics=harmonics.sort_index(axis=0)
        ind=pd.Index(harmonics.index)
        
        indn = ind.get_loc(n, method='nearest')
        for col in cols:
            harmonic_subgroups[i,j] = np.sqrt(np.sum([harmonics[col].iloc[indn-1]**2,harmonics[col].iloc[indn]**2,harmonics[col].iloc[indn+1]**2]))
            j=j+1
        j=0
        i=i+1
    
    harmonic_subgroups = pd.DataFrame(harmonic_subgroups,index=hz)

    return harmonic_subgroups

def total_harmonic_current_distortion(harmonics_subgroup,rated_current):    

    """
    Calculates the total harmonic current distortion (THC) based on IEC 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas DataFrame or Series
        the subgrouped current harmonics indexed by harmonic frequency
    
    rated_current: float
        the rated current of the energy device in Amps
    
    Returns
    --------
    THCD: pd.DataFrame
        the total harmonic current distortion 
    """
    assert isinstance(harmonics_subgroup, (pd.Series, pd.DataFrame)), 'harmonic_subgroups must be of type pd.DataFrame or pd.Series'
    assert isinstance(rated_current, float), 'rated_current must be a float'
    
    harmonics_sq = harmonics_subgroup.iloc[2:50]**2

    harmonics_sum=harmonics_sq.sum()

    THCD = (np.sqrt(harmonics_sum)/harmonics_subgroup.iloc[1])*100
    THCD = pd.DataFrame(THCD)  # converting to dataframe for Matlab

    return THCD

def interharmonics(harmonics,grid_freq):
    """
    calculates the interharmonics ffrom the harmonics of current

    Parameters
    -----------
    harmonics: pandas Series or DataFrame 
        harmonic amplitude indexed by the harmonic frequency 

    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    -------
    interharmonics: pandas DataFrame
        interharmonics groups
    """
    assert isinstance(harmonics, (pd.Series, pd.DataFrame)), 'harmonics must be of type pd.DataFrame or pd.Series'
    assert (grid_freq == 50 or grid_freq == 60), 'grid_freq must be either 50 or 60'
    

    if grid_freq == 60:
        
        hz = np.arange(0,3060,60)
    elif grid_freq == 50: 
        
        hz = np.arange(0,2550,50)
    
    j=0
    i=0
    cols=harmonics.columns
    interharmonics=np.ones((np.size(hz),np.size(cols)))
    for n in hz: 
        harmonics=harmonics.sort_index(axis=0)
        ind=pd.Index(harmonics.index)
        
        indn = ind.get_loc(n, method='nearest') 
        for col in cols:
            if grid_freq == 60:
                subset = harmonics[col].iloc[indn+1:indn+11]**2
                subset = subset.squeeze()
            else: 
                subset = harmonics[col].iloc[indn+1:indn+7]**2
                subset = subset.squeeze()
        
            interharmonics[i,j] = np.sqrt(np.sum(subset))
            j=j+1
        j=0
        i=i+1
    
    
    interharmonics = pd.DataFrame(interharmonics,index=hz)

    return interharmonics
