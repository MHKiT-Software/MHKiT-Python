import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy import signal
from scipy import fft, fftpack
from scipy.signal import hilbert


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

    x.to_numpy()
    
    a = np.fft.fft(x,axis=0)
    
    amp = np.abs(a) # amplitude of the harmonics
    
    freqfft = fftpack.fftfreq(len(x),d=1./freq)
    

    harmonics = pd.DataFrame(amp,index=freqfft)
    
    
    harmonics=harmonics.sort_index(axis=0)
    if grid_freq == 60:    
        hz = np.arange(0,3005,5)
    elif grid_freq == 50: 
        hz = np.arange(0,2505,5)
    
    ind=pd.Index(harmonics.index)
    indn = [None]*np.size(hz)
    i = 0
    for n in hz:
        indn[i] = ind.get_loc(n, method='nearest')
        i = i+1
    
    harmonics = harmonics.iloc[indn]
    harmonics.index = hz
    harmonics = harmonics.loc[~harmonics.index.duplicated(keep='first')]
    harmonics = harmonics/len(x)*2
    
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
        
        hz = np.arange(0,3000,60)
    elif grid_freq == 50: 
        
        hz = np.arange(0,2500,50)
    
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
    THCD: float
        the total harmonic current distortion 
    """
    assert isinstance(harmonics_subgroup, (pd.Series, pd.DataFrame)), 'harmonic_subgroups must be of type pd.DataFrame or pd.Series'
    assert isinstance(rated_current, float), 'rated_current must be a float'
    
    harmonics_sq = harmonics_subgroup.iloc[2:50]**2

    harmonics_sum=harmonics_sq.sum()

    THCD = (np.sqrt(harmonics_sum)/harmonics_subgroup.iloc[1])*100

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
        
        hz = np.arange(0,3000,60)
    elif grid_freq == 50: 
        
        hz = np.arange(0,2500,50)
    
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
