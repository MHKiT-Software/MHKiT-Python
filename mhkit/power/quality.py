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
        timeseries of voltage or current
    
    freq: float
        frequency of the timeseries data [Hz]
    
    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60
   
    
    Returns
    --------
    harmonics: pandas DataFrame 
        amplitude of the harmonics of the time series data indexed by the harmonic order
    """
   

    assert isinstance(x, (pd.Series, pd.DataFrame)), 'voltage must be of type pd.DataFrame'
    assert isinstance(freq, float), 'voltage must be of type pd.DataFrame'
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
    
    return harmonics


def harmonic_subgroups(harmonics, grid_freq): 
    """
    calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonics: pandas Series or DataFrame 
        RMS harmonic amplitude indexed by the harmonic order 
    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    --------
    harmonic_subgroups: pandas DataFrame
        harmonic subgroups 
    """        

    if grid_freq == 60:
        
        hz = np.arange(1,3000,60)
    elif grid_freq == 50: 
        
        hz = np.arange(1,2500,50)
    else:
        print('grid_freq must be either 60 or 50')
        pass
    
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
        the subgrouped RMS current harmonics indexed by harmonic order
    
    rated_current: float
        the rated current of the energy device in Amps
    
    Returns
    --------
    THCD: float
        the total harmonic current distortion 
    """
    
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
        RMS harmonic amplitude indexed by the harmonic order 

    grid_freq: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    -------
    interharmonics: pandas DataFrame
        interharmonics groups
    """
    #Note: work on the data types, df, Series, numpy to streamline this. Will I ever pass multiple columns of harmonics??
    

    if grid_freq == 60:
        
        hz = np.arange(0,3000,60)
    elif grid_freq == 50: 
        
        hz = np.arange(0,2500,50)
    else:
        print('grid_freq must be either 60 or 50')
        pass
    
    j=0
    i=0
    cols=harmonics.columns
    interharmonics=np.ones((np.size(hz),np.size(cols)))
    for n in hz: 
        harmonics=harmonics.sort_index(axis=0)
        ind=pd.Index(harmonics.index)
        
        indn = ind.get_loc(n, method='nearest')  
        if frequency == 60:
            subset = harmonics.iloc[indn+1:indn+11]**2
            subset = subset.squeeze()
        else: 
            subset = harmonics.iloc[indn+1:indn+7]**2
            subset = subset.squeeze()
        for col in cols:
            interharmonics[i,j] = np.sqrt(np.sum(subset))
            j=j+1
        j=0
        i=i+1
    
    
    interharmonics = pd.DataFrame(interharmonics,index=hz)

    return interharmonics
