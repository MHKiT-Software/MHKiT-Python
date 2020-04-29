import pandas as pd
import numpy as np
from scipy.optimize import fsolve as _fsolve
from scipy import signal as _signal
from itertools import product as _product


### Spectrum
def elevation_spectrum(eta, sample_rate, nnft, window='hann', detrend=True, noverlap=None):
    """
    Calculates the wave energy spectrum from wave elevation time-series
    
    Parameters
    ------------
    eta: pandas DataFrame 
        Wave surface elevation [m] indexed by time [datetime or s]
    sample_rate: float
        Data frequency [Hz]
    nnft: integer
        Number of bins in the Fast Fourier Transform
    window: string (optional)
        Signal window type. 'hann' is used by default given the broadband 
        nature of waves. See scipy.signal.get_window for more options.
    detrend: bool (optional)
        Specifies if a linear trend is removed from the data before calculating 
        the wave energy spectrum.  Data is detrended by default.
        
    Returns
    ---------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    """

    # TODO: Add confidence intervals, equal energy frequency spacing, and NDBC 
    #       frequency spacing
    # TODO: may need an assert for the length of nnft- signal.welch breaks when nfft is too short    
    # TODO: check for uniform sampling
    assert isinstance(eta, pd.DataFrame), 'eta must be of type pd.DataFrame'
    assert isinstance(sample_rate, (float,int)), 'sample_rate must be of type int or float'
    assert isinstance(nnft, int), 'nnft must be of type int'
    assert isinstance(window, str), 'window must be of type str'
    assert isinstance(detrend, bool), 'detrend must be of type bool'
    assert nnft > 0, 'nnft must be > 0'
    assert sample_rate > 0, 'sample_rate must be > 0'
       
    S = pd.DataFrame()
    for col in eta.columns:
        data = eta[col]
        if detrend:
            data = _signal.detrend(data.dropna(), axis=-1, type='linear', bp=0)
        [f, wave_spec_measured] = _signal.welch(data, fs=sample_rate, window=window, 
            nperseg=nnft, nfft=nnft, noverlap=noverlap) 
        S[col] = wave_spec_measured
    S.index=f
    S.columns = eta.columns
    
    return S


def pierson_moskowitz_spectrum(f, Tp):
    """
    Calculates Pierson-Moskowitz Spectrum from Tucker and Pitt (2001) 
    
    Parameters
    ------------
    f: numpy array
        Frequency [Hz]
    Tp: float/int
        Peak period [s]  
    
    Returns
    ---------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed frequency [Hz]

    """
    try:
        f = np.array(f)
    except:
        pass
    assert isinstance(f, np.ndarray), 'f must be of type np.ndarray'
    assert isinstance(Tp, (int,float)), 'Tp must be of type int or float'
    
    f.sort()
    g = 9.81   
    B_PM = (5/4)*(1/Tp)**(4)
    A_PM = 0.0081*g**2*(2*np.pi)**(-4)
    Sf  = A_PM*f**(-5)*np.exp(-B_PM*f**(-4)) 
     
    col_name = 'Pierson-Moskowitz ('+str(Tp)+'s)'
    S = pd.DataFrame(Sf, index=f, columns=[col_name])    

    return S

def bretschneider_spectrum(f, Tp, Hs):
    """
    Calculates Bretschneider Sprectrum from Tucker and Pitt (2001)
    
    Parameters
    ------------
    f: numpy array
        Frequency [Hz]
    Tp: float/int
        Peak period [s]
    Hs: float/int
        Significant wave height [m]        
    
    Returns
    ---------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]

    """
    try:
        f = np.array(f)
    except:
        pass
    assert isinstance(f, np.ndarray), 'f must be of type np.ndarray'
    assert isinstance(Tp, (int,float)), 'Tp must be of type int or float'
    assert isinstance(Hs, (int,float)), 'Hs must be of type int or float'

    f.sort()
    B_BS = (1.057/Tp)**4
    A_BS = B_BS*(Hs/2)**2
    Sf = A_BS*f**(-5)*np.exp(-B_BS*f**(-4))        

    col_name = 'Bretschneider ('+str(Hs)+'m,'+str(Tp)+'s)'
    S = pd.DataFrame(Sf, index=f, columns=[col_name])    
    
    return S


def jonswap_spectrum(f, Tp, Hs, gamma=3.3):
    """
    Calculates JONSWAP spectrum from Hasselmann et al (1973)
    
    Parameters
    ------------
    f: numpy array
        Frequency [Hz]
    Tp: float/int
        Peak period [s]
    Hs: float/int
        Significant wave height [m]    
    gamma: float (optional)
        Gamma
    
    Returns
    ---------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed frequency [Hz]
    """
    
    try:
        f = np.array(f)
    except:
        pass
    assert isinstance(f, np.ndarray), 'f must be of type np.ndarray'
    assert isinstance(Tp, (int,float)), 'Tp must be of type int or float'
    assert isinstance(Hs, (int,float)), 'Hs must be of type int or float'
    assert isinstance(gamma, (int,float)), 'gamma must be of type int or float'

    f.sort()    
    g = 9.81 
    
    # Cutoff frequencies for gamma function
    siga = 0.07 
    sigb = 0.09 
    
    fp = 1/Tp # peak frequency
    lind = np.where(f<=fp)
    hind = np.where(f>fp)
    Gf = np.zeros(f.shape)
    Gf[lind] = gamma**np.exp(-(f[lind]-fp)**2/(2*siga**2*fp**2))
    Gf[hind] = gamma**np.exp(-(f[hind]-fp)**2/(2*sigb**2*fp**2))
    S_temp = g**2*(2*np.pi)**(-4)*f**(-5)*np.exp(-(5/4)*(f/fp)**(-4))
    alpha_JS = Hs**(2)/16/np.trapz(S_temp*Gf,f)
    Sf = alpha_JS*S_temp*Gf # Wave Spectrum [m^2-s] 
    
    col_name = 'JONSWAP ('+str(Hs)+'m,'+str(Tp)+'s)'
    S = pd.DataFrame(Sf, index=f, columns=[col_name])  
    
    return S

### Metrics
def surface_elevation(S, time_index, seed=123, phases=None):
    """
    Calculates wave elevation time-series from spectrum using a random phase
    
    Parameters
    ------------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    time_index: numpy array
        Time used to create the wave elevation time-series [s],
        for example, time = np.arange(0,100,0.01)
    seed: int (optional)
        Random seed
    phases: numpy array (optional)
        Explicit phases for frequency components (overrides seed)
        for example, phases = np.random.rand(len(S)) * 2 * np.pi
        
    Returns
    ---------
    eta: pandas DataFrame
        Wave surface elevation [m] indexed by time [s]
    
    """
    try:
        time_index = np.array(time_index)
    except:
        pass
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    assert isinstance(time_index, np.ndarray), 'time_index must be of type np.ndarray'
    assert isinstance(seed, (type(None),int)), 'seed must be of type int'
    assert isinstance(phases, (type(None),np.ndarray)), 'phases must be of type np.ndarray'
    assert len(phases) == len(S), 'length of phases must match S'
    
    
    start_time = time_index[0]
    end_time = time_index[-1]
    
    f = pd.Series(S.index) # frequency
    f.index = f
    delta_f = f.diff()
    
    if phases is None:
        np.random.seed(seed)
        phase = pd.Series(2*np.pi*np.random.rand(f.size))
    else:
        phase = pd.Series(phases)
    phase.index = f
    phase = phase[start_time:end_time] # Should phase, omega, and A*delta_f be 
                                        #   truncated before computation?
    omega = pd.Series(2*np.pi*f) # angular freqency
    omega.index = f
    omega = omega[start_time:end_time]

    # Wave amplitude times delta f, truncated
    A = 2*S 
    A = A.multiply(delta_f, axis=0)
    A = np.sqrt(A)
    A = A.loc[start_time:end_time,:]
    
    # Product of omega and time
    B = np.array([x*y for x,y in _product(time_index, omega)])
    B = B.reshape((len(time_index),len(omega)))
    B = pd.DataFrame(B, index=time_index, columns=omega.index)

    C = np.real(np.exp(1j*(B+phase)))
    C = pd.DataFrame(C, index=time_index, columns=omega.index)
      
    eta = pd.DataFrame(columns=S.columns, index=time_index)
    for col in A.columns:
        eta[col] = (C*A[col]).sum(axis=1)

    return eta


def frequency_moment(S, N, frequency_bins=None):
    """
    Calculates the Nth frequency moment of the spectrum
    
    Parameters
    -----------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    N: int
        Moment (0 for 0th, 1 for 1st ....)
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    
    Returns
    -------
    m: pandas DataFrame 
        Nth Frequency Moment indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    assert isinstance(N, int), 'N must be of type int'
    
    # Eq 8 in IEC 62600-101 
    spec = S[S.index > 0] # omit frequency of 0
    
    f = spec.index 
    fn = np.power(f, N)
    if frequency_bins is None:
        delta_f = pd.Series(f).diff()
        delta_f[0] = f[1]-f[0]
    else:
         
        assert isinstance(frequency_bins, (np.ndarray,pd.Series,pd.DataFrame)), 'frequency_bins must be of type np.ndarray or pd.Series'
        delta_f = pd.Series(frequency_bins)

    delta_f.index = f
    # print(f)
    # print(delta_f.values)
        
    m = spec.multiply(fn,axis=0).multiply(delta_f,axis=0)
    m = m.sum(axis=0)
    
    m = pd.DataFrame(m, index=S.columns, columns = ['m'+str(N)])

    return m


def significant_wave_height(S, frequency_bins=None):
    """
    Calculates wave height from spectra

    Parameters
    ------------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    Hm0: pandas DataFrame 
        Significant wave height [m] index by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    # Eq 12 in IEC 62600-101
    
    Hm0 = 4*np.sqrt(frequency_moment(S,0,frequency_bins=frequency_bins))
    Hm0.columns = ['Hm0']
    
    return Hm0


def average_zero_crossing_period(S,frequency_bins=None):
    """
    Calculates wave average zero crossing period from spectra
    
    Parameters
    ------------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    Tz: pandas DataFrame
        Average zero crossing period [s] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    # Eq 15 in IEC 62600-101 
    m0 = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m2 = frequency_moment(S,2,frequency_bins=frequency_bins).squeeze()
    
    Tz = np.sqrt(m0/m2)
    Tz = pd.DataFrame(Tz, index=S.columns, columns = ['Tz'])
    
    return Tz


def average_crest_period(S,frequency_bins=None):
    """
    Calculates wave average crest period from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    Tavg: pandas DataFrame
        Average wave period [s] indexed by S.columns
    
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    m2 = frequency_moment(S,2,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m4 = frequency_moment(S,4,frequency_bins=frequency_bins).squeeze()
    
    Tavg = np.sqrt(m2/m4)
    Tavg = pd.DataFrame(Tavg, index=S.columns, columns=['Tavg'])
    
    return Tavg


def average_wave_period(S,frequency_bins=None):
    """
    Calculates mean wave period from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    Tm: pandas DataFrame 
        Mean wave period [s] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    m0 = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m1 = frequency_moment(S,1,frequency_bins=frequency_bins).squeeze() 
    
    Tm = np.sqrt(m0/m1)
    Tm = pd.DataFrame(Tm, index=S.columns, columns=['Tm'])    
    
    return Tm


def peak_period(S):
    """
    Calculates wave energy period from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
        
    Returns
    ---------
    Tp: pandas DataFrame
        Wave peak period [s] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    # Eq 14 in IEC 62600-101 
    fp = S.idxmax(axis=0) # Hz
    
    Tp = 1/fp
    Tp = pd.DataFrame(Tp, index=S.columns, columns=["Tp"])
    
    return Tp


def energy_period(S,frequency_bins=None):
    """
    Calculates wave energy period from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    Te: pandas DataFrame
        Wave energy period [s] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    mn1 = frequency_moment(S,-1,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m0  = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze()
    
    # Eq 13 in IEC 62600-101 
    Te = mn1/m0
    Te = pd.DataFrame(Te, index=S.columns, columns=['Te'])
    
    return Te

    
def spectral_bandwidth(S,frequency_bins=None):
    """
    Calculates bandwidth from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    e: pandas DataFrame
        Spectral bandwidth [s] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    m2 = frequency_moment(S,2,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m0 = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze()
    m4 = frequency_moment(S,4,frequency_bins=frequency_bins).squeeze()
    
    e = np.sqrt(1- (m2**2)/(m0/m4))
    e = pd.DataFrame(e, index=S.columns, columns=['e'])
    
    return e
    

def spectral_width(S,frequency_bins=None):
    """
    Calculates wave spectral width from spectra
    
    Parameters
    ------------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
        
    Returns
    ---------
    v: pandas DataFrame
        Spectral width [m] indexed by S.columns
    """
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    
    mn2 = frequency_moment(S,-2,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m0 = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze()
    mn1 = frequency_moment(S,-1,frequency_bins=frequency_bins).squeeze()
    
    # Eq 16 in IEC 62600-101 
    v = np.sqrt((m0*mn2/np.power(mn1,2))-1)
    v = pd.DataFrame(v, index=S.columns, columns=['v'])
    
    return v


def energy_flux(S, h, rho=1025, g=9.80665):
    """
    Calculates the omnidirectional wave energy flux of the spectra
    
    Parameters
    -----------
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    h: float
        Water depth [m]
    rho: float (optional)
        Water Density [kg/m3]
    g : float (optional)
        Gravitational acceleration [m/s^2]

    Returns
    -------
    J: pandas DataFrame
        Omni-directional wave energy flux [W/m] indexed by S.columns
    """
    # TODO: Add deep water flag
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(rho, (int,float)), 'rho must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'
    
    f = S.index
    
    k = wave_number(f,h,rho,g)
        
    # wave celerity (group velocity)
    Cg = wave_celerity(k,h,g).squeeze()
    
    # Calculating the wave energy flux, Eq 9 in IEC 62600-101 
    delta_f = pd.Series(f).diff()
    delta_f.index = f
    delta_f[f[0]] = delta_f[f[1]] # fill the initial NaN
    
    CgSdelF = S.multiply(delta_f, axis=0).multiply(Cg, axis=0)
    
    J = rho*g*CgSdelF.sum(axis=0)
    
    J = pd.DataFrame(J, index=S.columns, columns=["J"])
    
    return J


def wave_celerity(k, h, g=9.80665):
    """
    Calculates wave celerity (group velocity)
    
    Parameters
    -----------
    k: pandas DataFrame
        Wave number [1/m] indexed by frequency [Hz]
    h: float
        Water depth [m]
    g : float (optional)
        Gravitational acceleration [m/s^2]
        
    Returns
    -------
    Cg: pandas DataFrame
        Water celerity [?] indexed by frequency [Hz]
    """

    assert isinstance(k, pd.DataFrame), 'S must be of type pd.DataFrame'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'

    f = k.index
    k = k.squeeze() # convert to Series for calculation (returns a copy)
    
    # Eq 10 in IEC 62600-101
    Cg = (np.pi*f/k)*(1+(2*h*k)/np.sinh(2*h*k))
    Cg = pd.DataFrame(Cg, index=f, columns=["Cg"])
   
    return Cg


def wave_number(f, h, rho=1025, g=9.80665):
    """
    Calculates wave number
    
    To compute wave number from angular frequency (w), convert w to f before 
    using this function (f = w/2*pi)
    
    Parameters
    -----------
    f: numpy array
        Frequency [Hz]
    h: float
        Water depth [m]
    rho: float (optional)
        Water density [kg/m^3]
    g: float (optional)
        Gravitational acceleration [m/s^2]
    
    Returns
    -------
    k: pandas DataFrame
        Wave number [1/m] indexed by frequency [Hz]
    """
    try:
        f = np.array(f)
    except:
        pass
    assert isinstance(f, np.ndarray), 'f must be of type np.ndarray'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(rho, (int,float)), 'rho must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'
    
    w = 2*np.pi*f # angular frequency
    xi = w/np.sqrt(g/h) # note: =h*wa/sqrt(h*g/h)
    yi = xi*xi/np.power(1.0-np.exp(-np.power(xi,2.4908)),0.4015)
    k0 = yi/h # Initial guess without current-wave interaction
    
    # Eq 11 in IEC 62600-101 using initial guess from Guo (2002)
    def func(kk):
        val = np.power(w,2) - g*kk*np.tanh(kk*h)
        return val
    
    mask = np.abs(func(k0)) > 1e-9
    if mask.sum() > 0:
        k0_mask = k0[mask]
        w = w[mask]
        
        k, info, ier, mesg = _fsolve(func, k0_mask, full_output=True)
        assert ier == 1, 'Wave number not found. ' + mesg
        k0[mask] = k

    k = pd.DataFrame(k0, index=f, columns=['k'])
    
    return k
