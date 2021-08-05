from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve as _fsolve
from itertools import product as _product
from scipy import signal as _signal
import scipy.optimize as optim
import scipy.stats as stats
import pandas as pd
import numpy as np

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
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``.  Defaults to None. 
                
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
def surface_elevation(S, time_index, seed=123, frequency_bins=None,phases=None):
    """
    Calculates wave elevation time-series from spectrum
    
    Parameters
    ------------    
    S: pandas DataFrame
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    time_index: numpy array
        Time used to create the wave elevation time-series [s],
        for example, time = np.arange(0,100,0.01)
    seed: int (optional)
        Random seed
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    phases: numpy array or pandas DataFrame (optional)
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
    assert isinstance(time_index, np.ndarray), ('time_index must be of type' 
            'np.ndarray')
    assert isinstance(seed, (type(None),int)), 'seed must be of type int'
    assert isinstance(frequency_bins, (type(None), np.ndarray, pd.DataFrame)),( 
            "frequency_bins must be of type None, np.ndarray, or pd,DataFrame")
    assert isinstance(phases, (type(None), np.ndarray, pd.DataFrame)), (
            'phases must be of type None, np.ndarray, or pd,DataFrame')

    if frequency_bins is not None:
        assert frequency_bins.squeeze().shape == frequency_bins.squeeze().shape,(
            'shape of frequency_bins must match shape of S')
    if phases is not None:
        assert phases.squeeze().shape == S.squeeze().shape,( 
            'shape of phases must match shape of S')
        
    start_time = time_index[0]
    end_time = time_index[-1]

    f = pd.Series(S.index)
    f.index = f
    
    if frequency_bins is None:        
        delta_f = f.diff()
        #delta_f[0] = f[1]-f[0]

    elif isinstance(frequency_bins, np.ndarray):
        delta_f = pd.Series(frequency_bins, index=S.index)
    elif isinstance(frequency_bins, pd.DataFrame):
        assert len(frequency_bins.columns) == 1, ('frequency_bins must only'
                'contain 1 column')        
        delta_f = frequency_bins.squeeze()

    if phases is None:
        np.random.seed(seed)
        phase = pd.DataFrame(2*np.pi*np.random.rand(S.shape[0], S.shape[1]),
                             index=S.index, columns=S.columns)
    elif isinstance(phases, np.ndarray):
        phase = pd.DataFrame(phases, index=S.index, columns=S.columns)
    elif isinstance(phases, pd.DataFrame):
        phase = phases
        
    phase = phase[start_time:end_time] # Should phase, omega, and A*delta_f be 
                                        #   truncated before computation?
    
    omega = pd.Series(2*np.pi*f) 
    omega.index = f
    omega = omega[start_time:end_time]
    
    # Wave amplitude times delta f, truncated
    A = 2*S 
    A = A.multiply(delta_f, axis=0)
    A = np.sqrt(A)
    A = A.loc[start_time:end_time,:]

    eta = pd.DataFrame(columns=S.columns, index=time_index)
    for mcol in eta.columns:
        # Product of omega and time
        B = np.array([x*y for x,y in _product(time_index, omega)])
        B = B.reshape((len(time_index),len(omega)))
        B = pd.DataFrame(B, index=time_index, columns=omega.index)
    
        C = np.real(np.exp(1j*(B+phase[mcol])))
        C = pd.DataFrame(C, index=time_index, columns=omega.index)

        eta[mcol] = (C*A[mcol]).sum(axis=1)

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
    assert isinstance(S, (pd.Series,pd.DataFrame)), 'S must be of type pd.DataFrame or pd.Series'
    assert isinstance(N, int), 'N must be of type int'
    
    # Eq 8 in IEC 62600-101 
    spec = S[S.index > 0] # omit frequency of 0
    
    f = spec.index 
    fn = np.power(f, N)
    if frequency_bins is None:
        delta_f = pd.Series(f).diff()
        delta_f[0] = f[1]-f[0]
    else:
         
        assert isinstance(frequency_bins, (np.ndarray,pd.Series,pd.DataFrame)),(
         'frequency_bins must be of type np.ndarray or pd.Series')
        delta_f = pd.Series(frequency_bins)

    delta_f.index = f
 
    m = spec.multiply(fn,axis=0).multiply(delta_f,axis=0)
    m = m.sum(axis=0)
    if isinstance(S,pd.Series):
        m = pd.DataFrame(m, index=[0], columns = ['m'+str(N)])
    else:
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
    assert isinstance(S, (pd.Series,pd.DataFrame)), 'S must be of type pd.DataFrame or pd.Series'
    
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
    Calculates wave peak period from spectra
    
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

    assert isinstance(S, (pd.Series,pd.DataFrame)), 'S must be of type pd.DataFrame or pd.Series'
    
    mn1 = frequency_moment(S,-1,frequency_bins=frequency_bins).squeeze() # convert to Series for calculation
    m0  = frequency_moment(S,0,frequency_bins=frequency_bins).squeeze()
    
    # Eq 13 in IEC 62600-101 
    Te = mn1/m0
    if isinstance(S,pd.Series):
            Te = pd.DataFrame(Te, index=[0], columns=['Te'])
    else:
            Te = pd.DataFrame(Te, S.columns, columns=['Te'])
    
    
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


def energy_flux(S, h, deep=False, rho=1025, g=9.80665, ratio=2):
    """
    Calculates the omnidirectional wave energy flux of the spectra
    
    Parameters
    -----------
    S: pandas DataFrame or Series
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    h: float
        Water depth [m]
    deep: bool (optional)
        If True use the deep water approximation. Default False. When
        False a depth check is run to check for shallow water. The ratio
        of the shallow water regime can be changed using the ratio
        keyword.
    rho: float (optional)
        Water Density [kg/m^3]. Default = 1025 kg/m^3
    g : float (optional)
        Gravitational acceleration [m/s^2]. Default = 9.80665 m/s^2
    ratio: float or int (optional)
        Only applied if depth=False. If h/l > ratio,
        water depth will be set to deep. Default ratio = 2.

    Returns
    -------
    J: pandas DataFrame
        Omni-directional wave energy flux [W/m] indexed by S.columns
    """
    assert isinstance(S, (pd.Series,pd.DataFrame)), 'S must be of type pd.DataFrame or pd.Series'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(deep, bool), 'deep must be of type bool'
    assert isinstance(rho, (int,float)), 'rho must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'
    assert isinstance(ratio, (int,float)), 'ratio must be of type int or float'
    
    if deep:
        # Eq 8 in IEC 62600-100, deep water simpilification
        Te = energy_period(S)
        Hm0 = significant_wave_height(S)
        
        coeff = rho*(g**2)/(64*np.pi)

        J = coeff*(Hm0.squeeze()**2)*Te.squeeze()
        if isinstance(S,pd.Series):
            J = pd.DataFrame(J, index=[0], columns=["J"])
        else:
            J = pd.DataFrame(J, S.columns, columns=["J"])
        
        
    else:
        # deep water flag is false
        f = S.index

        k = wave_number(f, h, rho, g)

        # wave celerity (group velocity)
        Cg = wave_celerity(k, h, g, depth_check=True, ratio=ratio).squeeze()

        # Calculating the wave energy flux, Eq 9 in IEC 62600-101
        delta_f = pd.Series(f).diff()
        delta_f.index = f
        delta_f[f[0]] = delta_f[f[1]]  # fill the initial NaN

        CgSdelF = S.multiply(delta_f, axis=0).multiply(Cg, axis=0)

        J = rho * g * CgSdelF.sum(axis=0)

        if isinstance(S,pd.Series):
            J = pd.DataFrame(J, index=[0], columns=["J"])
        else:
            J = pd.DataFrame(J, S.columns, columns=["J"])

    return J


def wave_celerity(k, h, g=9.80665, depth_check=False, ratio=2):
    """
    Calculates wave celerity (group velocity)
    
    Parameters
    ----------
    k: pandas DataFrame or Series
        Wave number [1/m] indexed by frequency [Hz]
    h: float
        Water depth [m]
    g : float (optional)
        Gravitational acceleration [m/s^2]. Default 9.80665 m/s.
    depth_check: bool (optional)
        If True check depth regime. Default False.
    ratio: float or int (optional)
        Only applied if depth_check=True. If h/l > ratio,
        water depth will be set to deep. Default ratio = 2

    Returns
    -------
    Cg: pandas DataFrame
        Water celerity [m/s] indexed by frequency [Hz]
    """
    if isinstance(k, pd.DataFrame):
        k = k.squeeze()

    assert isinstance(k, pd.Series), 'S must be of type pd.Series'
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(g, (int,float)), 'g must be of type int or float'
    assert isinstance(depth_check, bool), 'depth_check must be of type bool'
    assert isinstance(ratio, (int,float)), 'ratio must be of type int or float'

    f = k.index
    k = k.values
    
    if depth_check:           
        l = wave_length(k)

        # get depth regime
        dr = depth_regime(l, h, ratio=ratio)

        # deep frequencies
        df = f[dr]
        dk = k[dr]
        
        # deep water approximation
        dCg = (np.pi * df / dk)
        dCg = pd.DataFrame(dCg, index=df, columns=["Cg"])
        
        # shallow frequencies
        sf = f[~dr]
        sk = k[~dr]
        sCg = (np.pi * sf / sk) * (1 + (2 * h * sk) / np.sinh(2 * h * sk))
        sCg = pd.DataFrame(sCg, index = sf, columns = ["Cg"])
        
        Cg = pd.concat([dCg, sCg]).sort_index()

    else: 
        # Eq 10 in IEC 62600-101
        Cg = (np.pi * f / k) * (1 + (2 * h * k) / np.sinh(2 * h * k))
        Cg = pd.DataFrame(Cg, index=f, columns=["Cg"])        
   
    return Cg

def wave_length(k):
    """
    Calculates wave length from wave number
    To compute: 2*pi/wavenumber

    Parameters
    -------------
    k: pandas Dataframe
        Wave number [1/m] indexed by frequency

    Returns
    ---------
    l: float or array
        Wave length [m] indexed by frequency
    """
    if isinstance(k, (int, float, list)):
        k = np.array(k)    
    elif isinstance(k, pd.DataFrame):        
        k = k.squeeze().values
    elif isinstance(k, pd.Series):           
        k = k.values

    assert isinstance(k, np.ndarray), 'k must be array-like'

    l = 2*np.pi/k
    
    return l

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
        f = np.atleast_1d(np.array(f))
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

def depth_regime(l, h, ratio=2):
    '''
    Calculates the depth regime based on wavelength and height
    Deep water: h/l > ratio
    This function exists so sinh in wave celerity doesn't blow
    up to infinity.

    P.K. Kundu, I.M. Cohen (2000) suggest h/l >> 1 for deep water (pg 209)
    Same citation as above, they also suggest for 3% accuracy, h/l > 0.28 (pg 210)
    However, since this function allows multiple wavelengths, higher ratio
    numbers are more accurate across varying wavelengths.

    Parameters
    ----------
    l: array-like
        wavelength [m]
    h: float or int
        water column depth [m]
    ratio: float or int (optional)
        if h/l > ratio, water depth will be set to deep. Default ratio = 2  

    Returns
    -------
    depth_reg: boolean or boolean array
        Boolean True if deep water, False otherwise
    '''
        
    if isinstance(l, (int, float, list)):
        l = np.array(l)    
    elif isinstance(l, pd.DataFrame):        
        l = l.squeeze().values
    elif isinstance(l, pd.Series):           
        l = l.values
    
    assert isinstance(l, (np.ndarray)), "l must be array-like"
    assert isinstance(h, (int, float)), "h must be of type int or float"

    depth_reg = h/l > ratio
    
    return  depth_reg


def environmental_contour(x1, x2, dt, period, **kwargs):
    '''
    Calculates environmental contours of extreme sea
    states using the improved joint probability distributions
    with the inverse first-order reliability method (I-FORM)
    probability for the desired return period (`period`). Given the
    period of interest, a circle of iso-probability is created
    in the principal component analysis (PCA) joint probability
    (`x1`, `x2`) reference frame.
    Using the joint probability value, the cumulative distribution
    function (CDF) of the marginal distribution is used to find
    the quantile of each component.
    Finally, using the improved PCA methodology,
    the component 2 contour lines are calculated from component 1 using
    the relationships defined in Exkert-Gallup et. al. 2016.

    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., & 
    Neary, V. S. (2016). Application of principal component 
    analysis (PCA) and improved joint probability distributions to 
    the inverse first-order reliability method (I-FORM) for predicting 
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        	
    dt : int or float
        `x1` and `x2` sample rate (seconds)
    period : int, float, or numpy array 
        Desired return period (years) for calculation of environmental
        contour, can be a scalar or a vector.
    **kwargs : optional        
        PCA: dict
            If provided, the principal component analysis (PCA) on x1, x2 
            is skipped. The PCA will be the same for a given x1, x2 
            therefore this step may be skipped if multiple calls to 
            environmental contours are made for the same x1, x2 pair. 
            The PCA dict may be obtained by setting return_PCA=True.
        bin_size : int
            Data points in each bin for the PCA. Default bin_size=250.		
        nb_steps : int
            Discretization of the circle in the normal space used for
            I-FORM calculation. Default nb_steps=1000.
        return_PCA: boolean
            Default False, if True will retun the PCA dictionary 

    Returns
    -------
    x1_contour : numpy array 
        Calculated x1 values along the contour boundary following
        return to original input orientation.
    x2_contour : numpy array 
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    PCA: dict (optional)
	    principal component analysis dictionary 
        Keys:
        -----       
        'principal_axes': sign corrected PCA axes 
        'shift'         : The shift applied to x2 
        'x1_fit'        : gaussian fit of x1 data
        'mu_param'      : fit to _mu_fcn
        'sigma_param'   : fit to _sig_fits 

    '''
    try: x1 = np.array(x1); 
    except: pass
    try: x2 = np.array(x2); 
    except: pass
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(dt, (int,float)), 'dt must be of type int or float'
    assert isinstance(period, (int,float,np.ndarray)), ('period must be'
                                          'of type int, float, or array')    
    PCA = kwargs.get("PCA", None)
    bin_size = kwargs.get("bin_size", 250)
    nb_steps = kwargs.get("nb_steps", 1000)
    return_PCA = kwargs.get("return_PCA", False)
    assert isinstance(PCA, (dict, type(None))), 'If specified PCA must be a dict'
    assert isinstance(bin_size, int), 'bin_size must be of type int'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    assert isinstance(return_PCA, bool), 'return_PCA must be of type bool'
    
    if isinstance(period, np.ndarray) and len(period) > 1 and period.ndim == 1:
        period = period.reshape(-1,1)    
    
    if PCA == None:
        PCA = _principal_component_analysis(x1, x2, bin_size=bin_size)
	
    dt_yrs = dt / ( 3600 * 24 * 365 )
    exceedance_probability = 1 / ( period / dt_yrs)
    iso_probability_radius = stats.norm.ppf((1 - exceedance_probability), 
                                             loc=0, scale=1)  
    discretized_radians = np.linspace(0, 2 * np.pi, nb_steps)
    
    x_component_iso_prob = iso_probability_radius * \
                            np.cos(discretized_radians)
    y_component_iso_prob = iso_probability_radius * \
                            np.sin(discretized_radians)
    
    x_quantile = stats.norm.cdf(x_component_iso_prob, loc=0, scale=1)
    y_quantile = stats.norm.cdf(y_component_iso_prob, loc=0, scale=1)
    
    # Use the inverse of cdf to calculate component 1 values           
    component_1 = stats.invgauss.ppf(x_quantile, 
                                     mu   =PCA['x1_fit']['mu'],
                                     loc  =PCA['x1_fit']['loc'], 
                                     scale=PCA['x1_fit']['scale'] )
    
    # Find Component 2 mu using first order linear regression
    mu_slope     = PCA['mu_fit'].slope
    mu_intercept = PCA['mu_fit'].intercept        
    component_2_mu = mu_slope * component_1 + mu_intercept
    
    # Find Componenet 2 sigma using second order polynomial fit
    sigma_polynomial_coeffcients =PCA['sigma_fit'].x
    component_2_sigma = np.polyval(sigma_polynomial_coeffcients, component_1)
                
    # Use calculated mu and sigma values to calculate C2 along the contour
    component_2 = stats.norm.ppf(y_quantile,
                                 loc  =component_2_mu, 
                                 scale=component_2_sigma)
                           
    # Convert contours back to the original reference frame
    principal_axes = PCA['principal_axes']
    shift = PCA['shift']
    pa00 = principal_axes[0, 0]
    pa01 = principal_axes[0, 1]

    x1_contour = (( pa00 * component_1 + pa01 * (component_2 - shift)) / \
                  (pa01**2 + pa00**2))                         
    x2_contour = (( pa01 * component_1 - pa00 * (component_2 - shift)) / \
                  (pa01**2 + pa00**2))                                    
    
    # Assign 0 value to any negative x1 contour values
    x1_contour = np.maximum(0, x1_contour)  
 
    if return_PCA:
        return np.transpose(x1_contour), np.transpose(x2_contour), PCA
    return np.transpose(x1_contour), np.transpose(x2_contour)

def _principal_component_analysis(x1, x2, bin_size=250):
    '''
    Performs a modified principal component analysis (PCA) 
    [Eckert et. al 2016] on two variables (`x1`, `x2`). The additional
    PCA is performed in 5 steps:
    1) Transform `x1` & `x2` into the principal component domain and shift
       the y-axis so that all values are positive and non-zero
    2) Fit the `x1` data in the transformed reference frame with an 
       inverse Gaussian Distribution
    3) Bin the transformed data into groups of size bin and find the 
       mean of `x1`, the mean of `x2`, and the standard deviation of `x2`
    4) Perform a first-order linear regression to determine a continuous
       the function relating the mean of the `x1` bins to mean of the `x2` bins
    5) Find a second-order polynomial which best relates the means of 
       `x1` to the standard deviation of `x2` using constrained optimization
         
    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., & 
    Neary, V. S. (2016). Application of principal component 
    analysis (PCA) and improved joint probability distributions to 
    the inverse first-order reliability method (I-FORM) for predicting 
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        
    bin_size : int
        Number of data points in each bin 
        
    Returns
    -------
    PCA: dict 
       Keys:
       -----       
       'principal_axes': sign corrected PCA axes 
       'shift'         : The shift applied to x2 
       'x1_fit'        : gaussian fit of x1 data
       'mu_param'      : fit to _mu_fcn
       'sigma_param'   : fit to _sig_fits            
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(bin_size, int), 'bin_size must be of type int'
    # Step 0: Perform Standard PCA          
    mean_location=0    
    x1_mean_centered = x1 - x1.mean(axis=0)
    x2_mean_centered = x2 - x2.mean(axis=0)
    n_samples_by_n_features = np.column_stack((x1_mean_centered, 
                                               x2_mean_centered))
    pca = skPCA(n_components=2)                                               
    pca.fit(n_samples_by_n_features)
    principal_axes = pca.components_
    
    # STEP 1: Transform data into new reference frame
    # Apply correct/expected sign convention    
    principal_axes = abs(principal_axes)  
    principal_axes[1, 1] = -principal_axes[1, 1]  

    # Rotate data into Principal direction 
    x1_and_x2 = np.column_stack((x1, x2))
    x1_x2_components = np.dot(x1_and_x2, principal_axes)  
    x1_components = x1_x2_components[:, 0]
    x2_components = x1_x2_components[:, 1]

    # Apply shift to Component 2 to make all values positive
    shift = abs(min(x2_components)) + 0.1
    x2_components = x2_components + shift 
    
    # STEP 2: Fit Component 1 data using a Gaussian Distribution
    x1_sorted_index = x1_components.argsort()
    x1_sorted = x1_components[x1_sorted_index]
    x2_sorted = x2_components[x1_sorted_index]
    
    x1_fit_results = stats.invgauss.fit(x1_sorted, floc=mean_location)
    x1_fit = { 'mu'    : x1_fit_results[0],
               'loc'   : x1_fit_results[1],
               'scale' : x1_fit_results[2]}
    
    # Step 3: Bin Data & find order 1 linear relation between x1 & x2 means
    N = len(x1)  
    minimum_4_bins = np.floor(N*0.25)
    if bin_size > minimum_4_bins:
        bin_size = minimum_4_bins
        msg=('To allow for a minimum of 4 bins the bin size has been' +
             f'set to {minimum_4_bins}')
        print(msg)

    N_multiples = N // bin_size
    max_N_multiples_index  =  N_multiples*bin_size
    
    x1_integer_multiples_of_bin_size = x1_sorted[0:max_N_multiples_index]    
    x2_integer_multiples_of_bin_size = x2_sorted[0:max_N_multiples_index] 
    
    x1_bins = np.split(x1_integer_multiples_of_bin_size, 
                       N_multiples)
    x2_bins = np.split(x2_integer_multiples_of_bin_size, 
                       N_multiples)
    
    x1_last_bin = x1_sorted[max_N_multiples_index:]    
    x2_last_bin = x2_sorted[max_N_multiples_index:]    
    
    x1_bins.append(x1_last_bin)
    x2_bins.append(x2_last_bin)
    
    x1_means = np.array([]) 
    x2_means = np.array([]) 
    x2_sigmas  = np.array([])     
    
    for x1_bin, x2_bin in zip(x1_bins, x2_bins):                    
        x1_means = np.append(x1_means, x1_bin.mean())                         
        x2_means = np.append(x2_means, x2_bin.mean())         
        x2_sigmas  = np.append(x2_sigmas, x2_bin.std()) 
    
    mu_fit = stats.linregress(x1_means, x2_means)    
    
    # STEP 4: Find order 2 relation between x1_mean and x2 standard deviation
    sigma_polynomial_order=2
    sig_0 = 0.1 * np.ones(sigma_polynomial_order+1)
    
    def _objective_function(sig_p, x1_means, x2_sigmas):
        return mean_squared_error(np.polyval(sig_p, x1_means), x2_sigmas)
    
    # Constraint Functions
    y_intercept_gt_0 = lambda sig_p: (sig_p[2])
    sig_polynomial_min_gt_0 = lambda sig_p: (sig_p[2] - (sig_p[1]**2) / \
                                             (4 * sig_p[0]))    
    constraints = ({'type': 'ineq', 'fun': y_intercept_gt_0},
                   {'type': 'ineq', 'fun': sig_polynomial_min_gt_0})    
    
    sigma_fit = optim.minimize(_objective_function, x0=sig_0, 
                               args=(x1_means, x2_sigmas),
                               method='SLSQP',constraints=constraints)     
    
    PCA = {'principal_axes': principal_axes, 
           'shift'         : shift, 
           'x1_fit'        : x1_fit, 
           'mu_fit'        : mu_fit, 
           'sigma_fit'     : sigma_fit }
    
    return PCA
