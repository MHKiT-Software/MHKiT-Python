from statsmodels.nonparametric.kde import KDEUnivariate 
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve as _fsolve
from itertools import product as _product
from scipy import signal as _signal
import matplotlib.pyplot as plt
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
                                          ' of type int, float, or array')    
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
	
    results = iso_prob_and_quantile(dt, period, nb_steps)
    x_component_iso_prob = results['x_component_iso_prob'] 
    y_component_iso_prob = results['y_component_iso_prob'] 
    x_quantile = results['x_quantile'] 
    y_quantile =results['y_quantile'] 
    
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


def iso_prob_and_quantile(dt, period, nb_steps):
    '''
    Calculates the iso-probability and the x, y quantiles along
    the iso probability radius
    
    Parameters
    ----------
    dt : int or float
        `x1` and `x2` sample rate (seconds)
    period: int, float
        Return period of interest in years
    nb_steps : int
        Discretization of the circle in the normal space. 
        Default nb_steps=1000.
    
    Returns
    -------
    results: Dictionay
        Dictionary of the iso-probability results
        Keys:
        'exceedance_probability' - probaility of exceedance
        'x_component_iso_prob' - x-component of iso probability circle
        'y_component_iso_prob' - y-component of iso probability circle
        'x_quantile' - CDF of x-component
        'y_quantile' - CDF of y-component
    '''
    assert isinstance(dt, (int,float)), 'dt must be of type int or float'
    assert isinstance(period, (int,float)), ('period must be'
                                           'of type int or float')
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
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
    
    results = { 'exceedance_probability' : exceedance_probability,
                'x_component_iso_prob': x_component_iso_prob,
                'y_component_iso_prob': y_component_iso_prob,
                'x_quantile': x_quantile,
                'y_quantile': y_quantile}
    return results


def copula_parameters(x1, x2, min_bin_count, initial_bin_max_val, bin_val_size):
    '''
    Returns an estimate of the Weibull and Lognormal distribution for
    x1 and x2 respectively. Additioanlly returns the estimates of the
    coeffcients from the mean and standard deviation of the Log of x2 
    given x1.
    
    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data 
    min_bin_count: int
        Sets the minimum number of bins allowed
    initial_bin_max_val: int, float
        Sets the max value of the first bin    
    bin_val_size: int, float
        The size of each bin after the initial bin
    
    Returns
    -------
    para_dist_1: array
        Weibull distribution parameters for  for component 1 
    para_dist_2: array
        Lognormal distribution parameters for component 2
    mean_cond: array
        Estimate coefficients of mean of Ln(x2|x1)
    std_cond: array
        Estimate coefficients of standard deviation of Ln(x2|x1)
    '''  
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(min_bin_count, int),'min_bin_count must be of type int'
    assert isinstance(bin_val_size, (int, float)), 'bin_val_size must be of type int or float'
    assert isinstance(initial_bin_max_val, (int, float)), 'initial_bin_max_val must be of type int or float'    
    
    # Binning
    x1_sorted_index = x1.argsort()
    x1_sorted = x1[x1_sorted_index]
    x2_sorted = x2[x1_sorted_index]
    
    # Because x1 is sorted we can find the max index using the following logic
    ind = np.array([])
    N_vals_lt_limit = sum(x1_sorted <= initial_bin_max_val)
    ind = np.append(ind, N_vals_lt_limit)

    # Make sure first bin isn't empty or too small to avoid errors        
    while ind == 0 or ind < min_bin_count:         
        ind = np.array([])    
        initial_bin_max_val += bin_val_size
        N_vals_lt_limit = sum(x1_sorted <= initial_bin_max_val)
        ind = np.append(ind, N_vals_lt_limit) 

    # Add bins until the total number of vals in between bins is less than the minimum bin size
    i=0
    bin_size_i=np.inf
    while bin_size_i >= min_bin_count:
        i+=1
        bin_i_max_val = initial_bin_max_val + bin_val_size*(i)
        N_vals_lt_limit = sum(x1_sorted <= bin_i_max_val)
        ind = np.append(ind, N_vals_lt_limit)
        bin_size_i = ind[i]-ind[i-1]
        
    # Weibull distribution parameters for  for component 1 using MLE    
    para_dist_1=stats.exponweib.fit(x1_sorted,floc=0,fa=1)
    # Lognormal distribution parameters for component 2 using MLE
    para_dist_2=stats.norm.fit(np.log(x2_sorted))
        
    # Parameters for conditional distribution of T|Hs for each bin
    num=len(ind) # num+1: number of bins
    para_dist_cond = []
    hss = []

    # Bin zero special case (lognormal dist over only 1 bin)
    # parameters for zero bin
    ind0 = range(0, int(ind[0]))
    x2_log0 = np.log(x2_sorted[ind0])
    x2_lognormal_dist0 = stats.norm.fit(x2_log0)
    para_dist_cond.append(x2_lognormal_dist0)  
    # mean of x1 (component 1 for zero bin)
    x1_bin0 = x1_sorted[range(0, int(ind[0])-1)]
    hss.append(np.mean(x1_bin0)) 

    # Intialize special case 2-bin lognormal Dist 
    bin_range = 2
    # parameters for 1 bin
    ind1 = range(0, int(ind[1]))
    x2_log1 = np.log(x2_sorted[ind1])
    x2_lognormal_dist1 = stats.norm.fit(x2_log1)
    para_dist_cond.append(x2_lognormal_dist1) 

    # mean of Hs (component 1 for bin 1)
    hss.append(np.mean(x1_sorted[range(0,int(ind[1])-1)])) 

    # lognormal Dist (lognormal dist over only 2 bins)
    for i in range(2,num):
        ind_i = range(int(ind[i-2]), int(ind[i]))
        x2_log_i = np.log(x2_sorted[ind_i])
        x2_lognormal_dist_i = stats.norm.fit(x2_log_i)
        para_dist_cond.append(x2_lognormal_dist_i);
        
        hss.append(np.mean(x1_sorted[ind_i]))

    # Estimate coefficient using least square solution (mean: third order, sigma: 2nd order)
    ind_f = range(int(ind[num-2]),int(len(x1)))
    x2_log_f = np.log(x2_sorted[ind_f])
    x2_lognormal_dist_f = stats.norm.fit(x2_log_f)
    para_dist_cond.append(x2_lognormal_dist_f)  # parameters for last bin

    # mean of Hs (component 1 for last bin)
    hss.append(np.mean(x1_sorted[ind_f])) 

    para_dist_cond = np.array(para_dist_cond)
    hss = np.array(hss)

    # cubic in Hs: a + bx + cx**2 + dx**3
    phi_mean = np.column_stack((np.ones(num+1), hss, hss**2, hss**3))
    # quadratic in Hs  a + bx + cx**2
    phi_std = np.column_stack((np.ones(num+1), hss, hss**2))

    # Estimate coefficients of mean of Ln(T|Hs)(vector 4x1) (cubic in Hs)
    mean_cond = np.linalg.lstsq(phi_mean, para_dist_cond[:,0])[0]
    # Estimate coefficients of standard deviation of Ln(T|Hs) (vector 3x1) (quadratic in Hs)
    std_cond = np.linalg.lstsq(phi_std, para_dist_cond[:,1])[0]
    
    return para_dist_1, para_dist_2, mean_cond, std_cond


def copula(x1, x2, dt, period, method, **kwargs):
    '''
    Returns a Dictionary of x1 and x2 copula components for each copula
    method passed. Methods include: 
    gaussian, gumbel, clayton, rosenblatt, nonparametric gaussian,
    nonparametric clayton, nonparametric gumbel, bivariate KDE,
    log bivariate KDE 
    
    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data 
    dt : int or float
        `x1` and `x2` sample rate (seconds)
    period: int, float
        Return period of interest in years
    method: string or list
        Copula method to apply. Options include ['gaussian', 'gumbel', 
         'clayton', 'rosenblatt', 'nonparametric_gaussian',
         'nonparametric_clayton', 'nonparametric_gumbel', 'bivariate_KDE'
         'bivariate_KDE_log']

    **kwargs
        min_bin_count: int
            Passed to copula_parameters to sets the minimum number of bins allowed. Deault = 40.
        initial_bin_max_val: int, float
            Passed to copula_parameters to set the max value of the first
            bin. Default = 1.
        bin_val_size: int, float
            Passed to copula_parameters to set the size of each bin after the initial bin.  Default 0.25.            
        nb_steps: int
            Discretization of the circle in the normal space used for
            copula component calculation. Default nb_steps=1000.
        bandwidth:
            Must specify bandwidth for bivariate KDE method. Default = None.
        Ndata_bivariate_KDE: int
            Must specify bivariate KDE method. Defines the contour space
            from which samples are taken. Default = 100.
        max_x1: float
            Defines the max value of x1 to discretize the KDE space
        max_x2: float
            Defines the max value of x2 to discretize the KDE space
    
    Returns
    -------
    copulas: Dictionary
        Dictionary of x1 and x2 copula components for each copula method
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
    
    bin_val_size = kwargs.get("bin_val_size", 0.25)
    nb_steps = kwargs.get("nb_steps", 1000)
    initial_bin_max_val=kwargs.get("initial_bin_max_val",1.)
    min_bin_count=kwargs.get("min_bin_count",40)
    bandwidth=kwargs.get("bandwidth", None)
    Ndata_bivariate_KDE = kwargs.get("Ndata_bivariate_KDE", 100) 
    max_x1 = kwargs.get("max_x1", None)
    max_x2 = kwargs.get("max_x2", None)
    
    assert isinstance(bin_val_size, (int, float)), 'bin_val_size must be of type int or float'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    assert isinstance(min_bin_count, int), 'min_bin_count must be of type int'
    assert isinstance(initial_bin_max_val, (int, float)), 'initial_bin_max_val must be of type int or float'
    if bandwidth == None:
        assert(not 'bivariate_KDE' in method), 'Must specify keyword bandwidth with bivariate KDE method'

    if isinstance(method, str):
        method = [method]

    results = iso_prob_and_quantile(dt, period, nb_steps)
    
    para_dist_1, para_dist_2, mean_cond, std_cond = copula_parameters(x1, x2, min_bin_count, initial_bin_max_val, bin_val_size)
    results['para_dist_1'] = para_dist_1
    results['para_dist_2'] = para_dist_2
    results['mean_cond'] = mean_cond
    results['std_cond'] = std_cond
    
    x_quantile = results['x_quantile'] 
    a=para_dist_1[0]
    c=para_dist_1[1]
    loc=para_dist_1[2]
    scale=para_dist_1[3]

    component_1 = stats.exponweib.ppf(x_quantile, a, c, loc=loc, scale=scale)
    
    copula_functions={'gaussian' : 
                         {'func':_gaussian_copula,
                         'vals':(x1, x2, results, component_1)},
                      'gumbel' : 
                          {'func':_gumbel_copula,
                           'vals':(x1, x2, results, component_1, nb_steps)},
                      'clayton' : 
                          {'func': _clayton_copula,
                           'vals':(x1, x2, results, component_1)},
                      'rosenblatt' : 
                          {'func':_rosenblatt_copula,
                          'vals':(x1, x2, results, component_1)},
                      'nonparametric_gaussian' : 
                          {'func':_nonparametric_gaussian_copula,
                           'vals':(x1, x2, results, nb_steps)},
                      'nonparametric_clayton' : 
                          {'func':_nonparametric_clayton_copula,
                           'vals':(x1, x2, results, nb_steps)},
                      'nonparametric_gumbel' : 
                          {'func':_nonparametric_gumbel_copula,
                           'vals':(x1, x2, results, nb_steps)},
                      'bivariate_KDE': 
                          {'func' :_bivariate_KDE,
                           'vals' : (x1, x2, bandwidth, results, nb_steps,
                                     Ndata_bivariate_KDE, max_x1, 
                                     max_x2 )},
                      'bivariate_KDE_log': 
                          {'func' :_bivariate_KDE,
                           'vals' : (x1, x2, bandwidth, results, nb_steps,
                                     Ndata_bivariate_KDE,max_x1, max_x2,
                                     {'log_transform':True})},                                     
                      
                      }
    copulas={}
    for meth in method:
        vals = copula_functions[meth]['vals']
        component_1, component_2 = copula_functions[meth]['func'](*vals)
        copulas[f'{meth}_x1'] = component_1
        copulas[f'{meth}_x2'] = component_2
    return copulas 
    

def _gaussian_copula(x1, x2, results, component_1):
    '''
    Extreme Sea State Gaussian Copula Contour function.
    This function calculates environmental contours of extreme sea states using a Gaussian copula and the inverse first-order reliability method.

    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        	
    results: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
          
    Returns
    -------    
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    component_2_Gaussian      
        Calculated x2 values along the contour boundary following
        return to original input orientation. 
    '''
    try: x1 = np.array(x1); 
    except: pass
    try: x2 = np.array(x2); 
    except: pass
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    # assert results is dict with key  'x_component_iso_prob'
    #'y_component_iso_prob','para_dist_2'
    assert isinstance(component_1, np.ndarray), 'x2 must be of type np.ndarray'
   
    x_component_iso_prob = results['x_component_iso_prob'] 
    y_component_iso_prob = results['y_component_iso_prob'] 
    
    # Calculate Kendall's tau
    tau=stats.kendalltau(x2,x1)[0] 
    rho_gau=np.sin(tau*np.pi/2.)

    z2_Gauss=stats.norm.cdf(y_component_iso_prob*np.sqrt(1.-rho_gau**2.)+rho_gau*x_component_iso_prob);

    para_dist_2 = results['para_dist_2'] 
    s=para_dist_2[1]
    loc=0
    scale=np.exp(para_dist_2[0])

    # lognormal inverse
    component_2_Gaussian = stats.lognorm.ppf(z2_Gauss, s=s, loc=loc, 
                                             scale=scale) 
        
    return component_1, component_2_Gaussian
    

def _gumbel_density(u, alpha):
    ''' 
    Calculates the Gumbel copula density
    
    Parameters
    ----------
    u: np.array
        Vector of equally spaced points between 0 and twice the
            maximum value of T.
    alpha: float
        Copula parameter. Must be greater than or equal to 1.
    
    Returns
    -------
    y: np.array
        Copula density function.
    '''    
    #Ignore divide by 0 warnings and resulting NaN warnings
    np.seterr(all='ignore')        
    v = -np.log(u)
    v = np.sort(v, axis=0)
    vmin = v[0, :]
    vmax = v[1, :]
    nlogC = vmax * (1 + (vmin / vmax) ** alpha) ** (1 / alpha)
    y = (alpha - 1 +nlogC)*np.exp(-nlogC+np.sum((alpha-1)*np.log(v)+v, axis =0) +(1-2*alpha)*np.log(nlogC))
    np.seterr(all='warn')

    return(y) 


def _gumbel_copula(x1, x2, results, component_1, nb_steps):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Gumbel copula and the inverse first-order reliability
    method.
    
    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        	
    results: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
          
    Returns
    -------    
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    component_2_Gumbel: array   
        Calculated x2 values along the contour boundary following
        return to original input orientation. 
    '''
    try: x1 = np.array(x1); 
    except: pass
    try: x2 = np.array(x2); 
    except: pass
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    # assert results is dict with key  'x_component_iso_prob'
    #'y_component_iso_prob','para_dist_2'
    assert isinstance(component_1, np.ndarray), 'x2 must be of type np.ndarray'
    
    x_quantile = results['x_quantile'] 
    y_quantile =results['y_quantile'] 
    para_dist_2 = results['para_dist_2'] 

    # Calculate Kendall's tau
    tau=stats.kendalltau(x2,x1)[0] 
    theta_gum = 1./(1.-tau)
    
    min_limit_2=0
    max_limit_2= np.ceil(np.amax(x2)*2)
    Ndata=1000
    
    x = np.linspace(min_limit_2, max_limit_2, Ndata)
    
    s=para_dist_2[1]
    scale=np.exp(para_dist_2[0])   
    z2 = stats.lognorm.cdf(x,s=s,loc=0,scale=scale)

    component_2_Gumbel = np.zeros(nb_steps)
    for k in range(nb_steps):
        z1 = np.array([x_quantile[k]]*Ndata)
        Z = np.array((z1,z2))
        Y = _gumbel_density(Z, theta_gum) 
        Y =np.nan_to_num(Y)
        # pdf 2|1, f(comp_2|comp_1)=c(z1,z2)*f(comp_2)
        p_x_x1 = Y*(stats.lognorm.pdf(x, s=s, loc=0, scale=scale)) 
        # Estimate CDF from PDF
        dum = np.cumsum(p_x_x1)
        cdf = dum/(dum[Ndata-1]) 
        # Result of conditional CDF derived based on Gumbel copula
        table = np.array((x, cdf)) 
        table = table.T
        for j in range(Ndata):
            if y_quantile[k] <= table[0,1]:
                component_2_Gumbel[k] = min(table[:,0])
                break
            elif y_quantile[k] <= table[j,1]:
                component_2_Gumbel[k] = (table[j,0]+table[j-1,0])/2
                break
            else:
                component_2_Gumbel[k] = table[:,0].max()
                
    return component_1, component_2_Gumbel
    
 
def _clayton_copula(x1, x2, results, component_1):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Clayton copula and the inverse first-order reliability
    method.
    
    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        	
    results: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
          
    Returns
    -------    
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    component_2_Clayton: array   
        Calculated x2 values along the contour boundary following
        return to original input orientation.     
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    # assert results is dict with key  'x_component_iso_prob'
    #'y_component_iso_prob','para_dist_2'
    assert isinstance(component_1, np.ndarray), 'x2 must be of type np.ndarray'

    x_quantile = results['x_quantile'] 
    y_quantile =results['y_quantile']     
    para_dist_2 = results['para_dist_2'] 

    # Calculate Kendall's tau
    tau=stats.kendalltau(x2,x1)[0] 
    
    theta_clay = (2.*tau)/(1.-tau)

    s=para_dist_2[1]
    scale=np.exp(para_dist_2[0])
    z2_Clay=((1.-x_quantile**(-theta_clay)+x_quantile**(-theta_clay)/y_quantile)**(theta_clay/(1.+theta_clay)))**(-1./theta_clay)
    # lognormal inverse
    component_2_Clayton = stats.lognorm.ppf(z2_Clay,s=s,loc=0,scale=scale) 
 
    return component_1, component_2_Clayton


def _rosenblatt_copula(x1, x2, results, component_1):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Rosenblatt transformation and the inverse first-order
    reliability method.
    
    Parameters
    ----------
    x1: numpy array 
        Component 1 data
    x2: numpy array 
        Component 2 data        	
    results: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
          
    Returns
    -------    
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not specifically used in this calculation but is passed through to
        create a consistient output from all copula methods.
    component_2_Rosenblatt: array   
        Calculated x2 values along the contour boundary following
        return to original input orientation.  
    '''
    try: x1 = np.array(x1); 
    except: pass
    try: x2 = np.array(x2); 
    except: pass
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    # assert results is dict with key  'x_component_iso_prob'
    #'y_component_iso_prob','para_dist_2'
    assert isinstance(component_1, np.ndarray), 'x2 must be of type np.ndarray'
    
    y_quantile =results['y_quantile'] 
    mean_cond = results['mean_cond'] 
    std_cond = results['std_cond'] 

    # mean of Ln(T) as a function of x1
    lamda_cond=mean_cond[0]+mean_cond[1]*component_1+mean_cond[2]*component_1**2+mean_cond[3]*component_1**3
    # Standard deviation of Ln(x2) as a function of x1
    sigma_cond=std_cond[0]+std_cond[1]*component_1+std_cond[2]*component_1**2                                
    # lognormal inverse
    component_2_Rosenblatt = stats.lognorm.ppf(y_quantile,s=sigma_cond,loc=0,scale=np.exp(lamda_cond))
 
    return component_1, component_2_Rosenblatt  
  

def _nonparametric_copula_parameters(x1, x2, max_x1=None, max_x2=None,
    nb_steps=1000):
    '''
    Calculates nonparametric copula parameters
    
    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data 
    max_x1: float
        Defines the max value of x1 to discretize the KDE space
    max_x2:float
        Defines the max value of x2 to discretize the KDE space
    nb_steps: int
        number of points used to discritize KDE space
    
    Returns
    -------
    nonpara_dist_1:
        x1 points in KDE space and Nonparametric CDF for x1
    nonpara_dist_2:
        x2 points in KDE space and Nonparametric CDF for x2
    nonpara_pdf_2:
        x2 points in KDE space and Nonparametric PDF for x2
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    if not max_x1:
        max_x1 = x1.max()*2
    if not max_x2:
        max_x2 = x2.max()*2
    assert isinstance(max_x1, float), 'max_x1 must be of type float'
    assert isinstance(max_x2, float), 'max_x2 must be of type float'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
    # Binning
    x1_sorted_index = x1.argsort()
    x1_sorted = x1[x1_sorted_index]
    x2_sorted = x2[x1_sorted_index]
    
    # Calcualte KDE bounds (potential input)
    min_limit_1 = 0
    min_limit_2 = 0
    
    # Discretize for KDE
    pts_x1 = np.linspace(min_limit_1, max_x1, nb_steps) 
    pts_x2 = np.linspace(min_limit_2, max_x2, nb_steps)
    
    # Calculate optimal bandwidth for T and Hs
    sig = stats.median_abs_deviation(x2_sorted)
    num = float(len(x2_sorted))
    bwT = sig*(4.0/(3.0*num))**(1.0/5.0)
    
    sig = stats.median_abs_deviation(x1_sorted)
    num = float(len(x1_sorted))
    bwHs = sig*(4.0/(3.0*num))**(1.0/5.0)
    
    # Nonparametric PDF for x2
    temp = KDEUnivariate(x2_sorted)
    temp.fit(bw = bwT)
    f_x2 = temp.evaluate(pts_x2)
    
    # Nonparametric CDF for x1
    temp = KDEUnivariate(x1_sorted)
    temp.fit(bw = bwHs)
    tempPDF = temp.evaluate(pts_x1)
    F_x1 = tempPDF/sum(tempPDF)
    F_x1 = np.cumsum(F_x1)
    
    # Nonparametric CDF for x2
    F_x2 = f_x2/sum(f_x2)
    F_x2 = np.cumsum(F_x2)
    
    nonpara_dist_1 = np.transpose(np.array([pts_x1, F_x1]))
    nonpara_dist_2 = np.transpose(np.array([pts_x2, F_x2]))
    nonpara_pdf_2 = np.transpose(np.array([pts_x2, f_x2]))
    
    return nonpara_dist_1, nonpara_dist_2, nonpara_pdf_2


def __nonparametric_component(z, nonpara_dist, nb_steps):
    '''
    Generalized method for calculating copula components
    
    Parameters
    ----------
    z: array
        CDF of isoprobability
    nonpara_dist: array
        x1 or x2 points in KDE space and Nonparametric CDF for x1 or x2
    nb_steps: int

    Returns
    -------
    component: array
        nonparametic component values   
    '''
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
    component=np.zeros(nb_steps)
    for k in range(0,nb_steps):
        for j in range(0,np.size(nonpara_dist,0)):
            if z[k] <= nonpara_dist[0,1]: 
                component[k] = min(nonpara_dist[:,0]) 
                break
            elif z[k] <= nonpara_dist[j,1]: 
                component[k] = (nonpara_dist[j,0] + nonpara_dist[j-1,0])/2
                break
            else:
                component[k]= max(nonpara_dist[:,0])
    return component


def _nonparametric_gaussian_copula(x1, x2, results, nb_steps):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Gaussian copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data    
    results: Dictionay
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.        
        
    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula 
    component_2_np_gaussian: array
        Component 2 nonparametric Gaussian copula 
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
    x_component_iso_prob = results['x_component_iso_prob'] 
    y_component_iso_prob = results['y_component_iso_prob'] 
    
    # Copula parameters
    nonpara_dist_1, nonpara_dist_2, nonpara_pdf_2 =_nonparametric_copula_parameters(x1, x2, nb_steps=nb_steps)

    # Calculate Kendall's tau    
    tau = stats.kendalltau(x2, x1)[0]
    rho_gau=np.sin(tau*np.pi/2.)

    # Component 1
    z1=stats.norm.cdf(x_component_iso_prob)
    z2=stats.norm.cdf(y_component_iso_prob*np.sqrt(1.-rho_gau**2.)+rho_gau*x_component_iso_prob)

    comps={1: {'z': z1,
               'nonpara_dist':nonpara_dist_1
              },
           2: {'z': z2,
               'nonpara_dist':nonpara_dist_2
              }
          }
    
    for c in comps:
        z = comps[c]['z']
        nonpara_dist = comps[c]['nonpara_dist']
        
        comps[c]['comp']=__nonparametric_component(z, nonpara_dist, nb_steps)

    component_1_np = comps[1]['comp']
    component_2_np_gaussian = comps[2]['comp']
    
    return component_1_np, component_2_np_gaussian


def _nonparametric_clayton_copula(x1, x2, results, nb_steps):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Clayton copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data    
    results: Dictionay
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.        
        
    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula 
    component_2_np_gaussian: array
        Component 2 nonparametric Clayton copula   
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
    x_component_iso_prob = results['x_component_iso_prob'] 
    
    x_quantile = results['x_quantile']
    y_quantile = results['y_quantile']
    
    # Copula parameters
    nonpara_dist_1, nonpara_dist_2, nonpara_pdf_2 =_nonparametric_copula_parameters(x1, x2, nb_steps=nb_steps)
        
    # Calculate Kendall's tau    
    tau = stats.kendalltau(x2, x1)[0]
    theta_clay = (2.*tau)/(1.-tau)
    
    # Component 1 (Hs)
    z1 = stats.norm.cdf(x_component_iso_prob)
    z2_clay=((1-x_quantile**(-theta_clay)
              +x_quantile**(-theta_clay)
              /y_quantile)**(theta_clay/(1.+theta_clay)))**(-1./theta_clay)
        
    comps={1: {'z': z1,
               'nonpara_dist':nonpara_dist_1
              },
           2: {'z': z2_clay,
               'nonpara_dist':nonpara_dist_2
              }
          }
    
    for c in comps:
        z = comps[c]['z']
        nonpara_dist = comps[c]['nonpara_dist']
        comps[c]['comp']=__nonparametric_component(z, nonpara_dist, nb_steps)

    component_1_np = comps[1]['comp']
    component_2_np_clayton = comps[2]['comp']
    
    return component_1_np, component_2_np_clayton

    
def _nonparametric_gumbel_copula(x1, x2, results, nb_steps):
    '''
    This function calculates environmental contours of extreme sea 
    states using a Gumbel copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data    
    results: Dictionay
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.        
        
    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula 
    component_2_np_gumbel: array
        Component 2 nonparametric Gumbel copula  
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    assert isinstance(nb_steps, int), 'nb_steps must be of type int'
    
    Ndata=1000
   
    x_quantile = results['x_quantile']
    y_quantile = results['y_quantile']
    
    # Copula parameters
    nonpara_dist_1, nonpara_dist_2, nonpara_pdf_2 =_nonparametric_copula_parameters(x1, x2, nb_steps=nb_steps)     
        
    # Calculate Kendall's tau    
    tau = stats.kendalltau(x2, x1)[0]
    theta_gum = 1./(1.-tau)   
    
    # Component 1 (Hs)
    z1 = x_quantile       
    component_1_np=__nonparametric_component(z1, nonpara_dist_1, nb_steps)

    pts_x2 = nonpara_pdf_2[:,0]
    f_x2 = nonpara_pdf_2[:,1]
    F_x2 = nonpara_dist_2[:,1]
    
    component_2_np_gumbel = np.zeros(nb_steps)
    for k in range(nb_steps):
        z1 = np.array([x_quantile[k]]*Ndata)
        Z = np.array((z1.T, F_x2))
        Y = _gumbel_density(Z, theta_gum)
        Y = np.nan_to_num(Y) 
        # pdf 2|1
        p_x2_x1 = Y*f_x2
        # Estimate CDF from PDF
        dum = np.cumsum(p_x2_x1)
        cdf = dum/(dum[Ndata-1])
        table = np.array((pts_x2, cdf))
        table = table.T
        for j in range(Ndata):
            if y_quantile[k] <= table[0,1]:
                component_2_np_gumbel[k] = min(table[:,0])
                break
            elif y_quantile[k] <= table[j,1]:
                component_2_np_gumbel[k] = (table[j,0]+table[j-1,0])/2
                break
            else: 
                component_2_np_gumbel[k] = max(table[:,0])
    
    return component_1_np, component_2_np_gumbel
    
    
def _bivariate_KDE(x1, x2, bw, results, nb_steps, Ndata_bivariate_KDE,
                   max_x1=None, max_x2=None, log_transform=False):
    '''
    Contours generated under this class will use a non-parametric KDE to
    fit the joint distribution. This function calculates environmental
    contours of extreme sea states using a bivariate KDE to estimate 
    the joint distribution. The contour is then calculcated directly 
    from the joint distribution.
    
    Parameters
    ----------
    x1: array 
        Component 1 data
    x2: array 
        Component 2 data 
    bw: np.array
        Array containing KDE bandwidth for x1 and x2        
    results: Dictionay
        Dictionary of the iso-probability results   
    nb_steps: int
        number of points used to discritize KDE space          
    max_x1: float
        Defines the max value of x1 to discretize the KDE space
    max_x2: float
        Defines the max value of x2 to discretize the KDE space

    Returns
    -------
    x1_bivariate_KDE: array
        Calculated x1 values along the contour boundary following
        return to original input orientation.
    x2_bivariate_KDE: array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    '''
    assert isinstance(x1, np.ndarray), 'x1 must be of type np.ndarray'    
    assert isinstance(x2, np.ndarray), 'x2 must be of type np.ndarray'
    if isinstance(max_x1, type(None)):
        max_x1 = x1.max()*2
    if isinstance(max_x2, type(None)):
        max_x2 = x2.max()*2

    assert isinstance(nb_steps, int), 'nb_steps must be of type int' 
    assert isinstance(max_x1, float), 'max_x1 must be of type float'
    assert isinstance(max_x2, float), 'max_x2 must be of type float'
       
     
    p_f = results['exceedance_probability']    

    min_limit_1 = 0.01
    min_limit_2 = 0.01
    pts_x2 = np.linspace(min_limit_2, max_x2, Ndata_bivariate_KDE) 
    pts_x1 = np.linspace(min_limit_1, max_x1, Ndata_bivariate_KDE)
    pt1,pt2 = np.meshgrid(pts_x2, pts_x1)
    mesh_pts_x2 = pt1.flatten()
    mesh_pts_x1 = pt2.flatten()

    # Transform gridded points using log
    ty = [x2, x1]
    xi = [mesh_pts_x2, mesh_pts_x1]    
    txi=xi
    if log_transform:            
        ty = [np.log(x2), np.log(x1)]                  
        txi = [np.log(mesh_pts_x2), np.log(mesh_pts_x1)]

    m = len(txi[0])
    n = len(ty[0])
    d = 2
    
    # Create contour
    f = np.zeros((1,m))
    weight = np.ones((1,n))
    for i in range(0,m):
        ftemp = np.ones((n,1))
        for j in range(0,d):
            z = (txi[j][i] - ty[j])/bw[j]
            fk = stats.norm.pdf(z)
            if log_transform:     
                fnew = fk*(1/np.transpose(xi[j][i]))
            else: 
                fnew = fk
            fnew = np.reshape(fnew, (n,1))
            ftemp = np.multiply(ftemp,fnew)
        f[:,i] = np.dot(weight,ftemp)

    fhat = f.reshape(100,100)
    vals = plt.contour(pt1,pt2,fhat, levels = [p_f])
    plt.clf()
    x1_bivariate_KDE = []
    x2_bivariate_KDE = []
    
    for i,seg in enumerate(vals.allsegs[0]):
        x1_bivariate_KDE.append(seg[:,1])
        x2_bivariate_KDE.append(seg[:,0])
       
    x1_bivariate_KDE = np.transpose(np.asarray(x1_bivariate_KDE)[0])
    x2_bivariate_KDE = np.transpose(np.asarray(x2_bivariate_KDE)[0])

    return x1_bivariate_KDE, x2_bivariate_KDE
    