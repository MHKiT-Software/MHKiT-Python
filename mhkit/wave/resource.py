from scipy.optimize import fsolve as _fsolve
from scipy import signal as _signal
import pandas as pd
import numpy as np
from scipy import stats

### Spectrum
def elevation_spectrum(eta, sample_rate, nnft, window='hann',
    detrend=True, noverlap=None):
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
        Specifies if a linear trend is removed from the data before
        calculating the wave energy spectrum.  Data is detrended by default.
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


def pierson_moskowitz_spectrum(f, Tp, Hs):
    """
    Calculates Pierson-Moskowitz Spectrum from IEC TS 62600-2 ED2 Annex C.2 (2019)

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
        Spectral density [m^2/Hz] indexed frequency [Hz]

    """
    try:
        f = np.array(f)
    except:
        pass
    assert isinstance(f, np.ndarray), 'f must be of type np.ndarray'
    assert isinstance(Tp, (int,float)), 'Tp must be of type int or float'
    assert isinstance(Hs, (int,float)), 'Hs must be of type int or float'

    f.sort()
    B_PM = (5/4)*(1/Tp)**4
    A_PM = B_PM*(Hs/2)**2

    # Avoid a divide by zero if the 0 frequency is provided
    # The zero frequency should always have 0 amplitude, otherwise
    # we end up with a mean offset when computing the surface elevation.
    Sf = np.zeros(f.size)
    if f[0] == 0.0:
        inds = range(1, f.size)
    else:
        inds = range(0, f.size)
    
    Sf[inds]  = A_PM*f[inds]**(-5)*np.exp(-B_PM*f[inds]**(-4))

    col_name = 'Pierson-Moskowitz ('+str(Tp)+'s)'
    S = pd.DataFrame(Sf, index=f, columns=[col_name])

    return S


def jonswap_spectrum(f, Tp, Hs, gamma=None):
    """
    Calculates JONSWAP Spectrum from IEC TS 62600-2 ED2 Annex C.2 (2019)

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
    assert isinstance(gamma, (int,float, type(None))), \
        'gamma must be of type int or float'

    f.sort()
    B_PM = (5/4)*(1/Tp)**4
    A_PM = B_PM*(Hs/2)**2

    # Avoid a divide by zero if the 0 frequency is provided
    # The zero frequency should always have 0 amplitude, otherwise
    # we end up with a mean offset when computing the surface elevation.
    S_f = np.zeros(f.size)
    if f[0] == 0.0:
        inds = range(1, f.size)
    else:
        inds = range(0, f.size)

    S_f[inds]  = A_PM*f[inds]**(-5)*np.exp(-B_PM*f[inds]**(-4))

    if not gamma:
        TpsqrtHs = Tp/np.sqrt(Hs);
        if TpsqrtHs <= 3.6:
            gamma = 5;
        elif TpsqrtHs > 5:
            gamma = 1;
        else:
            gamma = np.exp(5.75 - 1.15*TpsqrtHs);

    # Cutoff frequencies for gamma function
    siga = 0.07
    sigb = 0.09

    fp = 1/Tp # peak frequency
    lind = np.where(f<=fp)
    hind = np.where(f>fp)
    Gf = np.zeros(f.shape)
    Gf[lind] = gamma**np.exp(-(f[lind]-fp)**2/(2*siga**2*fp**2))
    Gf[hind] = gamma**np.exp(-(f[hind]-fp)**2/(2*sigb**2*fp**2))
    C = 1- 0.287*np.log(gamma)
    Sf = C*S_f*Gf

    col_name = 'JONSWAP ('+str(Hs)+'m,'+str(Tp)+'s)'
    S = pd.DataFrame(Sf, index=f, columns=[col_name])

    return S

### Metrics
def surface_elevation(S, time_index, seed=None, frequency_bins=None, phases=None, method='ifft'):
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
    frequency_bins: numpy array or pandas DataFrame (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    phases: numpy array or pandas DataFrame (optional)
        Explicit phases for frequency components (overrides seed)
        for example, phases = np.random.rand(len(S)) * 2 * np.pi
    method: str (optional)
        Method used to calculate the surface elevation. 'ifft'
        (Inverse Fast Fourier Transform) used by default if the
        given frequency_bins==None.
        'sum_of_sines' explicitly sums each frequency component
        and used by default if frequency_bins are provided.
        The 'ifft' method is significantly faster.

    Returns
    ---------
    eta: pandas DataFrame
        Wave surface elevation [m] indexed by time [s]

    """
    time_index = np.array(time_index)
    assert isinstance(S, pd.DataFrame), 'S must be of type pd.DataFrame'
    assert isinstance(time_index, np.ndarray), ('time_index must be of type'
            'np.ndarray')
    assert isinstance(seed, (type(None),int)), 'seed must be of type int'
    assert isinstance(frequency_bins, (type(None), np.ndarray, pd.DataFrame)),(
            "frequency_bins must be of type None, np.ndarray, or pd,DataFrame")
    assert isinstance(phases, (type(None), np.ndarray, pd.DataFrame)), (
            'phases must be of type None, np.ndarray, or pd,DataFrame')
    assert isinstance(method, str)

    if frequency_bins is not None:
        assert frequency_bins.squeeze().shape == (S.squeeze().shape[0],),(
            'shape of frequency_bins must match shape of S')
    if phases is not None:
        assert phases.squeeze().shape == S.squeeze().shape,(
            'shape of phases must match shape of S')
        
    if method is not None:
        assert method == 'ifft' or method == 'sum_of_sines',(
            f"unknown method {method}, options are 'ifft' or 'sum_of_sines'")
        
    if method == 'ifft':
        assert S.index.values[0] == 0, ('ifft method must have zero frequency defined')        

    f = pd.Series(S.index)
    f.index = f
    if frequency_bins is None:
        delta_f = f.values[1]-f.values[0]
        assert np.allclose(f.diff()[1:], delta_f)
    elif isinstance(frequency_bins, np.ndarray):
        delta_f = pd.Series(frequency_bins, index=S.index)
        method = 'sum_of_sines'
    elif isinstance(frequency_bins, pd.DataFrame):
        assert len(frequency_bins.columns) == 1, ('frequency_bins must only'
                'contain 1 column')
        delta_f = frequency_bins.squeeze()
        method = 'sum_of_sines'

    if phases is None:
        np.random.seed(seed)
        phase = pd.DataFrame(2*np.pi*np.random.rand(S.shape[0], S.shape[1]),
                             index=S.index, columns=S.columns)
    elif isinstance(phases, np.ndarray):
        phase = pd.DataFrame(phases, index=S.index, columns=S.columns)
    elif isinstance(phases, pd.DataFrame):
        phase = phases

    omega = pd.Series(2*np.pi*f)
    omega.index = f

    # Wave amplitude times delta f
    A = 2*S
    A = A.multiply(delta_f, axis=0)
    A = np.sqrt(A)

    if method == 'ifft':
        A_cmplx = A * (np.cos(phase) + 1j*np.sin(phase))

        def func(v):
            eta = np.fft.irfft(0.5 * v.values.squeeze() * time_index.size, time_index.size)
            return pd.Series(data=eta, index=time_index)
        
        eta = A_cmplx.apply(func)

    elif method == 'sum_of_sines':
        # Product of omega and time
        B = np.outer(time_index, omega)
        B = B.reshape((len(time_index), len(omega)))
        B = pd.DataFrame(B, index=time_index, columns=omega.index)

        # wave elevation
        eta = pd.DataFrame(columns=S.columns, index=time_index)
        for mcol in eta.columns:
            C = np.cos(B+phase[mcol])
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


def energy_period_to_peak_period(Te, gamma):
    """
    Convert from spectral energy period (Te) to peak period (Tp) using ITTC approximation for JONSWAP Spectrum.

    Approximation is given in "The Specialist Committee on Waves, Final Report
    and Recommendations to the 23rd ITTC", Proceedings of the 23rd ITTC - Volume
    2, Table A4.

    Parameters:
    ----------
    Te: float or array
        Spectral energy period [s]
    gamma: float or int
        Peak enhancement factor for JONSWAP spectrum

    Returns
    -------
    Tp: float or array
        Spectral peak period [s]
    """
    assert isinstance(Te, (float, np.ndarray)), 'Te must be a float or a ndarray'
    assert isinstance(gamma, (float, int)), 'gamma must be of type float or int'

    factor = 0.8255 + 0.03852*gamma - 0.005537*gamma**2 + 0.0003154*gamma**3

    return Te / factor


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
