from scipy.optimize import fsolve as _fsolve
from scipy import signal as _signal
import pandas as pd
import xarray as xr
import numpy as np
from mhkit.utils import to_numeric_array, convert_to_dataarray, convert_to_dataset


### Spectrum
def elevation_spectrum(
    eta,
    sample_rate,
    nnft,
    window="hann",
    detrend=True,
    noverlap=None,
    time_dimension="",
    to_pandas=True,
):
    """
    Calculates the wave energy spectrum from wave elevation time-series

    Parameters
    ------------
    eta: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
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
    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    S: pandas DataFrame or xr.Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    """

    # TODO: Add confidence intervals, equal energy frequency spacing, and NDBC
    #       frequency spacing
    # TODO: may need to raise an error for the length of nnft- signal.welch breaks when nfft is too short
    eta = convert_to_dataset(eta)
    if not isinstance(sample_rate, (float, int)):
        raise TypeError(
            f"sample_rate must be of type int or float. Got: {type(sample_rate)}"
        )
    if not isinstance(nnft, int):
        raise TypeError(f"nnft must be of type int. Got: {type(nnft)}")
    if not isinstance(window, str):
        raise TypeError(f"window must be of type str. Got: {type(window)}")
    if not isinstance(detrend, bool):
        raise TypeError(f"detrend must be of type bool. Got: {type(detrend)}")
    if not nnft > 0:
        raise ValueError(f"nnft must be > 0. Got: {nnft}")
    if not sample_rate > 0:
        raise ValueError(f"sample_rate must be > 0. Got: {sample_rate}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if time_dimension == "":
        time_dimension = list(eta.dims)[0]
    else:
        if time_dimension not in list(eta.dims):
            raise ValueError(
                f"time_dimension is not a dimension of eta ({list(eta.dims)}). Got: {time_dimension}."
            )
    time = eta[time_dimension]
    delta_t = time.values[1] - time.values[0]
    if not np.allclose(time.diff(dim=time_dimension)[1:], delta_t):
        raise ValueError(
            "Time bins are not evenly spaced. Create a constant "
            + "temporal spacing for eta."
        )

    S = xr.Dataset()
    for var in eta.data_vars:
        data = eta[var]
        if detrend:
            data = _signal.detrend(
                data.dropna(dim=time_dimension), axis=-1, type="linear", bp=0
            )
        [f, wave_spec_measured] = _signal.welch(
            data,
            fs=sample_rate,
            window=window,
            nperseg=nnft,
            nfft=nnft,
            noverlap=noverlap,
        )
        S[var] = (["Frequency"], wave_spec_measured)
    S = S.assign_coords({"Frequency": f})

    if to_pandas:
        S = S.to_dataframe()

    return S


def pierson_moskowitz_spectrum(f, Tp, Hs, to_pandas=True):
    """
    Calculates Pierson-Moskowitz Spectrum from IEC TS 62600-2 ED2 Annex C.2 (2019)

    Parameters
    ------------
    f: list, np.ndarray, pd.Series, xr.DataArray
        Frequency [Hz]
    Tp: float/int
        Peak period [s]
    Hs: float/int
        Significant wave height [m]
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    S: xarray Dataset
        Spectral density [m^2/Hz] indexed frequency [Hz]

    """
    f = to_numeric_array(f, "f")
    if not isinstance(Tp, (int, float)):
        raise TypeError(f"Tp must be of type int or float. Got: {type(Tp)}")
    if not isinstance(Hs, (int, float)):
        raise TypeError(f"Hs must be of type int or float. Got: {type(Hs)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    f.sort()
    B_PM = (5 / 4) * (1 / Tp) ** 4
    A_PM = B_PM * (Hs / 2) ** 2

    # Avoid a divide by zero if the 0 frequency is provided
    # The zero frequency should always have 0 amplitude, otherwise
    # we end up with a mean offset when computing the surface elevation.
    Sf = np.zeros(f.size)
    if f[0] == 0.0:
        inds = range(1, f.size)
    else:
        inds = range(0, f.size)

    Sf[inds] = A_PM * f[inds] ** (-5) * np.exp(-B_PM * f[inds] ** (-4))

    name = "Pierson-Moskowitz (" + str(Tp) + "s)"
    S = xr.Dataset(data_vars={name: (["Frequency"], Sf)}, coords={"Frequency": f})

    if to_pandas:
        S = S.to_pandas()

    return S


def jonswap_spectrum(f, Tp, Hs, gamma=None, to_pandas=True):
    """
    Calculates JONSWAP Spectrum from IEC TS 62600-2 ED2 Annex C.2 (2019)

    Parameters
    ------------
    f: list, np.ndarray, pd.Series, xr.DataArray
        Frequency [Hz]
    Tp: float/int
        Peak period [s]
    Hs: float/int
        Significant wave height [m]
    gamma: float (optional)
        Gamma
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    S: pandas Series or xarray DataArray
        Spectral density [m^2/Hz] indexed frequency [Hz]
    """
    f = to_numeric_array(f, "f")
    if not isinstance(Tp, (int, float)):
        raise TypeError(f"Tp must be of type int or float. Got: {type(Tp)}")
    if not isinstance(Hs, (int, float)):
        raise TypeError(f"Hs must be of type int or float. Got: {type(Hs)}")
    if not isinstance(gamma, (int, float, type(None))):
        raise TypeError(
            f"If specified, gamma must be of type int or float. Got: {type(gamma)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    f.sort()
    B_PM = (5 / 4) * (1 / Tp) ** 4
    A_PM = B_PM * (Hs / 2) ** 2

    # Avoid a divide by zero if the 0 frequency is provided
    # The zero frequency should always have 0 amplitude, otherwise
    # we end up with a mean offset when computing the surface elevation.
    S_f = np.zeros(f.size)
    if f[0] == 0.0:
        inds = range(1, f.size)
    else:
        inds = range(0, f.size)

    S_f[inds] = A_PM * f[inds] ** (-5) * np.exp(-B_PM * f[inds] ** (-4))

    if not gamma:
        TpsqrtHs = Tp / np.sqrt(Hs)
        if TpsqrtHs <= 3.6:
            gamma = 5
        elif TpsqrtHs > 5:
            gamma = 1
        else:
            gamma = np.exp(5.75 - 1.15 * TpsqrtHs)

    # Cutoff frequencies for gamma function
    siga = 0.07
    sigb = 0.09

    fp = 1 / Tp  # peak frequency
    lind = np.where(f <= fp)
    hind = np.where(f > fp)
    Gf = np.zeros(f.shape)
    Gf[lind] = gamma ** np.exp(-((f[lind] - fp) ** 2) / (2 * siga**2 * fp**2))
    Gf[hind] = gamma ** np.exp(-((f[hind] - fp) ** 2) / (2 * sigb**2 * fp**2))
    C = 1 - 0.287 * np.log(gamma)
    Sf = C * S_f * Gf

    name = "JONSWAP (" + str(Hs) + "m," + str(Tp) + "s)"
    S = xr.Dataset(data_vars={name: (["Frequency"], Sf)}, coords={"Frequency": f})

    if to_pandas:
        S = S.to_pandas()

    return S


### Metrics
def surface_elevation(
    S,
    time_index,
    seed=None,
    frequency_bins=None,
    phases=None,
    method="ifft",
    frequency_dimension="",
    to_pandas=True,
):
    """
    Calculates wave elevation time-series from spectrum

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    time_index: numpy array
        Time used to create the wave elevation time-series [s],
        for example, time = np.arange(0,100,0.01)
    seed: int (optional)
        Random seed
    frequency_bins: numpy array, pandas Series, or xarray DataArray (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    phases: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Explicit phases for frequency components (overrides seed)
        for example, phases = np.random.rand(len(S)) * 2 * np.pi
    method: str (optional)
        Method used to calculate the surface elevation. 'ifft'
        (Inverse Fast Fourier Transform) used by default if the
        given frequency_bins==None.
        'sum_of_sines' explicitly sums each frequency component
        and used by default if frequency_bins are provided.
        The 'ifft' method is significantly faster.
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    eta: pandas DataFrame or xarray Dataset
        Wave surface elevation [m] indexed by time [s]

    """
    time_index = to_numeric_array(time_index, "time_index")
    S = convert_to_dataset(S)
    if not isinstance(seed, (type(None), int)):
        raise TypeError(f"If specified, seed must be of type int. Got: {type(seed)}")
    if not isinstance(phases, type(None)):
        phases = convert_to_dataset(phases)
    if not isinstance(method, str):
        raise TypeError(f"method must be of type str. Got: {type(method)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )
    f = S[frequency_dimension]

    if not isinstance(frequency_bins, (type(None), np.ndarray)):
        frequency_bins = convert_to_dataarray(frequency_bins)
    elif isinstance(frequency_bins, np.ndarray):
        frequency_bins = xr.DataArray(
            data=frequency_bins,
            dims=frequency_dimension,
            coords={frequency_dimension: f},
        )
    if frequency_bins is not None:
        if not frequency_bins.squeeze().shape == f.shape:
            raise ValueError(
                "shape of frequency_bins must match shape of the frequency dimension of S"
            )
    if phases is not None:
        if not list(phases.data_vars) == list(S.data_vars):
            raise ValueError("phases must have the same variable names as S")
        for var in phases.data_vars:
            if not phases[var].shape == S[var].shape:
                raise ValueError(
                    "shape of variables in phases must match shape of variables in S"
                )
    if method is not None:
        if not (method == "ifft" or method == "sum_of_sines"):
            raise ValueError(f"Method must be 'ifft' or 'sum_of_sines'. Got: {method}")

    if method == "ifft":
        if not f[0] == 0:
            raise ValueError(
                f"ifft method must have zero frequency defined. Lowest frequency is: {S.index.values[0]}"
            )

    if frequency_bins is None:
        delta_f = f.values[1] - f.values[0]
        if not np.allclose(f.diff(dim=frequency_dimension)[1:], delta_f):
            raise ValueError(
                "Frequency bins are not evenly spaced. "
                + "Define 'frequency_bins' or create a constant "
                + "frequency spacing for S."
            )
    else:
        if not len(frequency_bins.squeeze().shape) == 1:
            raise ValueError("frequency_bins must only contain 1 column")
        delta_f = frequency_bins
        method = "sum_of_sines"

    omega = xr.DataArray(
        data=2 * np.pi * f, dims=frequency_dimension, coords={frequency_dimension: f}
    )

    eta = xr.Dataset()
    for var in S.data_vars:
        if phases is None:
            np.random.seed(seed)
            phase = xr.DataArray(
                data=2 * np.pi * np.random.rand(S[var].size),
                dims="Frequency",
                coords={"Frequency": f},
            )
        else:
            phase = phases[var]

        # Wave amplitude times delta f
        A = 2 * S[var]
        A = A * delta_f
        A = np.sqrt(A)

        if method == "ifft":
            A_cmplx = A * (np.cos(phase) + 1j * np.sin(phase))
            eta_tmp = np.fft.irfft(
                0.5 * A_cmplx.values * time_index.size, time_index.size
            )
            eta[var] = xr.DataArray(
                data=eta_tmp, dims="Time", coords={"Time": time_index}
            )

        elif method == "sum_of_sines":
            # Product of omega and time
            B = np.outer(time_index, omega)
            B = B.reshape((len(time_index), len(omega)))
            B = xr.DataArray(
                data=B,
                dims=["Time", "Frequency"],
                coords={"Time": time_index, "Frequency": f},
            )

            # wave elevation
            # eta = xr.DataArray(columns=S.columns, index=time_index)
            # for mcol in eta.columns:
            C = np.cos(B + phase)
            # C = xr.DataArray(data=C, index=time_index, columns=omega.index)
            eta[var] = (C * A).sum(axis=1)

    if to_pandas:
        eta = eta.to_dataframe()

    return eta


def frequency_moment(S, N, frequency_bins=None, frequency_dimension="", to_pandas=True):
    """
    Calculates the Nth frequency moment of the spectrum

    Parameters
    -----------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    N: int
        Moment (0 for 0th, 1 for 1st ....)
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    m: pandas DataFrame or xarray Dataset
        Nth Frequency Moment indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(N, int):
        raise TypeError(f"N must be of type int. Got: {type(N)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )
    f = S[frequency_dimension]

    # Eq 8 in IEC 62600-101
    S = S.sel({frequency_dimension: slice(1e-12, f.max())})  # omit frequency of 0
    f = S[frequency_dimension]  # reset frequency_dimension without the 0 frequency

    fn = np.power(f, N)
    if frequency_bins is None:
        delta_f = f.diff(dim=frequency_dimension)
        delta_f0 = f[1] - f[0]
        delta_f0 = delta_f0.assign_coords({frequency_dimension: f[0]})
        delta_f = xr.concat([delta_f0, delta_f], dim=frequency_dimension)
    else:
        delta_f = xr.DataArray(
            data=convert_to_dataarray(frequency_bins),
            dims=frequency_dimension,
            coords={frequency_dimension: f},
        )

    m = S * fn * delta_f
    m = m.sum(dim=frequency_dimension)

    m = _transform_dataset(m, "m" + str(N))

    if to_pandas:
        m = m.to_dataframe()

    return m


def significant_wave_height(S, frequency_bins=None, to_pandas=True):
    """
    Calculates wave height from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Hm0: pandas DataFrame or xarray Dataset
        Significant wave height [m] index by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Eq 12 in IEC 62600-101
    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "Hm0"}
    )
    Hm0 = 4 * np.sqrt(m0)

    if to_pandas:
        Hm0 = Hm0.to_dataframe()

    return Hm0


def average_zero_crossing_period(S, frequency_bins=None, to_pandas=True):
    """
    Calculates wave average zero crossing period from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Tz: pandas DataFrame or xarray Dataset
        Average zero crossing period [s] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Eq 15 in IEC 62600-101
    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "Tz"}
    )
    m2 = frequency_moment(S, 2, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m2": "Tz"}
    )

    Tz = np.sqrt(m0 / m2)

    if to_pandas:
        Tz = Tz.to_dataframe()

    return Tz


def average_crest_period(S, frequency_bins=None, to_pandas=True):
    """
    Calculates wave average crest period from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Tavg: pandas DataFrame or xarray Dataset
        Average wave period [s] indexed by S.columns

    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    m2 = frequency_moment(S, 2, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m2": "Tavg"}
    )
    m4 = frequency_moment(S, 4, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m4": "Tavg"}
    )

    Tavg = np.sqrt(m2 / m4)

    if to_pandas:
        Tavg = Tavg.to_dataframe()

    return Tavg


def average_wave_period(S, frequency_bins=None, to_pandas=True):
    """
    Calculates mean wave period from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Tm: pandas DataFrame or xarray Dataset
        Mean wave period [s] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "Tm"}
    )
    m1 = frequency_moment(S, 1, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m1": "Tm"}
    )

    Tm = np.sqrt(m0 / m1)

    if to_pandas:
        Tm = Tm.to_dataframe()

    return Tm


def peak_period(S, frequency_dimension="", to_pandas=True):
    """
    Calculates wave peak period from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Tp: pandas DataFrame or xarray Dataset
        Wave peak period [s] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )

    # Eq 14 in IEC 62600-101
    fp = S.idxmax(dim=frequency_dimension)  # Hz
    Tp = 1 / fp

    Tp = _transform_dataset(Tp, "Tp")

    if to_pandas:
        Tp = Tp.to_dataframe()

    return Tp


def energy_period(S, frequency_bins=None, to_pandas=True):
    """
    Calculates wave energy period from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    Te: pandas DataFrame or xarray Dataset
        Wave energy period [s] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    mn1 = frequency_moment(
        S, -1, frequency_bins=frequency_bins, to_pandas=False
    ).rename({"m-1": "Te"})
    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "Te"}
    )

    # Eq 13 in IEC 62600-101
    Te = mn1 / m0

    if to_pandas:
        Te = Te.to_dataframe()

    return Te


def spectral_bandwidth(S, frequency_bins=None, to_pandas=True):
    """
    Calculates bandwidth from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    e: pandas DataFrame or xarray Dataset
        Spectral bandwidth [s] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    m2 = frequency_moment(S, 2, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m2": "e"}
    )
    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "e"}
    )
    m4 = frequency_moment(S, 4, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m4": "e"}
    )

    e = np.sqrt(1 - (m2**2) / (m0 / m4))

    if to_pandas:
        e = e.to_dataframe()

    return e


def spectral_width(S, frequency_bins=None, to_pandas=True):
    """
    Calculates wave spectral width from spectra

    Parameters
    ------------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    frequency_bins: numpy array or pandas Series (optional)
        Bin widths for frequency of S. Required for unevenly sized bins
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    v: pandas DataFrame or xarray Dataset
        Spectral width [m] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    mn2 = frequency_moment(
        S, -2, frequency_bins=frequency_bins, to_pandas=False
    ).rename({"m-2": "v"})
    m0 = frequency_moment(S, 0, frequency_bins=frequency_bins, to_pandas=False).rename(
        {"m0": "v"}
    )
    mn1 = frequency_moment(
        S, -1, frequency_bins=frequency_bins, to_pandas=False
    ).rename({"m-1": "v"})

    # Eq 16 in IEC 62600-101
    v = np.sqrt((m0 * mn2 / np.power(mn1, 2)) - 1)

    if to_pandas:
        v = v.to_dataframe()

    return v


def energy_flux(
    S,
    h,
    deep=False,
    rho=1025,
    g=9.80665,
    ratio=2,
    frequency_dimension="",
    to_pandas=True,
):
    """
    Calculates the omnidirectional wave energy flux of the spectra

    Parameters
    -----------
    S: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
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
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    J: pandas DataFrame or xarray Dataset
        Omni-directional wave energy flux [W/m] indexed by S.columns
    """
    S = convert_to_dataset(S)
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(deep, bool):
        raise TypeError(f"deep must be of type bool. Got: {type(deep)}")
    if not isinstance(rho, (int, float)):
        raise TypeError(f"rho must be of type int or float. Got: {type(rho)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"ratio must be of type int or float. Got: {type(ratio)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )
    f = S[frequency_dimension]

    if deep:
        # Eq 8 in IEC 62600-100, deep water simplification
        Te = energy_period(S, to_pandas=False).rename({"Te": "J"})
        Hm0 = significant_wave_height(S, to_pandas=False).rename({"Hm0": "J"})

        coeff = rho * (g**2) / (64 * np.pi)

        J = coeff * (Hm0**2) * Te

    else:
        # deep water flag is false
        k = wave_number(f, h, rho, g, to_pandas=False)

        # wave celerity (group velocity)
        Cg = wave_celerity(k, h, g, depth_check=True, ratio=ratio, to_pandas=False)[
            "Cg"
        ]

        # Calculating the wave energy flux, Eq 9 in IEC 62600-101
        delta_f = f.diff(dim=frequency_dimension)
        delta_f0 = f[1] - f[0]
        delta_f0 = delta_f0.assign_coords({frequency_dimension: f[0]})
        delta_f = xr.concat([delta_f0, delta_f], dim=frequency_dimension)

        CgSdelF = S * delta_f * Cg

        J = rho * g * CgSdelF.sum(dim=frequency_dimension)
        J = _transform_dataset(J, "J")

    if to_pandas:
        J = J.to_dataframe()

    return J


def energy_period_to_peak_period(Te, gamma):
    """
    Convert from spectral energy period (Te) to peak period (Tp) using ITTC approximation for JONSWAP Spectrum.

    Approximation is given in "The Specialist Committee on Waves, Final Report
    and Recommendations to the 23rd ITTC", Proceedings of the 23rd ITTC - Volume
    2, Table A4.

    Parameters
    ----------
    Te: int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset
    gamma: float or int
        Peak enhancement factor for JONSWAP spectrum

    Returns
    -------
    Tp: float or array
        Spectral peak period [s]
    """
    if not isinstance(
        Te, (int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            f"Te must be an int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset. Got: {type(Te)}"
        )
    if not isinstance(gamma, (float, int)):
        raise TypeError(f"gamma must be of type float or int. Got: {type(gamma)}")

    factor = 0.8255 + 0.03852 * gamma - 0.005537 * gamma**2 + 0.0003154 * gamma**3

    Tp = Te / factor
    if isinstance(Tp, xr.Dataset):
        Tp.rename({"Te": "Tp"})

    return Tp


def wave_celerity(
    k, h, g=9.80665, depth_check=False, ratio=2, frequency_dimension="", to_pandas=True
):
    """
    Calculates wave celerity (group velocity)

    Parameters
    ----------
    k: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
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
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    Cg: pandas DataFrame or xarray Dataset
        Water celerity [m/s] indexed by frequency [Hz]
    """
    k = convert_to_dataarray(k)
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")
    if not isinstance(depth_check, bool):
        raise TypeError(f"depth_check must be of type bool. Got: {type(depth_check)}")
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"ratio must be of type int or float. Got: {type(ratio)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(k.coords)[0]
    elif frequency_dimension not in list(k.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of k ({list(k.dims)}). Got: {frequency_dimension}."
        )
    f = k[frequency_dimension]
    k = k.values

    if depth_check:
        l = wave_length(k)

        # get depth regime
        dr = depth_regime(l, h, ratio=ratio)

        # deep frequencies
        df = f[dr]
        dk = k[dr]

        # deep water approximation
        dCg = np.pi * df / dk
        dCg = xr.DataArray(
            data=dCg, dims=frequency_dimension, coords={frequency_dimension: df}
        )
        dCg.name = "Cg"

        # shallow frequencies
        sf = f[~dr]
        sk = k[~dr]
        sCg = (np.pi * sf / sk) * (1 + (2 * h * sk) / np.sinh(2 * h * sk))
        sCg = xr.DataArray(
            data=sCg, dims=frequency_dimension, coords={frequency_dimension: sf}
        )
        sCg.name = "Cg"

        Cg = xr.concat([dCg, sCg], dim=frequency_dimension).sortby(frequency_dimension)
        Cg.name = "Cg"

    else:
        # Eq 10 in IEC 62600-101
        Cg = (np.pi * f / k) * (1 + (2 * h * k) / np.sinh(2 * h * k))
        Cg = xr.DataArray(
            data=Cg, dims=frequency_dimension, coords={frequency_dimension: f}
        )
        Cg.name = "Cg"

    Cg = Cg.to_dataset()

    if to_pandas:
        Cg = Cg.to_dataframe()

    return Cg


def wave_length(k):
    """
    Calculates wave length from wave number
    To compute: 2*pi/wavenumber

    Parameters
    -------------
    k: int, float, numpy ndarray, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave number [1/m] indexed by frequency

    Returns
    ---------
    l: int, float, numpy ndarray, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave length [m] indexed by frequency. Output type is identical to the type of k.
    """
    if not isinstance(
        k, (int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            f"k must be an int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray or xr.Dataset. Got: {type(k)}"
        )

    l = 2 * np.pi / k

    return l


def wave_number(f, h, rho=1025, g=9.80665, to_pandas=True):
    """
    Calculates wave number

    To compute wave number from angular frequency (w), convert w to f before
    using this function (f = w/2*pi)

    Parameters
    -----------
    f: int, float, numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Frequency [Hz]
    h: float
        Water depth [m]
    rho: float (optional)
        Water density [kg/m^3]
    g: float (optional)
        Gravitational acceleration [m/s^2]
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    k: pandas DataFrame or xarray Dataset
        Wave number [1/m] indexed by frequency [Hz]
    """
    if isinstance(f, (int, float)):
        f = np.asarray([f])
    f = convert_to_dataarray(f)
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(rho, (int, float)):
        raise TypeError(f"rho must be of type int or float. Got: {type(rho)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    w = 2 * np.pi * f  # angular frequency
    xi = w / np.sqrt(g / h)  # note: =h*wa/sqrt(h*g/h)
    yi = xi * xi / np.power(1.0 - np.exp(-np.power(xi, 2.4908)), 0.4015)
    k0 = yi / h  # Initial guess without current-wave interaction

    # Eq 11 in IEC 62600-101 using initial guess from Guo (2002)
    def func(kk):
        val = np.power(w, 2) - g * kk * np.tanh(kk * h)
        return val

    mask = np.abs(func(k0)) > 1e-9
    if mask.sum() > 0:
        k0_mask = k0[mask]
        w = w[mask]

        k, info, ier, mesg = _fsolve(func, k0_mask, full_output=True)
        if not ier == 1:
            raise ValueError("Wave number not found. " + mesg)
        k0[mask] = k

    k0.name = "k"
    k = k0.to_dataset()

    if to_pandas:
        k = k.to_dataframe()

    return k


def depth_regime(l, h, ratio=2):
    """
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
    l: int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset
        wavelength [m]
    h: float or int
        water column depth [m]
    ratio: float or int (optional)
        if h/l > ratio, water depth will be set to deep. Default ratio = 2

    Returns
    -------
    depth_reg: boolean or boolean array-like
        Boolean True if deep water, False otherwise
    """
    if not isinstance(
        l, (int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            f"l must be of type int, float, np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, or xr.Dataset. Got: {type(l)}"
        )
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")

    depth_reg = h / l > ratio

    return depth_reg


def _transform_dataset(data, name):
    # Converting data from a Dataset into a DataArray will turn the variables
    # columns into a 'variable' dimension.
    # Converting it back to a dataset will keep this concise variable dimension
    # but in the expected xr.Dataset/pd.DataFrame format
    data = data.to_array()
    data = convert_to_dataset(data, name=name)
    data = data.rename({"variable": "index"})
    return data
