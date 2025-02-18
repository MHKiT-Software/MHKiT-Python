import numpy as np
import pandas as pd
import xarray as xr
import types
from scipy.stats import binned_statistic_2d as _binned_statistic_2d
from mhkit import wave
import matplotlib.pylab as plt
from os.path import join
from mhkit.utils import convert_to_dataarray
import warnings


def capture_width(P, J, to_pandas=True):
    """
    Calculates the capture width (sometimes called capture length).

    Parameters
    ------------
    P: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Power [W]
    J: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Omnidirectional wave energy flux [W/m]
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    CW: pandas Series or xarray DataArray
        Capture width [m]
    """
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    P = convert_to_dataarray(P)
    J = convert_to_dataarray(J)

    CW = P / J

    if to_pandas:
        CW = CW.to_pandas()

    return CW


def capture_length(P, J, to_pandas=True):
    """
    Alias for `capture_width`.
    """
    CW = capture_width(P, J, to_pandas)
    return CW


def statistics(X, to_pandas=True):
    """
    Calculates statistics, including count, mean, standard
    deviation (std), min, percentiles (25%, 50%, 75%), and max.

    Note that std uses a degree of freedom of N in accordance with
    Formula D.5 of IEC TS 62600-100 Ed. 2.0 en 2024.

    Parameters
    ------------
    X: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Data
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    stats: pandas Series or xarray DataArray
        Statistics
    """
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    X = convert_to_dataarray(X)

    count = X.count().item()
    mean = X.mean().item()
    std = _std_ddof0(X)
    q = X.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
    variables = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    stats = xr.DataArray(
        data=[count, mean, std, q[0], q[1], q[2], q[3], q[4]],
        dims="index",
        coords={"index": variables},
    )

    if to_pandas:
        stats = stats.to_pandas()

    return stats


def _std_ddof0(a):
    # Standard deviation with degree of freedom equal to N samples (delta degree of freedom = 0)
    if len(a) == 0:
        return np.nan
    elif len(a) == 1:
        return 0
    else:
        return np.std(a, ddof=0)


def _performance_matrix(X, Y, Z, statistic, x_centers, y_centers):
    # General performance matrix function

    # Convert bin centers to edges
    xi = [np.mean([x_centers[i], x_centers[i + 1]]) for i in range(len(x_centers) - 1)]
    xi.insert(0, np.float64(0))
    xi_end = (x_centers[-1] + np.diff(x_centers[-2:]) / 2)[0]
    xi.append(xi_end)

    yi = [np.mean([y_centers[i], y_centers[i + 1]]) for i in range(len(y_centers) - 1)]
    yi.insert(0, np.float64(0))
    yi_end = (y_centers[-1] + np.diff(y_centers[-2:]) / 2)[0]
    yi.append(yi_end)

    # Override standard deviation with degree of freedom equal to 1
    if statistic == "std":
        statistic = _std_ddof0

    # Provide function to compute frequency
    def _frequency(a):
        return len(a) / len(Z)

    if statistic == "frequency":
        statistic = _frequency

    zi, x_edge, y_edge, binnumber = _binned_statistic_2d(
        X, Y, Z, statistic, bins=[xi, yi], expand_binnumbers=False
    )

    # Warn if the X (Hm0) or Y (Te) spacing is greater than the IEC TS 62600-100 Ed. 2.0 en 2024 maxima (0.5m, 1.0s).
    dx_edge = np.diff(x_edge)
    if np.any(dx_edge > 0.5):
        warnings.warn(
            "Significant wave height bins are greater than the IEC TS 62600-100 limit of 0.5 meters."
        )
    dy_edge = np.diff(y_edge)
    if np.any(dy_edge > 1.0):
        warnings.warn(
            "Energy period bins are greater than the IEC TS 62600-100 limit of 1.0 seconds."
        )

    M = xr.DataArray(
        data=zi,
        dims=["x_centers", "y_centers"],
        coords={"x_centers": x_centers, "y_centers": y_centers},
    )

    return M


def capture_width_matrix(Hm0, Te, CW, statistic, Hm0_bins, Te_bins, to_pandas=True):
    """
    Generates a capture width matrix for a given statistic

    Note that IEC TS 62600-100 Ed. 2.0 en 2024 section 9.2.4 requires capture width matrices for
    the mean, std, count, min, and max.

    Parameters
    ------------
    Hm0: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Significant wave height from spectra [m]
    Te: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Energy period from spectra [s]
    CW : numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Capture width [m]
    statistic: string
        Statistic for each bin, options include: 'mean', 'std', 'median',
        'count', 'sum', 'min', 'max', and 'frequency'.  Note that 'std' uses
        a degree of freedom of N in accordance with Formula D.5 of IEC TS 62600-100 Ed. 2.0 en 2024.
    Hm0_bins: numpy array
        Bin centers for Hm0 [m]
    Te_bins: numpy array
        Bin centers for Te [s]
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    CWM: pandas DataFrame or xarray DataArray
         Capture width matrix with index equal to Hm0_bins and columns
         equal to Te_bins

    """
    Hm0 = convert_to_dataarray(Hm0)
    Te = convert_to_dataarray(Te)
    CW = convert_to_dataarray(CW)

    if not (isinstance(statistic, str) or callable(statistic)):
        raise TypeError(
            f"statistic must be of type str or callable. Got: {type(statistic)}"
        )
    if not isinstance(Hm0_bins, np.ndarray):
        raise TypeError(f"Hm0_bins must be of type np.ndarray. Got: {type(Hm0_bins)}")
    if not isinstance(Te_bins, np.ndarray):
        raise TypeError(f"Te_bins must be of type np.ndarray. Got: {type(Te_bins)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    CWM = _performance_matrix(Hm0, Te, CW, statistic, Hm0_bins, Te_bins)

    if to_pandas:
        CWM = CWM.to_pandas()

    return CWM


def capture_length_maxtrix(Hm0, Te, CW, statistic, Hm0_bins, Te_bins, to_pandas=True):
    """
    Alias for `capture_width_maxtrix`.
    """
    CWM = capture_width_matrix(Hm0, Te, CW, statistic, Hm0_bins, Te_bins, to_pandas)
    return CWM


def wave_energy_flux_matrix(Hm0, Te, J, statistic, Hm0_bins, Te_bins, to_pandas=True):
    """
    Generates a wave energy flux matrix for a given statistic

    Parameters
    ------------
    Hm0: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Significant wave height from spectra [m]
    Te: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Energy period from spectra [s]
    J : numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave energy flux from spectra [W/m]
    statistic: string
        Statistic for each bin, options include: 'mean', 'std', 'median',
        'count', 'sum', 'min', 'max', and 'frequency'. Note that 'std' uses
        a degree of freedom of N in accordance with Formula D.5 of IEC TS 62600-100 Ed. 2.0 en 2024.
    Hm0_bins: numpy array
        Bin centers for Hm0 [m]
    Te_bins: numpy array
        Bin centers for Te [s]
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    JM: pandas DataFrame or xarray DataArray
        Wave energy flux matrix with index equal to Hm0_bins and columns
        equal to Te_bins

    """
    Hm0 = convert_to_dataarray(Hm0)
    Te = convert_to_dataarray(Te)
    J = convert_to_dataarray(J)

    if not (isinstance(statistic, str) or callable(statistic)):
        raise TypeError(
            f"statistic must be of type str or callable. Got: {type(statistic)}"
        )
    if not isinstance(Hm0_bins, np.ndarray):
        raise TypeError(f"Hm0_bins must be of type np.ndarray. Got: {type(Hm0_bins)}")
    if not isinstance(Te_bins, np.ndarray):
        raise TypeError(f"Te_bins must be of type np.ndarray. Got: {type(Te_bins)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    JM = _performance_matrix(Hm0, Te, J, statistic, Hm0_bins, Te_bins)

    if to_pandas:
        JM = JM.to_pandas()

    return JM


def power_matrix(CWM, JM):
    """
    Generates a power matrix from a capture width matrix and wave energy
    flux matrix

    Parameters
    ------------
    CWM: pandas DataFrame, xarray DataArray, or xarray Dataset
        Capture width matrix
    JM: pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave energy flux matrix

    Returns
    ---------
    PM: pandas DataFrame, xarray DataArray, or xarray Dataset
        Power matrix

    """
    if not isinstance(CWM, (pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"CWM must be of type pd.DataFrame or xr.Dataset. Got: {type(CWM)}"
        )
    if not isinstance(JM, (pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"JM must be of type pd.DataFrame or xr.Dataset. Got: {type(JM)}"
        )

    PM = CWM * JM

    return PM


def mean_annual_energy_production_timeseries(CW, J):
    """
    Calculates mean annual energy production (MAEP) from time-series

    Parameters
    ------------
    CW: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Capture width
    J: numpy array, pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave energy flux

    Returns
    ---------
    maep: float
        Mean annual energy production

    """
    CW = convert_to_dataarray(CW)
    J = convert_to_dataarray(J)

    T = 8766  # Average length of a year (h)
    n = len(CW)

    maep = T / n * (CW * J).sum().item()

    return maep


def mean_annual_energy_production_matrix(CWM, JM, frequency):
    """
    Calculates mean annual energy production (MAEP) from matrix data
    along with data frequency in each bin

    Parameters
    ------------
    CWM: pandas DataFrame, xarray DataArray, or xarray Dataset
        Capture width
    JM: pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave energy flux
    frequency: pandas DataFrame, xarray DataArray, or xarray Dataset
        Data frequency for each bin

    Returns
    ---------
    maep: float
        Mean annual energy production

    """
    CWM = convert_to_dataarray(CWM)
    JM = convert_to_dataarray(JM)
    frequency = convert_to_dataarray(frequency)

    if not CWM.shape == JM.shape == frequency.shape:
        raise ValueError("CWM, JM, and frequency must be of the same size")
    if not np.abs(frequency.sum() - 1) < 1e-6:
        raise ValueError("Frequency components must sum to one.")

    T = 8766  # Average length of a year (h)
    maep = T * np.nansum(CWM * JM * frequency)

    return maep


def power_performance_workflow(
    S,
    h,
    P,
    statistic,
    frequency_bins=None,
    deep=False,
    rho=1205,
    g=9.80665,
    ratio=2,
    show_values=False,
    savepath="",
):
    """
    High-level function to compute power performance quantities of
    interest following IEC TS 62600-100 Ed. 2.0 en 2024 for given wave spectra.

    Parameters
    ------------
    S:  pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Spectral density [m^2/Hz] indexed by frequency [Hz]
    h: float
        Water depth [m]
    P: numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Power [W]
    statistic: string or list of strings
        Statistics for plotting capture width matrices,
        options include: "mean", "std", "median",
        "count", "sum", "min", "max", and "frequency".
        Note that "std" uses a degree of freedom of N in accordance with Formula D.5 of IEC TS 62600-100 Ed. 2.0 en 2024.
        To output capture width matrices for multiple binning parameters,
        define as a list of strings: statistic = ["", "", ""]
    frequency_bins: numpy array or pandas Series (optional)
       Bin widths for frequency of S. Required for unevenly sized bins
    deep: bool (optional)
        If True use the deep water approximation. Default False. When
        False a depth check is run to check for shallow water. The ratio
        of the shallow water regime can be changed using the ratio
        keyword.
    rho: float (optional)
        Water density [kg/m^3]. Default = 1025 kg/m^3
    g: float (optional)
        Gravitational acceleration [m/s^2]. Default = 9.80665 m/s^2
    ratio: float or int (optional)
        Only applied if depth=False. If h/l > ratio,
        water depth will be set to deep. Default ratio = 2.
    show_values : bool (optional)
        Show values on the scatter diagram. Default = False.
    savepath: string (optional)
        Path to save figure. Terminate with '\'. Default="".

    Returns
    ---------
    CWM: xarray dataset
        Capture width matrices

    maep_matrix: float
        Mean annual energy production
    """
    S = convert_to_dataarray(S)
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    P = convert_to_dataarray(P)
    if not isinstance(deep, bool):
        raise TypeError(f"deep must be of type bool. Got: {type(deep)}")
    if not isinstance(rho, (int, float)):
        raise TypeError(f"rho must be of type int or float. Got: {type(rho)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"ratio must be of type int or float. Got: {type(ratio)}")

    # Compute the enegy periods from the spectra data
    Te = wave.resource.energy_period(S, frequency_bins=frequency_bins, to_pandas=False)

    # Compute the significant wave height from the NDBC spectra data
    Hm0 = wave.resource.significant_wave_height(
        S, frequency_bins=frequency_bins, to_pandas=False
    )

    # Compute the energy flux from spectra data and water depth
    J = wave.resource.energy_flux(
        S, h, deep=deep, rho=rho, g=g, ratio=ratio, to_pandas=False
    )

    # Calculate capture width from power and energy flux
    CW = wave.performance.capture_width(P, J, to_pandas=False)

    # Generate bins for Hm0 and Te, input format (start, stop, step_size)
    Hm0_bins = np.arange(0, Hm0.values.max() + 0.5, 0.5)
    Te_bins = np.arange(0, Te.values.max() + 1, 1)

    # Create capture width matrices for each statistic based on IEC TS 62600-100 Ed. 2.0 en 2024
    # Median, sum, frequency additionally provided
    CWM = xr.Dataset()
    CWM["mean"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "mean", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["std"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "std", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["median"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "median", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["count"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "count", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["sum"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "sum", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["min"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "min", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["max"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "max", Hm0_bins, Te_bins, to_pandas=False
    )
    CWM["freq"] = wave.performance.capture_width_matrix(
        Hm0, Te, CW, "frequency", Hm0_bins, Te_bins, to_pandas=False
    )

    # Create wave energy flux matrix using mean
    JM = wave.performance.wave_energy_flux_matrix(
        Hm0, Te, J, "mean", Hm0_bins, Te_bins, to_pandas=False
    )

    # Calculate maep from matrix
    maep_matrix = wave.performance.mean_annual_energy_production_matrix(
        CWM["mean"], JM, CWM["freq"]
    )

    # Plot capture width matrices using statistic
    for str in statistic:
        if str not in list(CWM.data_vars):
            print("ERROR: Invalid Statistics passed")
            continue
        plt.figure(figsize=(12, 12), num="Capture Width Matrix " + str)
        ax = plt.gca()
        wave.graphics.plot_matrix(
            CWM[str],
            xlabel="Te (s)",
            ylabel="Hm0 (m)",
            zlabel=str + " of Capture Width",
            show_values=show_values,
            ax=ax,
        )
        plt.savefig(join(savepath, "Capture Width Matrix " + str + ".png"))

    return CWM, maep_matrix
