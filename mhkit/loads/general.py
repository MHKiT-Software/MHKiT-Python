"""
This module provides tools for analyzing and processing data signals
related to turbine blade performance and fatigue analysis. It implements
methodologies based on standards such as IEC TS 62600-3:2020 ED1,
incorporating statistical binning, moment calculations, and fatigue 
damage estimation using the rainflow counting algorithm. Key
functionalities include:

    - `bin_statistics`: Bins time-series data against a specified signal,
      such as wind speed, to calculate mean and standard deviation statistics
      for each bin, following IEC TS 62600-3:2020 ED1 guidelines. It supports
      output in both pandas DataFrame and xarray Dataset formats.

    - `blade_moments`: Calculates the flapwise and edgewise moments of turbine 
      blades using derived calibration coefficients and raw strain signals. 
      This function is crucial for understanding the loading and performance
      characteristics of turbine blades.

    - `damage_equivalent_load`: Estimates the damage equivalent load (DEL)
      of a single data signal using a 4-point rainflow counting algorithm.
      This method is vital for assessing fatigue life and durability of
      materials under variable amplitude loading.

References:
- C. Amzallag et. al., International Journal of Fatigue, 16 (1994) 287-293.
- ISO 12110-2, Metallic materials - Fatigue testing - Variable amplitude fatigue testing.
- G. Marsh et. al., International Journal of Fatigue, 82 (2016) 757-765.
"""

from typing import Union, List, Tuple, Optional
from scipy.stats import binned_statistic
import pandas as pd
import xarray as xr
import numpy as np
import fatpack
from mhkit.utils.type_handling import to_numeric_array


def bin_statistics(
    data: Union[pd.DataFrame, xr.Dataset],
    bin_against: np.ndarray,
    bin_edges: np.ndarray,
    data_signal: Optional[List[str]] = None,
    to_pandas: bool = True,
) -> Tuple[Union[pd.DataFrame, xr.Dataset], Union[pd.DataFrame, xr.Dataset]]:
    """
    Bins calculated statistics against data signal (or channel)
    according to IEC TS 62600-3:2020 ED1.

    Parameters
    -----------
    data : pandas DataFrame or xarray Dataset
       Time-series statistics of data signal(s)
    bin_against : array
        Data signal to bin data against (e.g. wind speed)
    bin_edges : array
        Bin edges with consistent step size
    data_signal : list, optional
        List of data signal(s) to bin, default = all data signals
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    --------
    bin_mean : pandas DataFrame or xarray Dataset
        Mean of each bin
    bin_std : pandas DataFrame or xarray Dataset
        Standard deviation of each bim
    """

    if not isinstance(data, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"data must be of type pd.DataFrame or xr.Dataset. Got: {type(data)}"
        )

    # Use _to_numeric_array to process bin_against and bin_edges
    bin_against = to_numeric_array(bin_against, "bin_against")
    bin_edges = to_numeric_array(bin_edges, "bin_edges")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # If input is pandas, convert to xarray
    if isinstance(data, pd.DataFrame):
        data = data.to_xarray()

    if data_signal is None:
        data_signal = []
    # Determine variables to analyze
    if len(data_signal) == 0:  # if not specified, bin all variables
        data_signal = list(data.keys())
    else:
        if not isinstance(data_signal, list):
            raise TypeError(
                f"data_signal must be of type list. Got: {type(data_signal)}"
            )

    # Pre-allocate variable dictionaries
    bin_stat_list = {}
    bin_std_list = {}

    # loop through data_signal and get binned means
    for signal_name in data_signal:
        # Bin data
        bin_stat_mean = binned_statistic(
            bin_against, data[signal_name], statistic="mean", bins=bin_edges
        )
        bin_stat_std = binned_statistic(
            bin_against, data[signal_name], statistic="std", bins=bin_edges
        )

        bin_stat_list[signal_name] = ("index", bin_stat_mean.statistic)
        bin_std_list[signal_name] = ("index", bin_stat_std.statistic)

    # Convert to Datasets
    bin_mean = xr.Dataset(
        data_vars=bin_stat_list,
        coords={"index": np.arange(0, len(bin_stat_mean.statistic))},
    )
    bin_std = xr.Dataset(
        data_vars=bin_std_list,
        coords={"index": np.arange(0, len(bin_stat_std.statistic))},
    )

    # Check for nans
    for variable in list(bin_mean.variables):
        if bin_mean[variable].isnull().any():
            print("Warning: bins for some variables may be empty!")
            break

    if to_pandas:
        bin_mean = bin_mean.to_pandas()
        bin_std = bin_std.to_pandas()

    return bin_mean, bin_std


def blade_moments(
    blade_coefficients: np.ndarray,
    flap_offset: float,
    flap_raw: np.ndarray,
    edge_offset: float,
    edge_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer function for deriving blade flap and edge moments using blade matrix.

    Parameters
    -----------
    blade_coefficients : numpy array
        Derived blade calibration coefficients listed in order of D1, D2, D3, D4
    flap_offset : float
        Derived offset of raw flap signal obtained during calibration process
    flap_raw : numpy array
        Raw strain signal of blade in the flapwise direction
    edge_offset : float
        Derived offset of raw edge signal obtained during calibration process
    edge_raw : numpy array
        Raw strain signal of blade in the edgewise direction

    Returns
    --------
    M_flap : numpy array
        Blade flapwise moment in SI units
    M_edge : numpy array
        Blade edgewise moment in SI units
    """

    # Convert and validate blade_coefficients, flap_raw, and edge_raw
    blade_coefficients = to_numeric_array(blade_coefficients, "blade_coefficients")
    flap_raw = to_numeric_array(flap_raw, "flap_raw")
    edge_raw = to_numeric_array(edge_raw, "edge_raw")

    if not isinstance(flap_offset, (float, int)):
        raise TypeError(
            f"flap_offset must be of type int or float. Got: {type(flap_offset)}"
        )
    if not isinstance(edge_offset, (float, int)):
        raise TypeError(
            f"edge_offset must be of type int or float. Got: {type(edge_offset)}"
        )

    # remove offset from raw signal
    flap_signal = flap_raw - flap_offset
    edge_signal = edge_raw - edge_offset

    # apply matrix to get load signals
    m_flap = blade_coefficients[0] * flap_signal + blade_coefficients[1] * edge_signal
    m_edge = blade_coefficients[2] * flap_signal + blade_coefficients[3] * edge_signal

    return m_flap, m_edge


def damage_equivalent_load(
    data_signal: np.ndarray,
    m: Union[float, int],
    bin_num: int = 100,
    data_length: Union[float, int] = 600,
) -> float:
    """
    Calculates the damage equivalent load of a single data signal (or channel)
    based on IEC TS 62600-3:2020 ED1. 4-point rainflow counting algorithm from
    fatpack module is based on the following resources:

    - `C. Amzallag et. al. Standardization of the rainflow counting method for
      fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293`
    - `ISO 12110-2, Metallic materials - Fatigue testing - Variable amplitude
      fatigue testing.`
    - `G. Marsh et. al. Review and application of Rainflow residue processing
      techniques for accurate fatigue damage estimation. International Journal
      of Fatigue, 82 (2016) 757-765`


    Parameters:
    -----------
    data_signal : array
        Data signal being analyzed
    m : float/int
        Fatigue slope factor of material
    bin_num : int
        Number of bins for rainflow counting method (minimum=100)
    data_length : float/int
        Length of measured data (seconds)

    Returns
    --------
    DEL : float
        Damage equivalent load (DEL) of single data signal
    """

    to_numeric_array(data_signal, "data_signal")
    if not isinstance(m, (float, int)):
        raise TypeError(f"m must be of type float or int. Got: {type(m)}")
    if not isinstance(bin_num, (float, int)):
        raise TypeError(f"bin_num must be of type float or int. Got: {type(bin_num)}")
    if not isinstance(data_length, (float, int)):
        raise TypeError(
            f"data_length must be of type float or int. Got: {type(data_length)}"
        )

    rainflow_ranges = fatpack.find_rainflow_ranges(data_signal, k=256)

    # Range count and bin
    n_rf, s_rf = fatpack.find_range_count(rainflow_ranges, bin_num)

    del_s = s_rf**m * n_rf / data_length
    del_value = del_s.sum() ** (1 / m)

    return del_value
