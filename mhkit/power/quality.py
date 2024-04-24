"""
This module contains functions for calculating various aspects of power quality, 
particularly focusing on the analysis of harmonics and interharmonics in electrical 
power systems. These functions are designed to assist in power quality assessments 
by providing tools to analyze voltage and current signals for their harmonic 
and interharmonic components based on the guidelines and methodologies 
outlined in IEC 61000-4-7.

Functions in this module include:

- harmonics: Calculates the harmonics from time series of voltage or current. 
  This function returns the amplitude of the time-series data harmonics indexed by 
  the harmonic frequency, aiding in the identification of harmonic distortions 
  within the power system.

- harmonic_subgroups: Computes the harmonic subgroups as per IEC 61000-4-7 standards. 
  Harmonic subgroups provide insights into the distribution of power across 
  different harmonic frequencies, which is crucial for understanding the behavior 
  of non-linear loads and their impact on the power quality.

- total_harmonic_current_distortion (THCD): Determines the total harmonic current 
  distortion, offering a summary metric that quantifies the overall level of 
  harmonic distortion present in the current waveform. This metric is essential 
  for assessing compliance with power quality standards and guidelines.

- interharmonics: Identifies and calculates the interharmonics present in the 
  power system. Interharmonics, which are frequencies that occur between the 
  fundamental and harmonic frequencies, can arise from various sources and 
  potentially lead to power quality issues.
"""

from typing import Union
import pandas as pd
import numpy as np
from scipy import fftpack
import xarray as xr
from mhkit.utils import convert_to_dataset


def harmonics(
    signal_data: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    freq: Union[float, int],
    grid_freq: int,
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates the harmonics from time series of voltage or current based on IEC 61000-4-7.

    Parameters
    -----------
    signal_data: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Time-series of voltage [V] or current [A]

    freq: float or Int
        Frequency of the time-series data [Hz]

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    harmonic_amplitudes: pandas DataFrame or xarray Dataset
        Amplitude of the time-series data harmonics indexed by the harmonic
        frequency with signal name columns
    """
    if not isinstance(signal_data, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "signal_data must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(signal_data)}"
        )

    if not isinstance(freq, (float, int)):
        raise TypeError(f"freq must be of type float or integer. Got {type(freq)}")

    if grid_freq not in [50, 60]:
        raise ValueError(f"grid_freq must be either 50 or 60. Got {grid_freq}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got {type(to_pandas)}")

    # Convert input to xr.Dataset
    signal_data = convert_to_dataset(signal_data, "data")

    sample_spacing = 1.0 / freq

    # Loop through all variables in signal_data
    harmonic_amplitudes = xr.Dataset()
    for var in signal_data.data_vars:
        dataarray = signal_data[var]
        dataarray = dataarray.to_numpy()

        frequency_bin_centers = fftpack.fftfreq(len(dataarray), d=sample_spacing)
        harmonics_amplitude = np.abs(np.fft.fft(dataarray, axis=0))

        harmonic_amplitudes = harmonic_amplitudes.assign(
            {var: (["frequency"], harmonics_amplitude)}
        )
        harmonic_amplitudes = harmonic_amplitudes.assign_coords(
            {"frequency": frequency_bin_centers}
        )
    harmonic_amplitudes = harmonic_amplitudes.sortby("frequency")

    if grid_freq == 60:
        hertz = np.arange(0, 3060, 5)
    elif grid_freq == 50:
        hertz = np.arange(0, 2570, 5)

    harmonic_amplitudes = harmonic_amplitudes.reindex(
        {"frequency": hertz}, method="nearest"
    )
    harmonic_amplitudes = (
        harmonic_amplitudes / len(signal_data[list(signal_data.dims)[0]]) * 2
    )

    if to_pandas:
        harmonic_amplitudes = harmonic_amplitudes.to_pandas()

    return harmonic_amplitudes


def harmonic_subgroups(
    harmonic_amplitudes: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    grid_freq: int,
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonic_amplitudes: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    subgroup_results: pandas DataFrame or xarray Dataset
        Harmonic subgroups indexed by harmonic frequency
        with signal name columns
    """
    if not isinstance(
        harmonic_amplitudes, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "harmonic_amplitudes must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(harmonic_amplitudes)}"
        )

    if grid_freq not in [50, 60]:
        raise ValueError(f"grid_freq must be either 50 or 60. Got {grid_freq}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if not isinstance(frequency_dimension, str):
        raise TypeError(
            f"frequency_dimension must be of type str. Got: {type(frequency_dimension)}"
        )

    # Convert input to xr.Dataset
    harmonic_amplitudes = convert_to_dataset(harmonic_amplitudes, "harmonic_amplitudes")

    if (
        frequency_dimension != ""
        and frequency_dimension not in harmonic_amplitudes.coords
    ):
        raise ValueError(
            "frequency_dimension was supplied but is not a dimension "
            + f"of harmonic_amplitudes. Got {frequency_dimension}"
        )

    if grid_freq == 60:
        hertz = np.arange(0, 3060, 60)
    else:
        hertz = np.arange(0, 2550, 50)

    # Sort input data index
    if frequency_dimension == "":
        frequency_dimension = list(harmonic_amplitudes.dims)[0]
    harmonic_amplitudes = harmonic_amplitudes.sortby(frequency_dimension)

    # Loop through all variables in harmonics
    subgroup_results = xr.Dataset()
    for var in harmonic_amplitudes.data_vars:
        dataarray = harmonic_amplitudes[var]
        subgroup = np.zeros(np.size(hertz))

        for ihz in np.arange(0, len(hertz)):
            current_frequency = hertz[ihz]
            ind = dataarray.indexes[frequency_dimension].get_loc(current_frequency)

            data_subset = dataarray.isel({frequency_dimension: [ind - 1, ind, ind + 1]})
            subgroup[ihz] = (data_subset**2).sum() ** 0.5

        subgroup_results = subgroup_results.assign({var: (["frequency"], subgroup)})
        subgroup_results = subgroup_results.assign_coords({"frequency": hertz})

    if to_pandas:
        subgroup_results = subgroup_results.to_pandas()

    return subgroup_results


def total_harmonic_current_distortion(
    harmonics_subgroup: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates the total harmonic current distortion (THC) based on IEC/TS 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Subgrouped current harmonics indexed by harmonic frequency

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    thcd_result: pd.DataFrame or xarray Dataset
        Total harmonic current distortion indexed by signal name with THCD column
    """
    if not isinstance(
        harmonics_subgroup, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "harmonics_subgroup must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(harmonics_subgroup)}"
        )

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if not isinstance(frequency_dimension, str):
        raise TypeError(
            f"frequency_dimension must be of type bool. Got: {type(frequency_dimension)}"
        )

    # Convert input to xr.Dataset
    harmonics_subgroup = convert_to_dataset(harmonics_subgroup, "harmonics_subgroup")

    if (
        frequency_dimension != ""
        and frequency_dimension not in harmonics_subgroup.coords
    ):
        raise ValueError(
            "frequency_dimension was supplied but is not a dimension "
            + f"of harmonics. Got {frequency_dimension}"
        )

    if frequency_dimension == "":
        frequency_dimension = list(harmonics_subgroup.dims)[0]
    harmonics_sq = harmonics_subgroup.isel({frequency_dimension: slice(2, 50)}) ** 2
    harmonics_sum = harmonics_sq.sum()

    thcd_result = (
        np.sqrt(harmonics_sum) / harmonics_subgroup.isel({frequency_dimension: 1})
    ) * 100

    if isinstance(thcd_result, xr.DataArray):
        thcd_result.name = ["THCD"]

    if to_pandas:
        thcd_result = thcd_result.to_pandas()

    return thcd_result


def interharmonics(
    harmonic_amplitudes: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    grid_freq: int,
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates the interharmonics from the harmonic_amplitudes of current

    Parameters
    -----------
    harmonic_amplitudes: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Harmonic amplitude indexed by the harmonic frequency

    grid_freq: int
        Value indicating if the power supply is 50 or 60 Hz. Options = 50 or 60

    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    -------
    interharmonic_groups: pandas DataFrame or xarray Dataset
        Interharmonics groups
    """
    if not isinstance(
        harmonic_amplitudes, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "harmonic_amplitudes must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(harmonic_amplitudes)}"
        )

    if grid_freq not in [50, 60]:
        raise ValueError(f"grid_freq must be either 50 or 60. Got {grid_freq}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Convert input to xr.Dataset
    harmonic_amplitudes = convert_to_dataset(harmonic_amplitudes, "harmonic_amplitudes")

    if (
        frequency_dimension != ""
        and frequency_dimension not in harmonic_amplitudes.coords
    ):
        raise ValueError(
            "frequency_dimension was supplied but is not a dimension "
            + f"of harmonic_amplitudes. Got {frequency_dimension}"
        )

    if grid_freq == 60:
        hertz = np.arange(0, 3060, 60)
    elif grid_freq == 50:
        hertz = np.arange(0, 2550, 50)

    # Sort input data index
    if frequency_dimension == "":
        frequency_dimension = list(harmonic_amplitudes.dims)[0]
    harmonic_amplitudes = harmonic_amplitudes.sortby(frequency_dimension)

    # Loop through all variables in harmonic_amplitudes
    interharmonic_groups = xr.Dataset()
    for var in harmonic_amplitudes.data_vars:
        dataarray = harmonic_amplitudes[var]
        subset = np.zeros(np.size(hertz))

        for ihz in np.arange(0, len(hertz)):
            current_frequency = hertz[ihz]
            ind = dataarray.indexes[frequency_dimension].get_loc(current_frequency)

            if grid_freq == 60:
                data = dataarray.isel({frequency_dimension: slice(ind + 1, ind + 11)})
                subset[ihz] = (data**2).sum() ** 0.5
            else:
                data = dataarray.isel({frequency_dimension: slice(ind + 1, ind + 7)})
                subset[ihz] = (data**2).sum() ** 0.5

        interharmonic_groups = interharmonic_groups.assign(
            {var: (["frequency"], subset)}
        )
        interharmonic_groups = interharmonic_groups.assign_coords({"frequency": hertz})

    if to_pandas:
        interharmonic_groups = interharmonic_groups.to_pandas()

    return interharmonic_groups
