"""
This module contains functions for calculating electrical power metrics from 
measured voltage and current data. It supports both direct current (DC) and 
alternating current (AC) calculations, including instantaneous frequency 
analysis for AC signals and power calculations for three-phase AC systems. 
The calculations can accommodate both line-to-neutral and line-to-line voltage 
measurements and offer flexibility in output formats, allowing results to be 
saved as either pandas DataFrames or xarray Datasets.

Functions:
    instantaneous_frequency: Calculates the instantaneous frequency of a measured
    voltage signal over time.
    
    dc_power: Computes the DC power from voltage and current measurements, providing
    both individual channel outputs and a gross power calculation.
    
    ac_power_three_phase: Calculates the magnitude of active AC power for three-phase
    systems, considering the power factor and voltage measurement configuration 
    (line-to-neutral or line-to-line).
"""

from typing import Union
import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import hilbert
from mhkit.utils import convert_to_dataset


def instantaneous_frequency(
    measured_voltage: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    time_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates instantaneous frequency of measured voltage

    Parameters
    -----------
    measured_voltage: pandas Series, pandas DataFrame, xarray DataArray,
        or xarray Dataset Measured voltage (V) indexed by time

    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    ---------
    frequency: pandas DataFrame or xarray Dataset
        Frequency of the measured voltage (Hz) indexed by time
        with signal name columns
    """
    if not isinstance(
        measured_voltage, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "measured_voltage must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(measured_voltage)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")
    if not isinstance(time_dimension, str):
        raise TypeError(
            f"time_dimension must be of type bool. Got: {type(time_dimension)}"
        )

    # Convert input to xr.Dataset
    measured_voltage = convert_to_dataset(measured_voltage, "data")

    if time_dimension != "" and time_dimension not in measured_voltage.coords:
        raise ValueError(
            "time_dimension was supplied but is not a dimension "
            + f"of measured_voltage. Got {time_dimension}"
        )

    # Get the dimension of interest
    if time_dimension == "":
        time_dimension = list(measured_voltage.coords)[0]

    # Calculate time step
    if isinstance(measured_voltage.coords[time_dimension].values[0], np.datetime64):
        time = (
            measured_voltage[time_dimension] - np.datetime64("1970-01-01 00:00:00")
        ) / np.timedelta64(1, "s")
    else:
        time = measured_voltage[time_dimension]
    d_t = np.diff(time)

    # Calculate frequency
    frequency = xr.Dataset()
    for var in measured_voltage.data_vars:
        freq = hilbert(measured_voltage[var])
        instantaneous_phase = np.unwrap(np.angle(freq))
        f_instantaneous = np.diff(instantaneous_phase) / (2.0 * np.pi) * (1 / d_t)

        frequency = frequency.assign({var: (time_dimension, f_instantaneous)})
        frequency = frequency.assign_coords(
            {time_dimension: measured_voltage.coords[time_dimension].values[0:-1]}
        )

    if to_pandas:
        frequency = frequency.to_pandas()

    return frequency


def dc_power(
    voltage: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    current: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates DC power from voltage and current

    Parameters
    -----------
    voltage: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured DC voltage [V] indexed by time

    current: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured three phase current [A] indexed by time

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    power_dc: pandas DataFrame or xarray Dataset
        DC power [W] from each channel and gross power indexed by time
    """
    if not isinstance(voltage, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "voltage must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(voltage)}"
        )
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "current must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(current)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Convert inputs to xr.Dataset
    voltage = convert_to_dataset(voltage, "voltage")
    current = convert_to_dataset(current, "current")

    # Check that sizes are the same
    if not (
        voltage.sizes == current.sizes
        and len(voltage.data_vars) == len(current.data_vars)
    ):
        raise ValueError("current and voltage must have the same shape")

    power_dc = xr.Dataset()
    gross = None

    # Multiply current and voltage variables together, in order they're assigned
    for i, (current_var, voltage_var) in enumerate(
        zip(current.data_vars, voltage.data_vars)
    ):
        temp = current[current_var] * voltage[voltage_var]
        power_dc = power_dc.assign({f"{i}": temp})
        if gross is None:
            gross = temp
        else:
            gross = gross + temp

    power_dc = power_dc.assign({"Gross": gross})

    if to_pandas:
        power_dc = power_dc.to_dataframe()

    return power_dc


def ac_power_three_phase(
    voltage: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    current: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    power_factor: float,
    line_to_line: bool = False,
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates magnitude of active AC power from line to neutral voltage and current

    Parameters
    -----------
    voltage: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured DC voltage [V] indexed by time

    current: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured three phase current [A] indexed by time

    power_factor: float
        Power factor for the efficiency of the system

    line_to_line: bool (Optional)
        Set to true if the given voltage measurements are line_to_line

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    --------
    power_ac: pandas DataFrame or xarray Dataset
        Magnitude of active AC power [W] indexed by time with Power column
    """
    if not isinstance(voltage, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "voltage must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(voltage)}"
        )
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "current must be of type pd.Series, pd.DataFrame, "
            + f"xr.DataArray, or xr.Dataset. Got {type(current)}"
        )
    if not isinstance(line_to_line, bool):
        raise TypeError(f"line_to_line must be of type bool. Got: {type(line_to_line)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Convert inputs to xr.Dataset
    voltage = convert_to_dataset(voltage, "voltage")
    current = convert_to_dataset(current, "current")

    # Check that sizes are the same
    if len(voltage.data_vars) != 3:
        raise ValueError("voltage must have three columns")
    if len(current.data_vars) != 3:
        raise ValueError("current must have three columns")
    if current.sizes != voltage.sizes:
        raise ValueError("current and voltage must be of the same size")

    power = dc_power(voltage, current, to_pandas=False)["Gross"]
    power.name = "Power"
    power = (
        power.to_dataset()
    )  # force xr.DataArray to be consistently in xr.Dataset format
    power_ac = np.abs(power) * power_factor

    if line_to_line:
        power_ac = power_ac * np.sqrt(3)

    if to_pandas:
        power_ac = power_ac.to_pandas()

    return power_ac
