import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import hilbert
import datetime

def instantaneous_frequency(um, dimension="", to_pandas=True):

    """
    Calculates instantaneous frequency of measured voltage

    Parameters
    -----------
    um: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured voltage (V) indexed by time 

    dimension: string (optional)
        Name of the xarray dimension corresponding to time.
        If not supplied, time is assumed to be the first dimension.

    to_pandas: bool (Optional)
        Flag to save output to pandas instead of xarray. Default = True.

    Returns
    ---------
    frequency: pandas DataFrame or xarray Dataset
        Frequency of the measured voltage (Hz) indexed by time  
        with signal name columns
    """  
    if not isinstance(um, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('um must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got{type(um)}')

    # Convert data into xr.Dataset
    if isinstance(um, (pd.DataFrame, pd.Series)):
        um = um.to_xarray()
    if um.name is None:
        um.name = 'data'
    if isinstance(um, xr.DataArray):
        um = um.to_dataset()

    # Get the dimension of interest
    if dimension == "":
        dimension = list(um.coords)[0]

    # Calculate time step
    if isinstance(um.coords[dimension].values[0], datetime.datetime):
        t = (um.index - datetime.datetime(1970,1,1)).total_seconds()
    else:
        t = um.index
    dt = np.diff(t)

    # Calculate frequency
    frequency = xr.Dataset()
    for var in um.data_vars:
        f = hilbert(um[var])
        instantaneous_phase = np.unwrap(np.angle(f))
        instantaneous_frequency = np.diff(instantaneous_phase) /(2.0*np.pi) * (1/dt)

        frequency = frequency.assign({var: (dimension, instantaneous_frequency)})
        frequency = frequency.assign_coords({dimension: um.coords[dimension].values[0:-1]})

    if to_pandas:
        frequency = frequency.to_pandas()

    return frequency

def dc_power(voltage, current, to_pandas=True):
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
    P: pandas DataFrame or xarray Dataset
        DC power [W] from each channel and gross power indexed by time
    """
    if not isinstance(voltage, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('voltage must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got{type(voltage)}')
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('current must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got{type(current)}')

    # Convert input to xarray.Dataset
    if isinstance(voltage, (pd.DataFrame, pd.Series)):
        voltage = voltage.to_xarray()
    if isinstance(current, (pd.DataFrame, pd.Series)):
        current = current.to_xarray()
    if isinstance(voltage, xr.DataArray):
        voltage = voltage.to_dataset()
    if isinstance(current, xr.DataArray):
        current = current.to_dataset()

    # Check that sizes are the same
    if not (voltage.sizes == current.sizes and len(voltage.data_vars) == len(current.data_vars)):
        raise ValueError('current and voltage must have the same shape')

    P = xr.Dataset()
    gross = None
    
    # Multiply current and voltage variables together, in order they're assigned
    for i, (current_var, voltage_var) in enumerate(zip(current.data_vars,voltage.data_vars)):
        temp = current[current_var]*voltage[voltage_var]
        P = P.assign({f'{i}': temp})
        if gross is None:
            gross = temp
        else:
            gross = gross + temp

    P = P.assign({'Gross': gross})

    if to_pandas:
        P = P.to_dataframe()

    return P

def ac_power_three_phase(voltage, current, power_factor, line_to_line=False, to_pandas=True):
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
    P: pandas DataFrame or xarray Dataset
        Magnitude of active AC power [W] indexed by time with Power column 
    """
    if not isinstance(voltage, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('voltage must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got{type(voltage)}')
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('current must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got{type(current)}')

    # Convert input to xarray.Dataset
    if isinstance(voltage, (pd.DataFrame, pd.Series)):
        voltage = voltage.to_xarray()
    if isinstance(current, (pd.DataFrame, pd.Series)):
        current = current.to_xarray()
    if isinstance(voltage, xr.DataArray):
        voltage = voltage.to_dataset()
    if isinstance(current, xr.DataArray):
        current = current.to_dataset()

    # Check that sizes are the same
    if not len(voltage.data_vars) == 3:
        raise ValueError('voltage must have three columns')
    if not len(current.data_vars) == 3:
        raise ValueError('current must have three columns')
    if not current.sizes == voltage.sizes:
        raise ValueError('current and voltage must be of the same size')

    power = dc_power(voltage, current, to_pandas=False)['Gross']
    power.name = 'Power'
    power = power.to_dataset() # force xr.DataArray to be consistently in xr.Dataset format
    P = np.abs(power) * power_factor

    if line_to_line:
        P = P * np.sqrt(3)

    if to_pandas:
        P = P.to_pandas()

    return P
