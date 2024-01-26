import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import hilbert

def instantaneous_frequency(um, time_dimension="", to_pandas=True):

    """
    Calculates instantaneous frequency of measured voltage

    Parameters
    -----------
    um: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Measured voltage (V) indexed by time

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
    if not isinstance(um, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('um must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(um)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')
    if not isinstance(time_dimension, str):
        raise TypeError(
            f'time_dimension must be of type bool. Got: {type(time_dimension)}')

    # Convert input to xr.Dataset
    um = _convert_to_dataset(um, 'data')
    
    if time_dimension != '' and time_dimension not in um.coords:
        raise ValueError('time_dimension was supplied but is not a dimension '
                         + f'of um. Got {time_dimension}')

    # Get the dimension of interest
    if time_dimension == "":
        time_dimension = list(um.coords)[0]

    # Calculate time step
    if isinstance(um.coords[time_dimension].values[0], np.datetime64):
        t = (um[time_dimension] - np.datetime64('1970-01-01 00:00:00'))/np.timedelta64(1, 's')
    else:
        t = um[time_dimension]
    dt = np.diff(t)

    # Calculate frequency
    frequency = xr.Dataset()
    for var in um.data_vars:
        f = hilbert(um[var])
        instantaneous_phase = np.unwrap(np.angle(f))
        instantaneous_frequency = np.diff(instantaneous_phase)/(2.0*np.pi) * (1/dt)

        frequency = frequency.assign({var: (time_dimension, instantaneous_frequency)})
        frequency = frequency.assign_coords({time_dimension: um.coords[time_dimension].values[0:-1]})

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
                        f'xr.DataArray, or xr.Dataset. Got {type(voltage)}')
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('current must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(current)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    # Convert inputs to xr.Dataset
    voltage = _convert_to_dataset(voltage, 'voltage')
    current = _convert_to_dataset(current, 'current')

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
                        f'xr.DataArray, or xr.Dataset. Got {type(voltage)}')
    if not isinstance(current, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
        raise TypeError('current must be of type pd.Series, pd.DataFrame, ' + 
                        f'xr.DataArray, or xr.Dataset. Got {type(current)}')
    if not isinstance(line_to_line, bool):
        raise TypeError(
            f'line_to_line must be of type bool. Got: {type(line_to_line)}')
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f'to_pandas must be of type bool. Got: {type(to_pandas)}')

    # Convert inputs to xr.Dataset
    voltage = _convert_to_dataset(voltage, 'voltage')
    current = _convert_to_dataset(current, 'current')

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

def _convert_to_dataset(data, name='data'):
    """
    Converts the given data to an xarray.Dataset.
    
    This function is designed to handle inputs that can be either a pandas DataFrame, a pandas Series,
    an xarray DataArray, or an xarray Dataset. It ensures that the output is consistently an xarray.Dataset.
    
    Parameters
    ----------
    data: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        The data to be converted. 
    
    name: str (Optional)
        The name to assign to the data variable in case the input is an xarray DataArray without a name.
        Default value is 'data'.
    
    Returns
    -------
    xarray.Dataset
        The input data converted to an xarray.Dataset. If the input is already an xarray.Dataset,
        it is returned as is.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> ds = _convert_to_dataset(df)
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>
    
    >>> series = pd.Series([1, 2, 3], name='C')
    >>> ds = _convert_to_dataset(series)
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>
    
    >>> data_array = xr.DataArray([1, 2, 3])
    >>> ds = _convert_to_dataset(data_array, name='D')
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>
    """
    if not isinstance(data, (pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset)):
        raise TypeError("Input data must be of type pandas.DataFrame, pandas.Series, "
                    "xarray.DataArray, or xarray.Dataset")

    if not isinstance(name, str):
        raise TypeError("The 'name' parameter must be a string")     

    # Takes data that could be pd.DataFrame, pd.Series, xr.DataArray, or 
    # xr.Dataset and converts it to xr.Dataset
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_xarray()

    if isinstance(data, xr.DataArray):
        if data.name is None:
            data.name = name # xr.DataArray.to_dataset() breaks if the data variable is unnamed
        data = data.to_dataset()

    return data
