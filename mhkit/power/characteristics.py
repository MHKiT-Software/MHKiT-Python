import pandas as pd
import numpy as np
from scipy.signal import hilbert
import datetime

def instantaneous_frequency(um):

    """
    Calculates instantaneous frequency of measured voltage
    
     
    Parameters
    -----------
    um: pandas Series or DataFrame
        Measured voltage (V) indexed by time 

        
    Returns
    ---------
    frequency: pandas DataFrame
        Frequency of the measured voltage (Hz) indexed by time  
        with signal name columns
    """  
    assert isinstance(um, (pd.Series, pd.DataFrame)), 'um must be of type pd.Series or pd.DataFrame'
   
    if isinstance(um.index[0], datetime.datetime):
        t = (um.index - datetime.datetime(1970,1,1)).total_seconds()
    else:
        t = um.index

    dt = pd.Series(t).diff()[1:]

    if isinstance(um,pd.Series):
        um = um.to_frame()

    columns  = um.columns
    frequency=pd.DataFrame(columns=columns)
    for column in um.columns:
        f = hilbert(um[column])
        instantaneous_phase = np.unwrap(np.angle(f))
        instantaneous_frequency = np.diff(instantaneous_phase) /(2.0*np.pi) * (1/dt)
        frequency[column] = instantaneous_frequency
        
    return frequency

def dc_power(voltage, current):
    """
    Calculates DC power from voltage and current

    Parameters
    -----------
    voltage: pandas Series or DataFrame
        Measured DC voltage [V] indexed by time
    current: pandas Series or DataFrame
        Measured three phase current [A] indexed by time
    
    Returns
    --------
    P: pandas DataFrame
        DC power [W] from each channel and gross power indexed by time
    """
    assert isinstance(voltage, (pd.Series, pd.DataFrame)), 'voltage must be of type pd.Series or pd.DataFrame'
    assert isinstance(current, (pd.Series, pd.DataFrame)), 'current must be of type pd.Series or pd.DataFrame'
    assert voltage.shape == current.shape, 'current and volatge must have the same shape'
    
    
    P = current.values * voltage.values
    P = pd.DataFrame(P) 
    P['Gross'] = P.sum(axis=1, skipna=True) 

    return P

def ac_power_three_phase(voltage, current, power_factor, line_to_line=False):
    """
    Calculates magnitude of active AC power from line to neutral voltage and current 

    Parameters
    -----------
    voltage: pandas DataFrame
        Time-series of three phase measured voltage [V] indexed by time
    current: pandas DataFrame 
        Time-series of three phase measured current [A] indexed by time
    power_factor: float 
        Power factor for the efficiency of the system
    line_to_line: bool
        Set to true if the given voltage measurements are line_to_line
    
    Returns
    --------
    P: pandas DataFrame
        Magnitude of active AC power [W] indexed by time with Power column 
    """
    assert isinstance(voltage, pd.DataFrame), 'voltage must be of type pd.DataFrame'
    assert isinstance(current, pd.DataFrame), 'current must be of type pd.DataFrame'
    assert len(voltage.columns) == 3, 'voltage must have three columns'
    assert len(current.columns) == 3, 'current must have three columns'
    assert current.shape == voltage.shape, 'current and voltage must be of the same size'
    

    abs_current = np.abs(current.values)
    abs_voltage = np.abs(voltage.values)

    if line_to_line:
        power = abs_current * (abs_voltage * np.sqrt(3))
    else:
        power = abs_current * abs_voltage
         
    power = pd.DataFrame(power) 
    P = power.sum(axis=1) * power_factor
    P = P.to_frame('Power')
    
    return P
