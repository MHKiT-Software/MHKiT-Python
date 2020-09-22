import pandas as pd 
import numpy as np 

def tip_speed_ratio(rotor_speed,rotor_diameter,inflow_speed):
    '''
    Function used to calculate the tip speed ratio (TSR) of a MEC device with rotor

    Parameters:
    -----------
    rotor_speed : numpy array
        Rotor speed [revolutions per second]
    rotor_diameter : float/int
        Diameter of rotor [m]
    inflow_speed : numpy array
        Velocity of inflow condition [m/s]

    Returns:
    --------
    TSR : numpy array
        Calculated tip speed ratio (TSR)
    '''
    
    try: rotor_speed = np.asarray(rotor_speed)
    except: 'rotor_speed must be of type np.ndarray'        
    try: inflow_speed = np.asarray(inflow_speed)
    except: 'inflow_speed must be of type np.ndarray'
    
    assert isinstance(rotor_diameter, (float,int)), 'rotor diameter must be of type int or float'


    rotor_velocity = rotor_speed * np.pi*rotor_diameter

    TSR = rotor_velocity / inflow_speed

    return TSR

def power_coefficient(power,inflow_speed,capture_area,rho):
    '''
    Function that calculates the power coefficient of MEC device

    Parameters:
    -----------
    power : numpy array
        Power output signal of device after losses [W]
    inflow_speed : numpy array
        Speed of inflow [m/s]
    capture_area : float/int
        Projected area of rotor normal to inflow [m^2]
    rho : float/int
        Density of environment [kg/m^3]

    Returns:
    --------
    Cp : numpy array
        Power coefficient of device [-]
    '''
    
    try: power = np.asarray(power)
    except: 'power must be of type np.ndarray'
    try: inflow_speed = np.asarray(inflow_speed)
    except: 'inflow_speed must be of type np.ndarray'
    
    assert isinstance(capture_area, (float,int)), 'capture_area must be of type int or float'
    assert isinstance(rho, (float,int)), 'rho must be of type int or float'

    # Predicted power from inflow
    power_in = (0.5 * rho * capture_area * inflow_speed**3)

    Cp = power / power_in 

    return Cp