import numpy as np

def circular(diameter):
    """
    Calculates the equivalent diameter and projected capture area of a 
    circular turbine
    
    Parameters
    ------------
    diameter : int/float
        Turbine diameter [m]
        
    Returns
    ---------
    equivalent_diameter : float
       Equivalent diameter [m]
    projected_capture_area : float
        Projected capture area [m^2]
    """
    assert isinstance(diameter, (int,float)), 'diameter must be of type int or float'
    
    equivalent_diameter = diameter
    projected_capture_area = 4.*np.pi*(equivalent_diameter**2.) 
    
    return equivalent_diameter, projected_capture_area

def ducted(duct_diameter):
    """
    Calculates the equivalent diameter and projected capture area of a
    ducted turbine
    
    Parameters
    ------------
    duct_diameter : int/float
        Duct diameter [m]
        
    Returns
    ---------
    equivalent_diameter : float
       Equivalent diameter [m]
    projected_capture_area : float
        Projected capture area [m^2]
    """
    assert isinstance(duct_diameter, (int,float)), 'duct_diameter must be of type int or float'
    
    equivalent_diameter = duct_diameter
    projected_capture_area = 4.*np.pi*(equivalent_diameter**2.) 

    return equivalent_diameter, projected_capture_area

def rectangular(h, w):
    """
    Calculates the equivalent diameter and projected capture area of a 
    retangular turbine
    
    Parameters
    ------------
    h : int/float
        Turbine height [m]
    w : int/float
        Turbine width [m]
        
    Returns
    ---------
    equivalent_diameter : float
       Equivalent diameter [m]
    projected_capture_area : float
        Projected capture area [m^2]
    """
    assert isinstance(h, (int,float)), 'h must be of type int or float'
    assert isinstance(w, (int,float)), 'w must be of type int or float'
    
    equivalent_diameter = np.sqrt(4.*h*w / np.pi) 
    projected_capture_area = h*w

    return equivalent_diameter, projected_capture_area

def multiple_circular(diameters):
    """
    Calculates the equivalent diameter and projected capture area of a 
    multiple circular turbine
    
    Parameters
    ------------
    diameters: list 
        List of device diameters [m]
        
    Returns
    ---------
    equivalent_diameter : float
       Equivalent diameter [m]
    projected_capture_area : float
        Projected capture area [m^2]
    """
    assert isinstance(diameters, list), 'diameters must be of type list'
    
    diameters_squared = [x**2 for x in diameters]
    equivalent_diameter = np.sqrt(sum(diameters_squared))
    projected_capture_area = 0.25*np.pi*sum(diameters_squared)

    return equivalent_diameter, projected_capture_area

def tip_speed_ratio(rotor_speed,rotor_diameter,inflow_speed):
    '''
    Function used to calculate the tip speed ratio (TSR) of a MEC device with rotor

    Parameters
    -----------
    rotor_speed : numpy array
        Rotor speed [revolutions per second]
    rotor_diameter : float/int
        Diameter of rotor [m]
    inflow_speed : numpy array
        Velocity of inflow condition [m/s]

    Returns
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

    Parameters
    -----------
    power : numpy array
        Power output signal of device after losses [W]
    inflow_speed : numpy array
        Speed of inflow [m/s]
    capture_area : float/int
        Projected area of rotor normal to inflow [m^2]
    rho : float/int
        Density of environment [kg/m^3]

    Returns
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
    
