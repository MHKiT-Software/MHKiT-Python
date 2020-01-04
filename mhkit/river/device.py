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
