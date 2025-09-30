"""
Computes device metrics such as equivalent diameter, tip speed ratio,
and capture area. Calculations are based on IEC TS 62600-300:2019 ED1.

"""

from typing import Union, Tuple, List
import numpy as np


def circular(diameter: Union[int, float]) -> Tuple[float, float]:
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
    if not isinstance(diameter, (int, float)):
        raise TypeError(f"diameter must be of type int or float. Got: {type(diameter)}")

    equivalent_diameter = diameter
    projected_capture_area = (1 / 4) * np.pi * (equivalent_diameter**2)

    return equivalent_diameter, projected_capture_area


def ducted(duct_diameter: Union[int, float]) -> Tuple[float, float]:
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
    if not isinstance(duct_diameter, (int, float)):
        raise TypeError(
            f"duct_diameter must be of type int or float. Got: {type(duct_diameter)}"
        )

    equivalent_diameter = duct_diameter
    projected_capture_area = (1 / 4) * np.pi * (equivalent_diameter**2)

    return equivalent_diameter, projected_capture_area


def rectangular(h: Union[int, float], w: Union[int, float]) -> Tuple[float, float]:
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
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(w, (int, float)):
        raise TypeError(f"w must be of type int or float. Got: {type(w)}")

    equivalent_diameter = np.sqrt(4.0 * h * w / np.pi)
    projected_capture_area = h * w

    return equivalent_diameter, projected_capture_area


def multiple_circular(diameters: List[Union[int, float]]) -> Tuple[float, float]:
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
    if not isinstance(diameters, list):
        raise TypeError(f"diameters must be of type list. Got: {type(diameters)}")

    diameters_squared = [x**2 for x in diameters]
    equivalent_diameter = np.sqrt(sum(diameters_squared))
    projected_capture_area = 0.25 * np.pi * sum(diameters_squared)

    return equivalent_diameter, projected_capture_area


def tip_speed_ratio(
    rotor_speed: Union[np.ndarray, List[Union[int, float]]],
    rotor_diameter: Union[int, float],
    inflow_speed: Union[np.ndarray, List[Union[int, float]]],
) -> np.ndarray:
    """
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
    tip_speed_ratio_values : numpy array
        Calculated tip speed ratio (TSR)
    """

    try:
        rotor_speed = np.asarray(rotor_speed)
    except (ValueError, TypeError) as exc:
        raise TypeError("rotor_speed must be convertible to np.ndarray") from exc

    try:
        inflow_speed = np.asarray(inflow_speed)
    except (ValueError, TypeError) as exc:
        raise TypeError("inflow_speed must be convertible to np.ndarray") from exc

    if not isinstance(rotor_diameter, (float, int)):
        raise TypeError(
            f"rotor_diameter must be of type int or float. Got: {type(rotor_diameter)}"
        )

    rotor_velocity = rotor_speed * np.pi * rotor_diameter

    tip_speed_ratio_values = rotor_velocity / inflow_speed

    return tip_speed_ratio_values


def power_coefficient(
    power: Union[np.ndarray, List[Union[int, float]]],
    inflow_speed: Union[np.ndarray, List[Union[int, float]]],
    capture_area: Union[int, float],
    rho: Union[int, float],
) -> np.ndarray:
    """
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
    power_coeff : numpy array
        Power coefficient of device [-]
    """

    try:
        power = np.asarray(power)
    except (ValueError, TypeError) as exc:
        raise TypeError("power must be convertible to np.ndarray") from exc

    try:
        inflow_speed = np.asarray(inflow_speed)
    except (ValueError, TypeError) as exc:
        raise TypeError("inflow_speed must be convertible to np.ndarray") from exc

    if not isinstance(capture_area, (float, int)):
        raise TypeError(
            f"capture_area must be of type int or float. Got: {type(capture_area)}"
        )
    if not isinstance(rho, (float, int)):
        raise TypeError(f"rho must be of type int or float. Got: {type(rho)}")

    # Predicted power from inflow
    power_in = 0.5 * rho * capture_area * inflow_speed**3

    power_coeff = power / power_in

    return power_coeff
