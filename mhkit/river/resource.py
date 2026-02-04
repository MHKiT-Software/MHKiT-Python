"""
Computes resource assessment metrics, including exceedance probability,
inflow velocity, and power (theoretical resource). Calculations are based
on IEC TS 62600-301:2019 ED1.

"""

from typing import Union, Tuple
import xarray as xr
import numpy as np
from scipy.stats import linregress as _linregress
from scipy.stats import rv_histogram as _rv_histogram
from pandas import DataFrame, Series
from mhkit.utils import convert_to_dataarray


def froude_number(
    v: Union[int, float], h: Union[int, float], g: Union[int, float] = 9.80665
) -> float:
    """
    Calculate the Froude Number of the river, channel or duct flow,
    to check subcritical flow assumption (if Fr <1).

    Parameters
    ------------
    v : int/float
        Average velocity [m/s].
    h : int/float
        Mean hydraulic depth float [m].
    g : int/float
        Gravitational acceleration [m/s2].

    Returns
    ---------
    froude_num : float
        Froude Number of the river [unitless].

    """
    if not isinstance(v, (int, float)):
        raise TypeError(f"v must be of type int or float. Got: {type(v)}")
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")

    froude_num = v / np.sqrt(g * h)

    return froude_num


def exceedance_probability(
    discharge: Union[Series, DataFrame, xr.DataArray, xr.Dataset],
    dimension: str = "",
    to_pandas: bool = True,
) -> Union[DataFrame, xr.Dataset]:
    """
    Calculates the exceedance probability

    Parameters
    ----------
    discharge : pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Discharge indexed by time [datetime or s].

    dimension: string (optional)
        Name of the relevant xarray dimension. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    exceedance_prob : pandas DataFrame or xarray Dataset
        Exceedance probability [unitless] indexed by time [datetime or s]
    """
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    discharge = convert_to_dataarray(discharge)

    if dimension == "":
        dimension = list(discharge.coords)[0]

    # Calculate exceedance probability
    rank = discharge.rank(dim=dimension)
    rank = len(discharge[dimension]) - rank + 1  # convert to descending rank
    exceedance_prob = 100 * rank / (len(discharge[dimension]) + 1)
    exceedance_prob.name = "exceedance_probability"

    exceedance_prob = exceedance_prob.to_dataset()  # for matlab

    if to_pandas:
        exceedance_prob = exceedance_prob.to_pandas()

    return exceedance_prob


def polynomial_fit(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.poly1d, float]:
    """
    Returns a polynomial fit for y given x of order n
    with an R-squared score of the fit

    Parameters
    -----------
    x : numpy array
        x data for polynomial fit.
    y : numpy array
        y data for polynomial fit.
    n : int
        order of the polynomial fit.

    Returns
    ----------
    polynomial_coefficients : numpy polynomial
        List of polynomial coefficients
    r_squared : float
        Polynomial fit coefficient of determination

    """
    try:
        x = np.array(x)
    except (ValueError, TypeError) as exc:
        raise TypeError("x must be convertible to np.ndarray") from exc
    try:
        y = np.array(y)
    except (ValueError, TypeError) as exc:
        raise TypeError("y must be convertible to np.ndarray") from exc

    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be of type np.ndarray. Got: {type(y)}")
    if not isinstance(n, int):
        raise TypeError(f"n must be of type int. Got: {type(n)}")

    # Get coefficients of polynomial of order n
    polynomial_coefficients = np.poly1d(np.polyfit(x, y, n))

    # Calculate the coefficient of determination
    _, _, r_value, _, _ = _linregress(y, polynomial_coefficients(x))
    r_squared = r_value**2

    return polynomial_coefficients, r_squared


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def discharge_to_velocity(
    discharge: Union[np.ndarray, DataFrame, Series, xr.DataArray, xr.Dataset],
    polynomial_coefficients: np.poly1d,
    dimension: str = "",
    to_pandas: bool = True,
) -> Union[DataFrame, xr.Dataset]:
    """
    Calculates velocity given discharge data and the relationship between
    discharge and velocity at an individual turbine

    Parameters
    ------------
    discharge : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Discharge data [m3/s] indexed by time [datetime or s]
    polynomial_coefficients : numpy polynomial
        List of polynomial coefficients that describe the relationship between
        discharge and velocity at an individual turbine
    dimension: string (optional)
        Name of the relevant xarray dimension. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ------------
    velocity: pandas DataFrame or xarray Dataset
        Velocity [m/s] indexed by time [datetime or s]
    """
    if not isinstance(polynomial_coefficients, np.poly1d):
        raise TypeError(
            "polynomial_coefficients must be of "
            f"type np.poly1d. Got: {type(polynomial_coefficients)}"
        )
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type str. Got: {type(to_pandas)}")

    discharge = convert_to_dataarray(discharge)

    if dimension == "":
        dimension = list(discharge.coords)[0]

    # Calculate velocity using polynomial
    velocity = xr.DataArray(
        data=polynomial_coefficients(discharge),
        dims=dimension,
        coords={dimension: discharge[dimension]},
    )
    velocity.name = "velocity"

    velocity = velocity.to_dataset()  # for matlab

    if to_pandas:
        velocity = velocity.to_pandas()

    return velocity


def velocity_to_power(
    velocity: Union[np.ndarray, DataFrame, Series, xr.DataArray, xr.Dataset],
    polynomial_coefficients: np.poly1d,
    cut_in: Union[int, float],
    cut_out: Union[int, float],
    dimension: str = "",
    to_pandas: bool = True,
) -> Union[DataFrame, xr.Dataset]:
    """
    Calculates power given velocity data and the relationship
    between velocity and power from an individual turbine

    Parameters
    ----------
    velocity : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Velocity [m/s] indexed by time [datetime or s]
    polynomial_coefficients : numpy polynomial
        List of polynomial coefficients that describe the relationship between
        velocity and power at an individual turbine
    cut_in: int/float
        Velocity values below cut_in are not used to compute power
    cut_out: int/float
        Velocity values above cut_out are not used to compute power
    dimension: string (optional)
        Name of the relevant xarray dimension. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    power : pandas DataFrame or xarray Dataset
        Power [W] indexed by time [datetime or s]
    """
    if not isinstance(polynomial_coefficients, np.poly1d):
        raise TypeError(
            "polynomial_coefficients must be"
            f"of type np.poly1d. Got: {type(polynomial_coefficients)}"
        )
    if not isinstance(cut_in, (int, float)):
        raise TypeError(f"cut_in must be of type int or float. Got: {type(cut_in)}")
    if not isinstance(cut_out, (int, float)):
        raise TypeError(f"cut_out must be of type int or float. Got: {type(cut_out)}")
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    velocity = convert_to_dataarray(velocity)

    if dimension == "":
        dimension = list(velocity.coords)[0]

    # Calculate power using polynomial
    power_values = polynomial_coefficients(velocity)

    # Power for velocity values outside lower and upper bounds Turbine produces 0 power
    power_values[velocity < cut_in] = 0.0
    power_values[velocity > cut_out] = 0.0

    power = xr.DataArray(
        data=power_values, dims=dimension, coords={dimension: velocity[dimension]}
    )
    power.name = "power"

    power = power.to_dataset()

    if to_pandas:
        power = power.to_pandas()

    return power


def energy_produced(
    power_data: Union[np.ndarray, DataFrame, Series, xr.DataArray, xr.Dataset],
    seconds: Union[int, float],
) -> float:
    """
    Returns the energy produced for a given time period provided
    exceedance probability and power.

    Parameters
    ----------
    power_data : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Power [W] indexed by time [datetime or s]
    seconds: int or float
        Seconds in the time period of interest

    Returns
    -------
    energy : float
        Energy [J] produced in the given length of time
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be of type int or float. Got: {type(seconds)}")

    power_data = convert_to_dataarray(power_data)

    # Calculate histogram of power
    hist_values, edges = np.histogram(power_data, 100)
    # Create a distribution
    hist_dist = _rv_histogram([hist_values, edges])
    # Sample range for pdf
    x = np.linspace(edges.min(), edges.max(), 1000)
    # Calculate the expected value of power
    expected_power = np.trapezoid(x * hist_dist.pdf(x), x=x)
    # Note: Built-in Expected Value method often throws warning
    # EV = hist_dist.expect(lb=edges.min(), ub=edges.max())
    # Calculate energy
    energy = seconds * expected_power

    return energy
