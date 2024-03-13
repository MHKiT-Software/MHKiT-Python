import xarray as xr
import numpy as np
from scipy.stats import linregress as _linregress
from scipy.stats import rv_histogram as _rv_histogram
from mhkit.utils import convert_to_dataarray


def Froude_number(v, h, g=9.80665):
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
    Fr : float
        Froude Number of the river [unitless].

    """
    if not isinstance(v, (int, float)):
        raise TypeError(f"v must be of type int or float. Got: {type(v)}")
    if not isinstance(h, (int, float)):
        raise TypeError(f"h must be of type int or float. Got: {type(h)}")
    if not isinstance(g, (int, float)):
        raise TypeError(f"g must be of type int or float. Got: {type(g)}")

    Fr = v / np.sqrt(g * h)

    return Fr


def exceedance_probability(D, dimension="", to_pandas=True):
    """
    Calculates the exceedance probability

    Parameters
    ----------
    D : pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Discharge indexed by time [datetime or s].

    dimension: string (optional)
        Name of the relevant xarray dimension. If not supplied,
        defaults to the first dimension. Does not affect pandas input.

    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    F : pandas DataFrame or xarray Dataset
        Exceedance probability [unitless] indexed by time [datetime or s]
    """
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    D = convert_to_dataarray(D)

    if dimension == "":
        dimension = list(D.coords)[0]

    # Calculate exceedance probability (F)
    rank = D.rank(dim=dimension)
    rank = len(D[dimension]) - rank + 1  # convert to descending rank
    F = 100 * rank / (len(D[dimension]) + 1)
    F.name = "F"

    F = F.to_dataset()  # for matlab

    if to_pandas:
        F = F.to_pandas()

    return F


def polynomial_fit(x, y, n):
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
    R2 : float
        Polynomical fit coeffcient of determination

    """
    try:
        x = np.array(x)
    except:
        pass
    try:
        y = np.array(y)
    except:
        pass
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type np.ndarray. Got: {type(x)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be of type np.ndarray. Got: {type(y)}")
    if not isinstance(n, int):
        raise TypeError(f"n must be of type int. Got: {type(n)}")

    # Get coeffcients of polynomial of order n
    polynomial_coefficients = np.poly1d(np.polyfit(x, y, n))

    # Calculate the coeffcient of determination
    slope, intercept, r_value, p_value, std_err = _linregress(
        y, polynomial_coefficients(x)
    )
    R2 = r_value**2

    return polynomial_coefficients, R2


def discharge_to_velocity(D, polynomial_coefficients, dimension="", to_pandas=True):
    """
    Calculates velocity given discharge data and the relationship between
    discharge and velocity at an individual turbine

    Parameters
    ------------
    D : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
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
    V: pandas DataFrame or xarray Dataset
        Velocity [m/s] indexed by time [datetime or s]
    """
    if not isinstance(polynomial_coefficients, np.poly1d):
        raise TypeError(
            f"polynomial_coefficients must be of type np.poly1d. Got: {type(polynomial_coefficients)}"
        )
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type str. Got: {type(to_pandas)}")

    D = convert_to_dataarray(D)

    if dimension == "":
        dimension = list(D.coords)[0]

    # Calculate velocity using polynomial
    V = xr.DataArray(
        data=polynomial_coefficients(D),
        dims=dimension,
        coords={dimension: D[dimension]},
    )
    V.name = "V"

    V = V.to_dataset()  # for matlab

    if to_pandas:
        V = V.to_pandas()

    return V


def velocity_to_power(
    V, polynomial_coefficients, cut_in, cut_out, dimension="", to_pandas=True
):
    """
    Calculates power given velocity data and the relationship
    between velocity and power from an individual turbine

    Parameters
    ----------
    V : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Velocity [m/s] indexed by time [datetime or s]
    polynomial_coefficients : numpy polynomial
        List of polynomial coefficients that describe the relationship between
        velocity and power at an individual turbine
    cut_in: int/float
        Velocity values below cut_in are not used to compute P
    cut_out: int/float
        Velocity values above cut_out are not used to compute P
    dimension: string (optional)
        Name of the relevant xarray dimension. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    P : pandas DataFrame or xarray Dataset
        Power [W] indexed by time [datetime or s]
    """
    if not isinstance(polynomial_coefficients, np.poly1d):
        raise TypeError(
            f"polynomial_coefficients must be of type np.poly1d. Got: {type(polynomial_coefficients)}"
        )
    if not isinstance(cut_in, (int, float)):
        raise TypeError(f"cut_in must be of type int or float. Got: {type(cut_in)}")
    if not isinstance(cut_out, (int, float)):
        raise TypeError(f"cut_out must be of type int or float. Got: {type(cut_out)}")
    if not isinstance(dimension, str):
        raise TypeError(f"dimension must be of type str. Got: {type(dimension)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type str. Got: {type(to_pandas)}")

    V = convert_to_dataarray(V)

    if dimension == "":
        dimension = list(V.coords)[0]

    # Calculate velocity using polynomial
    power = polynomial_coefficients(V)

    # Power for velocity values outside lower and upper bounds Turbine produces 0 power
    power[V < cut_in] = 0.0
    power[V > cut_out] = 0.0

    P = xr.DataArray(data=power, dims=dimension, coords={dimension: V[dimension]})
    P.name = "P"

    P = P.to_dataset()

    if to_pandas:
        P = P.to_pandas()

    return P


def energy_produced(P, seconds):
    """
    Returns the energy produced for a given time period provided
    exceedance probability and power.

    Parameters
    ----------
    P : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Power [W] indexed by time [datetime or s]
    seconds: int or float
        Seconds in the time period of interest

    Returns
    -------
    E : float
        Energy [J] produced in the given length of time
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be of type int or float. Got: {type(seconds)}")

    P = convert_to_dataarray(P)

    # Calculate Histogram of power
    H, edges = np.histogram(P, 100)
    # Create a distribution
    hist_dist = _rv_histogram([H, edges])
    # Sample range for pdf
    x = np.linspace(edges.min(), edges.max(), 1000)
    # Calculate the expected value of Power
    expected_val_of_power = np.trapz(x * hist_dist.pdf(x), x=x)
    # Note: Built-in Expected Value method often throws warning
    # EV = hist_dist.expect(lb=edges.min(), ub=edges.max())
    # Energy
    E = seconds * expected_val_of_power

    return E
