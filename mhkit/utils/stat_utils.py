"""
This module contains functions to perform various statistical calculations 
on continuous data. It includes functions for calculating statistics such as
mean, max, min, and standard deviation over specific windows, as well as functions
for vector/directional statistics. The module also provides utility functions 
to unwrap vectors, compute magnitudes and phases in 2D/3D, and calculate 
the root mean squared values of vector components.

Functions:
----------
- get_statistics: Calculates statistics for continuous data.
- vector_statistics: Calculates vector mean and standard deviation.
- unwrap_vector: Unwraps vector data to fall within a 0-360 degree range.
- magnitude_phase: Computes magnitude and phase for 2D or 3D data.
- unorm: Computes root mean squared value of 3D vectors.
"""

from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from mhkit import qc


def _calculate_statistics(
    datachunk: pd.DataFrame, vector_channels: List[str]
) -> Dict[str, Union[pd.Series, float]]:
    """
    Calculate the mean, max, min, and standard deviation for the given datachunk.
    Also calculate vector statistics for vector_channels.

    Parameters
    ----------
    datachunk : pandas DataFrame
        A chunk of data on which to perform statistics.
    vector_channels : list
        List of vector channel names formatted in deg (0-360).

    Returns
    -------
    stats : dict
        A dictionary containing 'means', 'maxs', 'mins', and 'stdevs'.
    """
    means = datachunk.mean()
    maxs = datachunk.max()
    mins = datachunk.min()
    stdevs = datachunk.std()

    for v in vector_channels:
        vector_avg, vector_std = vector_statistics(datachunk[v])
        # overwrite scalar average and std for channel
        means[v] = vector_avg
        stdevs[v] = vector_std

    return {"means": means, "maxs": maxs, "mins": mins, "stdevs": stdevs}


def get_statistics(
    data: pd.DataFrame,
    freq: Union[float, int],
    period: Union[float, int] = 600,
    vector_channels: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate mean, max, min and stdev statistics of continuous data for a
    given statistical window. Default length of statistical window (period) is
    based on IEC TS 62600-3:2020 ED1. Also allows calculation of statistics for multiple statistical
    windows of continuous data and accounts for vector/directional channels.

    Parameters
    ------------
    data : pandas DataFrame
        Data indexed by datetime with columns of data to be analyzed
    freq : float/int
        Sample rate of data [Hz]
    period : float/int
        Statistical window of interest [sec], default = 600
    vector_channels : string or list (optional)
        List of vector/directional channel names formatted in deg (0-360)

    Returns
    ---------
    means,maxs,mins,stdevs : pandas DataFrame
        Calculated statistical values from the data, indexed by the first timestamp
    """
    if vector_channels is None:
        vector_channels = []

    if isinstance(vector_channels, str):
        vector_channels = [vector_channels]

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be of type pd.DataFrame. Got: {type(data)}")
    if not isinstance(freq, (float, int)):
        raise TypeError(f"freq must be of type int or float. Got: {type(freq)}")
    if not isinstance(period, (float, int)):
        raise TypeError(f"period must be of type int or float. Got: {type(period)}")
    if not isinstance(vector_channels, list):
        raise TypeError(
            f"vector_channels must be a list of strings. Got: {type(vector_channels)}"
        )

    data.index = data.index.round("1ms")
    data_qc = qc.check_timestamp(data, 1 / freq)["cleaned_data"]

    if len(data_qc) % (period * freq) > 0:
        remain = len(data_qc) % (period * freq)
        data_qc = data_qc.iloc[0 : -int(remain)]
        print(
            f"WARNING: there were not enough data points in the last statistical period. \
              Last {remain} points were removed."
        )

    time = []
    means = []
    maxs = []
    mins = []
    stdevs = []

    step = period * freq
    for i in range(int(len(data_qc) / step)):
        datachunk = data_qc.iloc[i * step : (i + 1) * step]
        if datachunk.isnull().any().any():
            print("NaNs found in statistical window...check timestamps!")
            input("Press <ENTER> to continue")
            continue

        time.append(datachunk.index.values[0])

        # Calculate statistics for this chunk
        stats = _calculate_statistics(datachunk, vector_channels)

        means.append(stats["means"])
        maxs.append(stats["maxs"])
        mins.append(stats["mins"])
        stdevs.append(stats["stdevs"])

    # Convert lists to DataFrames
    means = pd.DataFrame(means, index=time)
    maxs = pd.DataFrame(maxs, index=time)
    mins = pd.DataFrame(mins, index=time)
    stdevs = pd.DataFrame(stdevs, index=time)

    return means, maxs, mins, stdevs


def vector_statistics(
    data: Union[pd.Series, np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function used to calculate statistics for vector/directional channels based on
    routine from Campbell data logger and Yamartino algorithm

    Parameters
    ----------
    data : pandas Series, numpy array, list
        Vector channel to calculate statistics on [deg, 0-360]

    Returns
    -------
    vector_avg : numpy array
        Vector mean statistic
    vector_std : numpy array
        Vector standard deviation statistic
    """
    try:
        data = np.array(data)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Error converting data to numpy array: {e}") from e

    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # calculate mean
    u_x = sum(np.sin(data * np.pi / 180)) / len(data)
    u_y = sum(np.cos(data * np.pi / 180)) / len(data)
    vector_avg = 90 - np.arctan2(u_y, u_x) * 180 / np.pi
    if vector_avg < 0:
        vector_avg = vector_avg + 360
    elif vector_avg > 360:
        vector_avg = vector_avg - 360
    # calculate standard deviation
    # round to 8th decimal place to reduce roundoff error
    magsum = round((u_x**2 + u_y**2) * 1e8) / 1e8
    epsilon = (1 - magsum) ** 0.5
    if not np.isreal(epsilon):  # check if epsilon is imaginary (error)
        vector_std = 0
        print("WARNING: epsilon contains imaginary value")
    else:
        vector_std = np.arcsin(epsilon) * (1 + 0.1547 * epsilon**3) * 180 / np.pi

    return vector_avg, vector_std


def unwrap_vector(data: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """
    Function used to unwrap vectors into 0-360 deg range

    Parameters
    ------------
    data : pandas Series, numpy array, list
        Data points to be unwrapped [deg]

    Returns
    ---------
    data : numpy array
        Data points unwrapped between 0-360 deg
    """
    # Check data types
    try:
        data = np.array(data)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Error converting data to numpy array: {e}") from e

    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # Loop through and unwrap points
    for i, value in enumerate(data):
        if value < 0:
            data[i] = value + 360
        elif value > 360:
            data[i] = value - 360

    if max(data) > 360 or min(data) < 0:
        data = unwrap_vector(data)
    return data


def magnitude_phase(
    x: Union[float, int, np.ndarray],
    y: Union[float, int, np.ndarray],
    z: Optional[Union[float, int, np.ndarray]] = None,
) -> Union[
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]],
]:
    """
    Retuns magnitude and phase in two or three dimensions.

    Parameters
    ----------
    x: array_like
        x-component
    y: array_like
        y-component
    z: array_like
        z-component defined positive up. (Optional) Default None.

    Returns
    -------
    mag: float or array
        magnitude of the vector
    theta: float or array
        radians from the x-axis
    phi: float or array
        radians from z-axis defined as positive up. Optional: only
        returned when z is passed.
    """
    x = np.array(x)
    y = np.array(y)

    three_d = False
    if not isinstance(z, type(None)):
        z = np.array(z)
        three_d = True

    if not isinstance(x, (float, int, np.ndarray)):
        raise TypeError(f"x must be of type float, int, or np.ndarray. Got: {type(x)}")
    if not isinstance(y, (float, int, np.ndarray)):
        raise TypeError(f"y must be of type float, int, or np.ndarray. Got: {type(y)}")
    if not isinstance(z, (type(None), float, int, np.ndarray)):
        raise TypeError(
            f"If specified, z must be of type float, int, or np.ndarray. Got: {type(z)}"
        )

    if three_d:
        mag = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(np.sqrt(x**2 + y**2), z)
        return mag, theta, phi

    mag = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return mag, theta


def unorm(
    x: Union[np.ndarray, np.float64, pd.Series],
    y: Union[np.ndarray, np.float64, pd.Series],
    z: Union[np.ndarray, np.float64, pd.Series],
) -> Union[np.ndarray, np.float64]:
    """
    Calculates the root mean squared value given three arrays.

    Parameters
    ----------
    x: array
        One input for the root mean squared calculation.(eq. x velocity)
    y: array
        One input for the root mean squared calculation.(eq. y velocity)
    z: array
        One input for the root mean squared calculation.(eq. z velocity)

    Returns
    -------
    u_norm : array
       The root mean squared of x, y, and z.

    Example
    -------
    If the inputs are [1,2,3], [4,5,6], and [7,8,9] the code take the
    cordinationg value from each array and calculates the root mean squared.
    The resulting output is [ 8.1240384,  9.64365076, 11.22497216].
    """

    if not isinstance(x, (np.ndarray, np.float64, pd.Series)):
        raise TypeError(
            f"x must be of type np.ndarray, np.float64, or pd.Series. Got: {type(x)}"
        )
    if not isinstance(y, (np.ndarray, np.float64, pd.Series)):
        raise TypeError(
            f"y must be of type np.ndarray, np.float64, or pd.Series. Got: {type(y)}"
        )
    if not isinstance(z, (np.ndarray, np.float64, pd.Series)):
        raise TypeError(
            f"z must be of type np.ndarray, np.float64, or pd.Series. Got: {type(z)}"
        )
    if not all([len(x) == len(y), len(y) == len(z)]):
        raise ValueError("lengths of arrays must match")

    xyz = np.array([x, y, z])
    u_norm = np.linalg.norm(xyz, axis=0)

    return u_norm
