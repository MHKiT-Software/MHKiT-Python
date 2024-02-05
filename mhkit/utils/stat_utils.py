from mhkit import qc
import pandas as pd
import numpy as np


def get_statistics(data, freq, period=600, vector_channels=[]):
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
    # Check data type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be of type pd.DataFrame. Got: {type(data)}")
    if not isinstance(freq, (float, int)):
        raise TypeError(f"freq must be of type int or float. Got: {type(freq)}")
    if not isinstance(period, (float, int)):
        raise TypeError(f"period must be of type int or float. Got: {type(period)}")
    # catch if vector_channels is not an string array
    if isinstance(vector_channels, str):
        vector_channels = [vector_channels]
    if not isinstance(vector_channels, list):
        raise TypeError(
            f"vector_channels must be a list of strings. Got: {type(vector_channels)}"
        )

    # Check timestamp using qc module
    data.index = data.index.round("1ms")
    dataQC = qc.check_timestamp(data, 1 / freq)
    dataQC = dataQC["cleaned_data"]

    # Check to see if data length contains enough data points for statistical window
    if len(dataQC) % (period * freq) > 0:
        remain = len(dataQC) % (period * freq)
        dataQC = dataQC.iloc[0 : -int(remain)]
        print(
            "WARNING: there were not enough data points in the last statistical period. Last "
            + str(remain)
            + " points were removed."
        )

    # Pre-allocate lists
    time = []
    means = []
    maxs = []
    mins = []
    stdev = []

    # Get data chunks to performs stats on
    step = period * freq
    for i in range(int(len(dataQC) / (period * freq))):
        datachunk = dataQC.iloc[i * step : (i + 1) * step]
        # Check whether there are any NaNs in datachunk
        if datachunk.isnull().any().any():
            print("NaNs found in statistical window...check timestamps!")
            input("Press <ENTER> to continue")
            continue
        else:
            # Get stats
            time.append(datachunk.index.values[0])  # time vector
            maxs.append(datachunk.max())  # maxes
            mins.append(datachunk.min())  # mins
            means.append(datachunk.mean())  # means
            stdev.append(datachunk.std())  # standard deviation
            # calculate vector averages and std
            for v in vector_channels:
                vector_avg, vector_std = vector_statistics(datachunk[v])
                # overwrite scalar average for channel
                means[i][v] = vector_avg
                stdev[i][v] = vector_std  # overwrite scalar std for channel

    # Convert to DataFrames and set index
    means = pd.DataFrame(means, index=time)
    maxs = pd.DataFrame(maxs, index=time)
    mins = pd.DataFrame(mins, index=time)
    stdevs = pd.DataFrame(stdev, index=time)

    return means, maxs, mins, stdevs


def vector_statistics(data):
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
    except:
        pass
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # calculate mean
    Ux = sum(np.sin(data * np.pi / 180)) / len(data)
    Uy = sum(np.cos(data * np.pi / 180)) / len(data)
    vector_avg = 90 - np.arctan2(Uy, Ux) * 180 / np.pi
    if vector_avg < 0:
        vector_avg = vector_avg + 360
    elif vector_avg > 360:
        vector_avg = vector_avg - 360
    # calculate standard deviation
    # round to 8th decimal place to reduce roundoff error
    magsum = round((Ux**2 + Uy**2) * 1e8) / 1e8
    epsilon = (1 - magsum) ** 0.5
    if not np.isreal(epsilon):  # check if epsilon is imaginary (error)
        vector_std = 0
        print("WARNING: epsilon contains imaginary value")
    else:
        vector_std = np.arcsin(epsilon) * (1 + 0.1547 * epsilon**3) * 180 / np.pi

    return vector_avg, vector_std


def unwrap_vector(data):
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
    except:
        pass
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # Loop through and unwrap points
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = data[i] + 360
        elif data[i] > 360:
            data[i] = data[i] - 360
    if max(data) > 360 or min(data) < 0:
        data = unwrap_vector(data)
    return data


def magnitude_phase(x, y, z=None):
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

    threeD = False
    if not isinstance(z, type(None)):
        z = np.array(z)
        threeD = True

    if not isinstance(x, (float, int, np.ndarray)):
        raise TypeError(f"x must be of type float, int, or np.ndarray. Got: {type(x)}")
    if not isinstance(y, (float, int, np.ndarray)):
        raise TypeError(f"y must be of type float, int, or np.ndarray. Got: {type(y)}")
    if not isinstance(z, (type(None), float, int, np.ndarray)):
        raise TypeError(
            f"If specified, z must be of type float, int, or np.ndarray. Got: {type(z)}"
        )

    if threeD:
        mag = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(np.sqrt(x**2 + y**2), z)
        return mag, theta, phi
    else:
        mag = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return mag, theta


def unorm(x, y, z):
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
    unorm : array
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
    unorm = np.linalg.norm(xyz, axis=0)

    return unorm
