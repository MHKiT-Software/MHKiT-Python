import datetime as dt
import pandas as pd
import numpy as np


def matlab_to_datetime(matlab_datenum):
    """
    Convert MATLAB datenum format to Python datetime

    Parameters
    ------------
    matlab_datenum : numpy array
        MATLAB datenum to be converted

    Returns
    ---------
    time : DateTimeIndex
        Python datetime values
    """
    # Check data types
    try:
        matlab_datenum = np.array(matlab_datenum, ndmin=1)
    except:
        pass
    if not isinstance(matlab_datenum, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(data)}")

    # Pre-allocate
    time = []
    # loop through dates and convert
    for t in matlab_datenum:
        day = dt.datetime.fromordinal(int(t))
        dayfrac = dt.timedelta(days=t % 1) - dt.timedelta(days=366)
        time.append(day + dayfrac)

    time = np.array(time)
    time = pd.to_datetime(time)
    return time


def excel_to_datetime(excel_num):
    """
    Convert Excel datenum format to Python datetime

    Parameters
    ------------
    excel_num : numpy array
        Excel datenums to be converted

    Returns
    ---------
    time : DateTimeIndex
        Python datetime values
    """
    # Check data types
    try:
        excel_num = np.array(excel_num)
    except:
        pass
    if not isinstance(excel_num, np.ndarray):
        raise TypeError(f"excel_num must be of type np.ndarray. Got: {type(excel_num)}")

    # Convert to datetime
    time = pd.to_datetime("1899-12-30") + pd.to_timedelta(excel_num, "D")

    return time
