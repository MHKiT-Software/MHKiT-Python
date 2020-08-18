from pecos.utils import index_to_datetime
import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt 
from mhkit import qc

_matlab = False # Private variable indicating if mhkit is run through matlab

def get_statistics(data,freq,period=600):
    """
    Calculate mean, max, min and stdev statistics of continuous data for a 
    given statistical window. Default length of statistical window (period) is
    based on IEC TS 62600-3:2020 ED1. Also allows calculation of statistics for multiple statistical
    windows of continuous data.

    Parameters
    ------------
    data : pandas DataFrame
        Data indexed by datetime with columns of data to be analyzed 
    freq : float/int
        Sample rate of data [Hz]
    period : float/int
        Statistical window of interest [sec], default = 600 
    
    Returns
    ---------
    means,maxs,mins,stdevs : pandas DataFrame
        Calculated statistical values from the data, indexed by the first timestamp
    """
    # Check data type
    assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFrame'
    assert isinstance(freq, (float,int)), 'freq must be of type int or float'
    assert isinstance(period, (float,int)), 'freq must be of type int or float'

    # Check timestamp using qc module
    data.index = data.index.round('1ms')
    dataQC = qc.check_timestamp(data,1/freq)
    dataQC = dataQC['cleaned_data']
    
    # Check to see if data length contains enough data points for statistical window
    if len(dataQC)%(period*freq) > 0:
        remain = len(dataQC) % (period*freq)
        dataQC = dataQC.iloc[0:-int(remain)]
        print('WARNING: there were not enough data points in the last statistical period. Last '+str(remain)+' points were removed.')
    
    # Pre-allocate lists
    time = []
    means = []
    maxs = []
    mins = []
    stdev = []

    # Get data chunks to performs stats on
    step = period*freq
    for i in range(int(len(dataQC)/(period*freq))):
        datachunk = dataQC.iloc[i*step:(i+1)*step]
        # Check whether there are any NaNs in datachunk
        if datachunk.isnull().any().any(): 
            print('NaNs found in statistical window...check timestamps!')
            input('Press <ENTER> to continue')
            continue
        else:
            # Get stats
            time.append(datachunk.index.values[0])
            means.append(datachunk.mean())
            maxs.append(datachunk.max())
            mins.append(datachunk.min())
            stdev.append(datachunk.std())

    # Convert to DataFrames and set index
    means = pd.DataFrame(means,index=time)
    maxs = pd.DataFrame(maxs,index=time)
    mins = pd.DataFrame(mins,index=time)
    stdevs = pd.DataFrame(stdev,index=time)

    return means,maxs,mins,stdevs

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
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    # Loop through and unwrap points
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = data[i]+360
        elif data[i] > 360:
            data[i] = data[i]-360
    if max(data) > 360 or min(data) < 0:
        data = unwrap_vector(data)
    return data

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
        matlab_datenum = np.array(matlab_datenum,ndmin=1)
    except:
        pass
    assert isinstance(matlab_datenum, np.ndarray), 'data must be of type np.ndarray'

    # Pre-allocate
    time = []
    # loop through dates and convert
    for t in matlab_datenum:
        day = dt.datetime.fromordinal(int(t))
        dayfrac = dt.timedelta(days=t%1) - dt.timedelta(days = 366)
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
    assert isinstance(excel_num, np.ndarray), 'data must be of type np.ndarray'

    # Convert to datetime
    time = pd.to_datetime('1899-12-30')+pd.to_timedelta(excel_num,'D')

    return time                
    