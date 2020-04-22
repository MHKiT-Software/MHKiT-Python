from pecos.utils import index_to_datetime
import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt 
from mhkit import qc

_matlab = False # Private variable indicating if mhkit is run through matlab

def get_stats(data,freq,period=600):
    """
    function used to obtain statistics from a dataset

    Parameters:
    ----------------------
    data : pandas dataframe
        time-indexed dataframe containg variable(s) to be analyzed with statistical window
    period : float/int
        statistical window of interest (ex. 600 seconds) [sec]
    freq : float/int
        sample rate of data [Hz]
    
    Returns:
    ----------------------
    means,maxs,mins,stds : pandas dataframes
        dataframes containing calculated statistical values of data
    """
    # check data type
    assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFrame'
    assert isinstance(freq, (float,int)), 'freq must be of type int or float'
    assert isinstance(period, (float,int)), 'freq must be of type int or float'

    # check timestamp using qc module
    data.index = data.index.round('1ms')
    dataQC = qc.check_timestamp(data,1/freq)
    dataQC = dataQC['cleaned_data']
    
    # check to see if data length contains enough data points for statistical window
    if len(dataQC)%(period*freq) > 0:
        remain = len(dataQC) % (period*freq)
        dataQC = dataQC.iloc[0:-int(remain)]
        print('WARNING: there were not enought data points in the last statistical period. Last '+str(remain)+' points were removed.')
    
    # pre-allocate lists
    time = []
    means = []
    maxs = []
    mins = []
    stdev = []

    # grab data chunks to performs stats on
    step = period*freq
    for i in range(int(len(dataQC)/(period*freq))):
        datachunk = dataQC.iloc[i*step:(i+1)*step]
        # check whether there are any NaNs in datachunk
        if datachunk.isnull().any().any(): 
            continue
        else:
            # get stats
            time.append(datachunk.index.values[0])
            means.append(datachunk.mean())
            maxs.append(datachunk.max())
            mins.append(datachunk.min())
            stdev.append(datachunk.std())

    # convert to dataframes and set index
    means = pd.DataFrame(means,index=time)
    maxs = pd.DataFrame(maxs,index=time)
    mins = pd.DataFrame(mins,index=time)
    stdev = pd.DataFrame(stdev,index=time)

    # TODO: handle vector averaging

    return means,maxs,mins,stdev

def unwrapvec(data):
    """
    function used to unwrap vectors into 0-360 deg range

    Parameters:
    ---------------
    data : pd.Series, numpy array, list
        list of data points to be unwrapped [deg]
    
    Returns:
    --------------
    data : numpy array
        returns list of data points unwrapped between 0-360 deg
    """
    # check data types
    try:
        data = np.array(data)
    except:
        pass
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    # loop through and unwrap points
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = data[i]+360
        elif data[i] > 360:
            data[i] = data[i]-360
    if max(data) > 360 or min(data) < 0:
        data = unwrapvec(data)
    return data

def matlab2datetime(matlab_datenum):
    """
    conversion of matlab datenum format to python datetime

    Parameters:
    ----------------
    matlab_datenum : np.array
        array of matlab datenum to be converted

    Returns:
    -----------------
    time : np.array
        array of corresponding python datetime values
    """
    # check data types
    try:
        matlab_datenum = np.array(matlab_datenum)
    except:
        pass
    assert isinstance(matlab_datenum, np.ndarray), 'data must be of type np.ndarray'

    # pre-allocate
    time = []
    # loop through dates and convert
    for t in matlab_datenum:
        day = dt.datetime.fromordinal(int(t))
        dayfrac = dt.timedelta(days=t%1) - dt.timedelta(days = 366)
        time.append(day + dayfrac)
    
    time = np.array(time)
    return time

def excel2datetime(excel_num):
    """
    conversion of matlab datenum format to python datetime

    Parameters:
    ----------------
    matlab_datenum : np.array
        array of matlab datenum to be converted

    Returns:
    -----------------
    time : np.array
        array of corresponding python datetime values
    """
    # check data types
    try:
        excel_num = np.array(excel_num)
    except:
        pass
    assert isinstance(excel_num, np.ndarray), 'data must be of type np.ndarray'

    # convert to datetime
    time = pd.to_datetime('1899-12-30')+pd.to_timedelta(excel_num,'D')

    return time
