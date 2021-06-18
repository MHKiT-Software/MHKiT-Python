from pecos.utils import index_to_datetime
import matplotlib.pyplot as plt 
import datetime as dt
from mhkit import qc
import pandas as pd 
import numpy as np 


_matlab = False # Private variable indicating if mhkit is run through matlab

def get_statistics(data,freq,period=600,vector_channels=[]):
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
    assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFrame'
    assert isinstance(freq, (float,int)), 'freq must be of type int or float'
    assert isinstance(period, (float,int)), 'freq must be of type int or float'
    # catch if vector_channels is not an string array
    if isinstance(vector_channels,str): vector_channels = [vector_channels]
    assert isinstance(vector_channels, list), 'vector_channels must be a list of strings'

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
            time.append(datachunk.index.values[0]) # time vector
            maxs.append(datachunk.max()) # maxes
            mins.append(datachunk.min()) # mins
            means.append(datachunk.mean()) # means
            stdev.append(datachunk.std()) # standard deviation
            # calculate vector averages and std
            for v in vector_channels:
                vector_avg, vector_std = vector_statistics(datachunk[v])            
                means[i][v] = vector_avg # overwrite scalar average for channel
                stdev[i][v] = vector_std # overwrite scalar std for channel
        
    # Convert to DataFrames and set index
    means = pd.DataFrame(means,index=time)
    maxs = pd.DataFrame(maxs,index=time)
    mins = pd.DataFrame(mins,index=time)
    stdevs = pd.DataFrame(stdev,index=time)

    return means,maxs,mins,stdevs

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
    try: data = np.array(data)
    except: pass
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'
    
    # calculate mean
    Ux = sum(np.sin(data*np.pi/180))/len(data)
    Uy = sum(np.cos(data*np.pi/180))/len(data)
    vector_avg = (90 - np.arctan2(Uy,Ux)*180/np.pi)
    if vector_avg<0: vector_avg = vector_avg+360
    elif vector_avg>360: vector_avg = vector_avg-360
    # calculate standard deviation              
    magsum = round((Ux**2 + Uy**2)*1e8)/1e8 # round to 8th decimal place to reduce roundoff error
    epsilon = (1-magsum)**0.5
    if not np.isreal(epsilon): # check if epsilon is imaginary (error)
        vector_std = 0
        print('WARNING: epsilon contains imaginary value')
    else:
        vector_std = np.arcsin(epsilon)*(1+0.1547*epsilon**3)*180/np.pi

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
    
    
def magnitude_phase(x,y,z=None):
    '''
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
    '''
    x=np.array(x)
    y=np.array(y)

    threeD=False
    if not isinstance(z, type(None)):
        z=np.array(z)
        threeD=True
        
    assert isinstance(x, (float,int,np.ndarray))
    assert isinstance(y, (float,int,np.ndarray))
    assert isinstance(z, (type(None),float,int,np.ndarray))
        
    if threeD:
        mag = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y,x)
        phi = np.arctan2(np.sqrt(x**2+y**2),z)
        return mag, theta, phi
    else:
        mag = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return mag, theta
        
    