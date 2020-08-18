from pecos.utils import index_to_datetime
from mhkit.wave.io import _ndbc_supported_params
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
                

def ndbc_dates_to_datetime(parameter, data, 
                           return_date_cols=False):
    '''
    Takes a DataFrame and converts the NDBC date columns 
	(e.g. "#YY  MM DD hh mm") to datetime. Returns a DataFrame with the 
	removed NDBC date columns a new ['date'] columns with DateTime Format.
    
    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'
        'stdmet':   'Standard Meteorological Current Year Historical Data'
    data: DataFrame
        Dataframe with headers (e.g. ['YY', 'MM', 'DD', 'hh', {'mm'}])
    return_date_col: Bool (optional)
        Default False. When true will return list of NDBC date columns
            
        
    Returns
    -------
    date: Series
        Series with NDBC dates dropped and new ['date']
        column in DateTime format
    ndbc_date_cols: list
        List of the DataFrame columns headers for dates as provided by 
        NDBC
    '''
    assert isinstance(data, pd.DataFrame), 'filenames must be of type pd.DataFrame' 
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(return_date_cols, bool), 'return_date_cols must be of type bool'
    supported =_ndbc_supported_params(parameter)
      
    df = data.copy(deep=True)     
    cols = df.columns.values.tolist()
    
    try:
        minutes_loc  = cols.index('mm')
        minutes=True
    except:
        minutes=False
    
    row_0_is_units = False
    year_string = [ col for col in  cols if col.startswith('Y')]
    if not year_string:
        year_string = [ col for col in  cols if col.startswith('#')]        
        if not year_string:
            print(f'ERROR: Could Not Find Year Column in {cols}')
        year_string = year_string[0]
        year_fmt = '%Y'        
        if str(df[year_string][0]).startswith('#'):
            row_0_is_units = True
            df = df.drop(df.index[0])
           
    elif year_string[0] == 'YYYY':
        year_string = year_string[0]
        year_fmt = '%Y'
    elif year_string[0]=='YY':
        year_string = year_string[0]
        year_fmt = '%y' 
    if minutes:
        ndbc_date_cols = [year_string, 'MM', 'DD', 'hh', 'mm']
    else:
        ndbc_date_cols = [year_string, 'MM', 'DD', 'hh']

               
    df = _date_string_to_datetime(df, ndbc_date_cols, year_fmt)        
    date = df['date']    
    if row_0_is_units:
        date = pd.concat([pd.Series([np.nan]),date])    
    del df
    
    if return_date_cols:
        return date, ndbc_date_cols
    return date

    
def _date_string_to_datetime(df, columns, year_fmt):
    '''
    Takes a NDBC df and creates a datetime from multiple columns headers
    by combining each column into a single string. Then the datetime 
    method is applied  given the expected format. 
    
    Parameters
    ----------
    df: DataFrame
        Dataframe with columns (e.g. ['YY', 'MM', 'DD', 'hh', {'mm'}])
    columns: list 
        list of strings for the columns to consider   
        (e.g. ['YY', 'MM', 'DD', 'hh', {'mm'}])
    year_fmt: str
        Specifies if year is 2 digit or 4 digit for datetime 
        interpretation
       
    Returns
    -------
    df: DataFrame
        The passed df with a new column ['date'] with the datetime format           
    '''
    assert isinstance(df, pd.DataFrame), 'df must be of type pd.DataFrame' 
    assert isinstance(columns, list), 'Columns must be a list'
    assert isinstance(year_fmt, str), 'year_fmt must be a string'
    
    # Convert to str and zero pad
    for key in columns:
        df[key] = df[key].astype(str).str.zfill(2)
    
    df['date_string'] = df[columns[0]]
    for column in columns[1:]:
        df['date_string'] = df[['date_string', column]].apply(lambda x: ''.join(x), axis=1)
    df['date'] = pd.to_datetime(df['date_string'], format=f'{year_fmt}%m%d%H%M')
    del df['date_string']
    
    return df
    