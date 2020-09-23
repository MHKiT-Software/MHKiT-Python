import pandas as pd
import numpy as np
import netCDF4
import datetime
import time
import calendar


def request_data(stn,startdate,enddate):
    """
    Requests CDIP data by wave buoy data file (from http://cdip.ucsd.edu/).
    

    Parameters
    ------------
    stn: string
        Station number of CDIP wave buoy
    startdate: string
        Start date in MM/DD/YYYY, e.g. '04/01/2012'
    enddate: string
        End date in MM/DD/YYYY, e.g. '04/30/2012'
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to header row 
        
    """
    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    
    # CDIP Realtime Dataset URL
    # data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
    
    # Open Remote Dataset from CDIP THREDDS Server
    nc = netCDF4.Dataset(data_url)
    
    # Read Buoy Variables
    ncTime = nc.variables['sstTime'][:]
    timeall = [datetime.datetime.fromtimestamp(t) for t in ncTime] # Convert ncTime variable to datetime stamps
    Hs = nc.variables['waveHs']
    Tp = nc.variables['waveTp']
    Dp = nc.variables['waveDp'] 
    
    # Create a variable of the Buoy Name and Month Name, to use in plot title
    buoyname = nc.variables['metaStationName'][:]
    buoytitle = buoyname[:-40].data.tostring().decode()
    
    month_name = calendar.month_name[int(startdate[0:2])]
    year_num = (startdate[6:10])
    
    ##################################
    # Local Indexing Functions
    ##################################
    # Find nearest value in numpy array
    def _find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    
    # Convert from human-format to UNIX timestamp
    def _getUnixTimestamp(humanTime,dateFormat):
        unixTimestamp = int(time.mktime(datetime.datetime.strptime(humanTime, dateFormat).timetuple()))
        return unixTimestamp
    
    
    ##################################
    # Time Index Values
    ##################################
    unixstart = _getUnixTimestamp(startdate,"%m/%d/%Y") 
    neareststart = _find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
    nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date
    
    unixend = _getUnixTimestamp(enddate,"%m/%d/%Y")
    future = _find_nearest(ncTime, unixend)  # Find the closest unix timestamp
    futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date    
    
    timeall = timeall[nearIndex:futureIndex]
    Hs = Hs[nearIndex:futureIndex]
    Tp = Tp[nearIndex:futureIndex]
    Dp = Dp[nearIndex:futureIndex]
    
    return timeall, Hs, Tp, Dp


def _read_file(file_name, missing_values=['MM',9999,999,99]):
    """
    Reads a CDIP wave buoy data file (from http://cdip.ucsd.edu/).
    

    Parameters
    ------------
    file_name: string
        Name of NDBC wave buoy data file
    
    missing_value: list of values
        List of values that denote missing data    
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to header row 
        
    metadata: dict or None
        Dictionary with {column name: units} key value pairs when the CDIP file  
        contains unit information, otherwise None is returned
    """

    
    return data, metadata
