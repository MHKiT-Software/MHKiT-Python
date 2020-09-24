import pandas as pd
import numpy as np
import netCDF4
import datetime
import time

def request_data(stn,start_date,end_date,data_type='Historic'):
    """
    Requests CDIP data by wave buoy data file (from http://cdip.ucsd.edu/).
    

    Parameters
    ------------
    stn: string
        Station number of CDIP wave buoy
    start_date: string
        Start date in MM/DD/YYYY, e.g. '04/01/2012'
    end_date: string
        End date in MM/DD/YYYY, e.g. '04/30/2012'
    data_type: string
        'Realtime' or 'Historic', default = 'Historic'
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to the data 
        signal, for it includes: Hs, Tp, and Dp
        
    """
    assert isinstance(stn, str), 'stn must be of type str'
    assert isinstance(start_date, str), 'start_date must be of type str'
    assert isinstance(end_date, str), 'end_date must be of type str'
    assert isinstance(data_type, str), 'data_type must be of type str'
    
    if isinstance(start_date, str):        
        assert len(start_date) == 10, ('Start time must be of format MM/DD/YYYY'
        f' got: {start_date}')
        assert start_date.find('/') == 2, ('Start time must be of format MM/DD/YYYY'
        f' got: {start_date}')
    
    if isinstance(end_date, str):        
        assert len(end_date) == 10, ('End time must be of format MM/DD/YYYY'
        f' got: {end_date}')
        assert end_date.find('/') == 2, ('End time must be of format MM/DD/YYYY'
        f' got: {end_date}')

    if isinstance(data_type, str):        
        assert data_type == 'Realtime' or data_type == 'Historic', ('Data type must be either Historic or Realtime'
        f' got: {data_type}')

        
    # Access Historic or Realtime data from CDIP Thredds
    if data_type == 'Historic':
        # CDIP Archived Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    
    elif data_type == 'Realtime':
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
    
    # Open Remote Dataset from CDIP THREDDS Server
    nc = netCDF4.Dataset(data_url)
    ##################################
    # Avilable CDIP data
    ##################################
    # nc.variables.keys()

    # Create a variable of the Buoy Name and Month Name, to use in plot title
    buoyname = nc.variables['metaStationName'][:]
    buoytitle = buoyname[:-40].data.tostring().decode()
    
    # Read Buoy Variables
    # ncTime = nc.variables['sstTime'][:]    
    ncTime = nc.variables['waveTime'][:]
    # Convert ncTime variable to datetime stamps
    timeall = [datetime.datetime.fromtimestamp(t) for t in ncTime] 
    Hs = nc.variables['waveHs']
    Tp = nc.variables['waveTp']
    Dp = nc.variables['waveDp'] 
    
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
    unixstart = _getUnixTimestamp(start_date,"%m/%d/%Y") 
    neareststart = _find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
    nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date
    
    unixend = _getUnixTimestamp(end_date,"%m/%d/%Y")
    future = _find_nearest(ncTime, unixend)  # Find the closest unix timestamp
    futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date    
    
    timeall = timeall[nearIndex:futureIndex]
    Hs = Hs[nearIndex:futureIndex]
    Tp = Tp[nearIndex:futureIndex]
    Dp = Dp[nearIndex:futureIndex]
    
    data = pd.DataFrame(data = {'Hs': Hs,'Tp': Tp,'Dp': Dp}, index = timeall)
    
    return data, buoytitle


