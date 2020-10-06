import pandas as pd
import numpy as np
import netCDF4
import datetime
import time

def request_data(stn,start_date='',end_date='',yeardate='',data_type='Historic'):
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
    yeardate: string
        Year date, e.g. '2001'
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
    assert isinstance(yeardate, str), 'yeardate must be of type str'
    assert isinstance(data_type, str), 'data_type must be of type str'
    
    if start_date !='':        
        assert len(start_date) == 10, ('Start date must be of format MM/DD/YYYY'
        f' got: {start_date}')
        assert start_date.find('/') == 2, ('Start date must be of format MM/DD/YYYY'
        f' got: {start_date}')
    
    if end_date !='':        
        assert len(end_date) == 10, ('End date must be of format MM/DD/YYYY'
        f' got: {end_date}')
        assert end_date.find('/') == 2, ('End date must be of format MM/DD/YYYY'
        f' got: {end_date}')

    if yeardate !='':
        assert len(yeardate) == 4, ('Year date must be of format YYYY'
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
    if start_date !='':
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
        
    elif yeardate != '':        
        # Create array of month numbers to cycle through to grab Hs data
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        # Create array of lists of Hs data for each month 
        timeindex_start = []
        timeindex_end = []
        monthcount = 0
        
        for monthi in months:
            startdate = months[monthcount] + "/" + "01/" + str(yeardate) # Set start and end dates of each month, using the above 'months' array
            enddate = months[monthcount] + "/" + "28/" + str(yeardate) # Set the end date to Day 28, to account for February's short length
           
            unixstart = _getUnixTimestamp(startdate,"%m/%d/%Y")
            nearest_date = _find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
            near_index = np.where(ncTime==nearest_date)[0][0]  # Grab the index number of found date
            
            unixend = _getUnixTimestamp(enddate,"%m/%d/%Y")
            future_date = _find_nearest(ncTime, unixend)  # Find the closest unix timestamp
            future_index = np.where(ncTime==future_date)[0][0]  # Grab the index number of found date    
            
            monthcount = monthcount+1
            timeindex_start.append(near_index) # Append 'month start date' and 'month end date' index numbers for each month to corresponding array
            timeindex_end.append(future_index)
            
        timeall = timeall[timeindex_start[0]:timeindex_end[-1]]
        Hs = Hs[timeindex_start[0]:timeindex_end[-1]]                
        
        data = pd.DataFrame(data = {'Hs': Hs}, index = timeall)        
    
    return data, buoytitle


