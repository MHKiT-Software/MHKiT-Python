import pandas as pd
import numpy as np
import datetime
import netCDF4
import time

def _validate_date(date_text):
    '''
    Checks date format to ensure MM/DD/YYYY format
    
    Parameters
    ----------
    date_text: string
        Date string format to check
    Returns
    -------
    dt: datetime
    '''
    
    try:
        dt = datetime.datetime.strptime(date_text, '%m-%d-%Y')
    except ValueError:
        raise ValueError("Incorrect data format, should be MM-DD-YYYY")
    return dt


def request_netCDF(station_number, data_type):
    '''
    Returns historic or realtime data from CDIP THREDDS server
   
    Parameters:
    station_number: string
        CDIP station number of interest
    data_type: string
        Either 'Historic' or 'Realtime'
   
    Returns
    -------
    nc: netCDF Object
        netCDF data for the given station number and data type
    '''
   
    if data_type == 'Historic':
        cdip_archive= 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive'
        data_url =  f'{cdip_archive}/{station_number}p1/{station_number}p1_historic.nc'
    elif data_type == 'Realtime':
        cdip_realtime = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime'
        data_url = f'{cdip_realtime}/{station_number}p1_rt.nc'
    nc = netCDF4.Dataset(data_url)
    return nc


def request_data(station_number, year=None, start_date=None, 
                     end_date=None, data_type='Historic', include2DVars=False):
    """
    Requests CDIP data by wave buoy data file (from http://cdip.ucsd.edu/).
    
    Parameters
    ------------
    station_number: string
        Station number of CDIP wave buoy
    year: string 
        Year date, e.g. '2001'        
    start_date: string 
        Start date in MM-DD-YYYY, e.g. '04-01-2012'
    end_date: string 
        End date in MM/DD/YYYY, e.g. '04-30-2012'
    data_type: string
        Either 'Historic' or 'Realtime'   
    include2DVars: boolean
        Will return 2D data from cdip. Enabling this will add significant 
        processing time.        
    
    Returns
    ---------
    data: DataFrame 
        Data indexed by datetime with columns named according to the data 
        signal, for it includes: Hs, Tp, and Dp       
    """
    assert isinstance(station_number, str), f'station_number must be of type str'
    assert isinstance(start_date, (str, type(None))), 'start_date must be of type str'
    assert isinstance(end_date, (str, type(None))), 'end_date must be of type str'
    assert isinstance(year, (str,type(None))), 'year must be of type str'
  
    if not any([year, start_date, end_date]):
        Exception('Must specify either a year, a start_date, or start_date & end_date')
    
    if year:
        try:
            start_year = datetime.datetime.strptime(year, '%Y')
        except ValueError:
            raise ValueError("Incorrect year format, should be YYYY")
        else:            
            next_year = datetime.datetime.strptime(f'{int(year)+1}', '%Y')
            end_year = next_year - datetime.timedelta(days=1)
    if start_date:        
        start_date = _validate_date(start_date)   
    if end_date:
        end_date = _validate_date(end_date)
        if not start_date:
            raise Exception('start_date must be speficied with end_date')         
            
    
    nc = request_netCDF(station_number, data_type)    

       
     
    
    time_all = nc.variables['waveTime'][:].compressed().astype('datetime64[s]')
    time_steps = len(time_all)
    
    time_variables={}
    metadata={}
    
    allVariables = [var for var in nc.variables]
    allVariables.remove('waveTime')
    
    twoDimensionalVars = [ 'waveEnergyDensity', 'waveMeanDirection', 
                           'waveA1Value', 'waveB1Value', 'waveA2Value', 
                           'waveB2Value', 'waveCheckFactor', 'waveSpread', 
                           'waveM2Value', 'waveN2Value']
    
    if not include2DVars:
        for var in twoDimensionalVars:
            allVariables.remove(var)
    
    for var in allVariables:
        print(var)
        #if var == 'waveEnergyDensity':
        #   import ipdb; ipdb.set_trace()
        variable = nc.variables[var][:].compressed()
        if len(variable) == time_steps:
            time_variables[var] = variable
        else:
            metadata[var] = variable
     
    data = pd.DataFrame(time_variables, index=time_all)   
     
    if start_date:
        start_string = start_date.strftime('%Y-%m-%d')

        if end_date:
            end_string = end_date.strftime('%Y-%m-%d')
            data = data[start_string:end_string]     
        else:
            data = data[start_string:end_string]        
                        
    elif year: 
        start_string = start_year.strftime('%Y-%m-%d')
        end_string = end_year.strftime('%Y-%m-%d')
        data = data[start_string:end_string]

    buoy_name = nc.variables['metaStationName'][:].compressed().tostring()           
    data.name = buoy_name
    return data, metadata







def request_realtime(station_number, start_date='', end_date='', year_date='',
                     data_type='Historic'):
    """
    Requests CDIP data by wave buoy data file (from http://cdip.ucsd.edu/).
    

    Parameters
    ------------
    station_number: string
        Station number of CDIP wave buoy
    start_date: string 
        Start date in MM/DD/YYYY, e.g. '04/01/2012'
    end_date: string 
        End date in MM/DD/YYYY, e.g. '04/30/2012'
    year_date: string 
        Year date, e.g. '2001'
    data_type: string 
        'Realtime' or 'Historic', default = 'Historic'
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to the data 
        signal, for it includes: Hs, Tp, and Dp
        
    """
    assert isinstance(stn, str), f'station_number must be of type str'
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
        assert data_type == 'Historic', ('If specifying yeardate, data_type must be "Historic"')

    if isinstance(data_type, str):        
        assert data_type == 'Realtime' or data_type == 'Historic', ('Data type must be either Historic or Realtime'
        f' got: {data_type}')
    
    if len(start_date) == 0 and data_type == 'Historic': 
        assert yeardate !='', ('If not setting a date range, you must set a year')
    
    if len(yeardate) == 0 and data_type == 'Historic':
        assert start_date !='', ('If not setting a year, you must set a date')
        
    # Access Historic or Realtime data from CDIP Thredds
    if data_type == 'Historic':
        # CDIP Archived Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    
    elif data_type == 'Realtime':
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
    
    # Open Remote Dataset from CDIP THREDDS Server
    nc = netCDF4.Dataset(data_url)

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
        if end_date !='':
            unixend = _getUnixTimestamp(end_date,"%m/%d/%Y")
            assert unixend > unixstart, ('end_date must be later than start_date')
            future = _find_nearest(ncTime, unixend)  # Find the closest unix timestamp
            futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date
        else:
            futureIndex = -1   
        
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