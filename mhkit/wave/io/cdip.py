import pandas as pd
import numpy as np
import datetime
import netCDF4
import time


def _validate_date(date_text):
    '''
    Checks date format to ensure YYYY-MM-DD format and return date in
    datetime format.
    
    Parameters
    ----------
    date_text: string
        Date string format to check
        
    Returns
    -------
    dt: datetime
    '''  
    assert isinstance(date_text, str), (f'date_text must be' / 
                                              'of type string')
    try:
        dt = datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    return dt


def request_netCDF(station_number, data_type):
    '''
    Returns historic or realtime data from CDIP THREDDS server
   
    Parameters
    ----------
    station_number: string
        CDIP station number of interest
    data_type: string
        Either 'historic' or 'realtime'
   
    Returns
    -------
    nc: netCDF Object
        netCDF data for the given station number and data type
    '''
    assert isinstance(station_number, str), (f'station_number must be' / 
                                              'of type string')
    assert isinstance(data_type, str), (f'data_type must be' / 
                                              'of type string')
    assert data_type in ['historic', 'realtime'], ('data_type must be'\
        f' "historic" or "realtime". Got: {data_type}')                                              
    if data_type == 'historic':
        cdip_archive= 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive'
        data_url =  f'{cdip_archive}/{station_number}p1/{station_number}p1_historic.nc'
    elif data_type == 'realtime':
        cdip_realtime = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime'
        data_url = f'{cdip_realtime}/{station_number}p1_rt.nc'
    
    nc = netCDF4.Dataset(data_url)
    return nc


def _start_and_end_of_year(year):
    '''
    Returns a datetime start and end for a given year
    
    Parameters
    ----------
    year: int
        Year to get start and end dates
        
    Returns
    -------
    start_year: datetime object
        start of the year
    end_year: datetime object
        end of the year
    
    '''
    assert isinstance(year, (type(None),int,list)), 'year must be of type int'
    
    try:
        year = str(year)
        start_year = datetime.datetime.strptime(year, '%Y')
    except ValueError:
        raise ValueError("Incorrect years format, should be YYYY")
    else:            
        next_year = datetime.datetime.strptime(f'{int(year)+1}', '%Y')
        end_year = next_year - datetime.timedelta(days=1)
    return start_year, end_year


def get_netcdf_variables(nc, start_stamp=None, end_stamp=None, 
                             include_2D_variables=False):
    '''
    Interates over and extracts variables from CDIP Bouy data
    
    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type
    start_stamp: float
        Data of interest start in seconds since epoch
    end_stamp: float
        Data of interest end in seconds since epoch   
    include2DVars: boolean
        Will return all 2D data. Enabling this will add significant 
        processing time. It is reccomened to call `request_netCDF` 
        function directly and process 2D data of interest. 
    Returns
    -------
    time_variable: dictionary
        1D variables indexed by time    
    metadata: dictionary
        Anything not of length time            
    '''
    
    assert isinstance(nc, netCDF4.Dataset), 'nc must be netCDF4 dataset'
    assert isinstance(start_stamp, (float,int,type(None))), ('start_stamp'/
        'must be float or None')
    assert isinstance(start_stamp, (float,int, type(None))), ('end_stamp'/
        'must be float or None')
    assert isinstance(include_2D_variables, bool), ('include_2D_variables'/
        'must be a boolean')
    time_variables={}
    metadata={}
       
    time_all = nc.variables['waveTime'][:].compressed()
    time_range_all = [time_all[0].astype('datetime64[s]'), 
                  time_all[-1].astype('datetime64[s]')]

    if not start_stamp:
        start_stamp = pd.to_datetime(time_range_all[0]).timestamp() 
    if not end_stamp:
        end_stamp = pd.to_datetime(time_range_all[1]).timestamp() 
    
    
    masked_time = np.ma.masked_outside(nc.variables['waveTime'][:], 
                               start_stamp, end_stamp)
    mask = masked_time.mask                               
    time_variables['waveTime'] = masked_time.compressed()
    
    
    allVariables = [var for var in nc.variables]
    allVariables.remove('waveTime')
        
    twoDimensionalVars = [ 'waveEnergyDensity', 'waveMeanDirection', 
                           'waveA1Value', 'waveB1Value', 'waveA2Value', 
                           'waveB2Value', 'waveCheckFactor', 'waveSpread', 
                           'waveM2Value', 'waveN2Value']
    
    
    for var in twoDimensionalVars:
        allVariables.remove(var)
    
    for var in allVariables:      
        variable = nc.variables[var][:].compressed()
        
        if variable.size == masked_time.size:              
            variable = np.ma.masked_array(variable, mask)
            time_variables[var] = variable.compressed()
        else:
            metadata[var] = nc.variables[var][:].compressed()
            
    if include_2D_variables:
        vars2D={}
        columns=metadata['waveFrequency']
        for var in twoDimensionalVars:
            data = nc.variables[var][:].data
            variable = pd.DataFrame(data,index=time_variables['waveTime'],
                                    columns=columns)
            vars2D[var] = variable
            import ipdb;ipdb.set_trace()
    return time_variables, metadata


def request_data(station_number, years=None, start_date=None, 
                     end_date=None, data_type='historic', 
                     include_2D_variables=False):
    '''
    Requests CDIP data by wave buoy data file (from http://cdip.ucsd.edu/).
    
    Parameters
    ----------
    station_number: string
        Station number of CDIP wave buoy
    years: int or list of int
        Year date, e.g. 2001 or [2009, 2010]        
    start_date: string 
        Start date in MM-DD-YYYY, e.g. '04-01-2012'
    end_date: string 
        End date in MM/DD/YYYY, e.g. '04-30-2012'
    data_type: string
        Either 'historic' or 'realtime'   
    include2DVars: boolean
        Will return all 2D data. Enabling this will add significant 
        processing time. It is reccomened to call `request_netCDF` 
        function directly and process 2D variable of interest.
    
    Returns
    -------
    data: DataFrame 
        Data indexed by datetime with columns named according to the data 
        signal, for it includes: Hs, Tp, and Dp       
    '''
    assert isinstance(station_number, str), (f'station_number must be' / 
                                              'of type string')
    assert isinstance(start_date, (str, type(None))), ('start_date' /
        'must be of type str')
    assert isinstance(end_date, (str, type(None))), ('end_date must be' / 
        'of type str')
    assert isinstance(years, (type(None),int,list)), ('years must be of'/
        'type int or list of ints')
    assert isinstance(data_type, str), (f'data_type must be' / 
                                              'of type string')        
    assert data_type in ['historic', 'realtime'], 'data_type must be'\
        f' "historic" or "realtime". Got: {data_type}'
  
    if not any([years, start_date, end_date]):
        raise Exception('Must specify either a year, a start_date,'
                        'a end date or start_date & end_date')

    if start_date:        
        start_date = _validate_date(start_date)   
    if end_date:
        end_date = _validate_date(end_date)   
        if start_date > end_date:
            raise Exception(f'start_date ({start_date}) must be'/
                f'before end_date ({end_date})')
        elif start_date == end_date:
            raise Exception(f'start_date ({start_date}) cannot be'/
                f'the same as end_date ({end_date})')
    
    multiyear=False
    if years:
        if isinstance(years,int):
            start_date, end_date = _start_and_end_of_year(years)
        elif isinstance(years,list):
            if len(years)==1:
                start_date, end_date = _start_and_end_of_year(years[0])
            else:
                multiyear=True
    nc = request_netCDF(station_number, data_type)

    time_all = nc.variables['waveTime'][:].compressed()
    time_range_all = [time_all[0].astype('datetime64[s]'), 
                  time_all[-1].astype('datetime64[s]')]

    if start_date:
        if start_date > time_range_all[0] and start_date < time_range_all[1]:
            start_stamp = start_date.timestamp()
        else:
            print(f'WARNING: Provided start_date ({start_date}) is ' 
            f'not in the returned data range {time_range_all} \n' 
            f'Setting start_date to the earliest date in range '
            f'{time_range_all[0]}')
            start_stamp = pd.to_datetime(time_range_all[0]).timestamp()         
    
    if end_date:
        if end_date > time_range_all[0] and end_date < time_range_all[1]:
            end_stamp = end_date.timestamp()
        else:
            print(f'WARNING: Provided end_date ({end_date}) is ' 
            f'not in the returned data range {time_range_all} \n' 
            f'Setting end_date to the latest date in range '
            f'{time_range_all[1]}')
            end_stamp = pd.to_datetime(time_range_all[1]).timestamp()        
    
    
    if start_date and not end_date:
        end_stamp = pd.to_datetime(time_range_all[1]).timestamp()            

    elif end_date and not start_date:
        start_stamp = pd.to_datetime(time_range_all[0]).timestamp()
      
    if not multiyear:
        time_variables, metadata = get_netcdf_variables(nc, 
                       start_stamp=start_stamp, end_stamp=end_stamp, 
                       include_2D_variables=include_2D_variables)  
        
        time_slice = pd.to_datetime(time_variables['waveTime'][:], unit='s')
        data = pd.DataFrame(time_variables, index=time_slice)                         
    elif multiyear:
        mYear={}
        multiyear_metadata={}
        for year in years: 
            start_year, end_year = _start_and_end_of_year(year) 
            start_stamp = start_year.timestamp()
            end_stamp = end_year.timestamp()
            
            time_variables, metadata = get_netcdf_variables(nc, 
                       start_stamp=start_stamp, end_stamp=end_stamp,  
                       include_2D_variables=include_2D_variables) 
                       
            time_slice = pd.to_datetime(time_variables['waveTime'][:], unit='s')
            data = pd.DataFrame(time_variables, index=time_slice)                        
            mYear[year] = data
            multiyear_metadata[year] = metadata
        data = pd.concat([v for k,v in mYear.items()])

    buoy_name = nc.variables['metaStationName'][:].compressed().tostring()
    data.name = buoy_name
    
    return data, metadata
