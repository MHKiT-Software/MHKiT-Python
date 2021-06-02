from datetime import timezone
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
    else:
        dt = dt.replace(tzinfo=timezone.utc)
        
    return dt


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


def _dates_to_timestamp(nc, start_date=None, end_date=None):
    '''
    Returns timestamps from dates. 
    
    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type   
    start_date: string 
        Start date in YYYY-MM-DD, e.g. '2012-04-01'
    end_date: string 
        End date in YYYY-MM-DD, e.g. '2012-04-30'        
        
    Returns
    -------
    start_stamp: float
         seconds since the Epoch to start_date    
    end_stamp: float
         seconds since the Epoch to end_date
    '''
    assert isinstance(start_date, (str, type(None))), ('start_date' /
        'must be of type str')
    assert isinstance(end_date, (str, type(None))), ('end_date must be' / 
        'of type str')
        
    time_all = nc.variables['waveTime'][:].compressed()
    time_range_all = [datetime.datetime.fromtimestamp(time_all[0]).replace(tzinfo=timezone.utc), 
                      datetime.datetime.fromtimestamp(time_all[-1]).replace(tzinfo=timezone.utc)]
    
    if start_date:        
        start_datetime = _validate_date(start_date)   
    if end_date:
        end_datetime = _validate_date(end_date)   
        if start_datetime > end_datetime:
            raise Exception(f'start_date ({start_datetime}) must be'+
                f'before end_date ({end_datetime})')
        elif start_datetime == end_datetime:
            raise Exception(f'start_date ({start_datetime}) cannot be'+
                f'the same as end_date ({end_datetime})')
    
    if start_date:
        if start_datetime > time_range_all[0] and start_datetime < time_range_all[1]:
            start_stamp = start_datetime.replace(tzinfo=timezone.utc).timestamp()
        else:
            print(f'WARNING: Provided start_date ({start_datetime}) is ' 
            f'not in the returned data range {time_range_all} \n' 
            f'Setting start_date to the earliest date in range '
            f'{time_range_all[0]}')
            start_stamp = pd.to_datetime(time_range_all[0]).replace(tzinfo=timezone.utc).timestamp()         
    
    if end_date:
        if end_datetime > time_range_all[0] and end_datetime < time_range_all[1]:
            end_stamp = end_datetime.replace(tzinfo=timezone.utc).timestamp()
        else:
            print(f'WARNING: Provided end_date ({end_datetime}) is ' 
            f'not in the returned data range {time_range_all} \n' 
            f'Setting end_date to the latest date in range '
            f'{time_range_all[1]}')
            end_stamp = pd.to_datetime(time_range_all[1]).replace(tzinfo=timezone.utc).timestamp()        
    
    
    if start_date and not end_date:
        end_stamp = pd.to_datetime(time_range_all[1]).replace(tzinfo=timezone.utc).timestamp()            

    elif end_date and not start_date:
        start_stamp = pd.to_datetime(time_range_all[0]).replace(tzinfo=timezone.utc).timestamp()
        
    if not start_date:
        start_stamp = pd.to_datetime(time_range_all[0]).replace(tzinfo=timezone.utc).timestamp()
    if not end_date:
        end_stamp = pd.to_datetime(time_range_all[1]).replace(tzinfo=timezone.utc).timestamp()

    return start_stamp, end_stamp 

    
def request_netCDF(station_number, data_type):
    '''
    Returns historic or realtime data from CDIP THREDDS server
   
    Parameters
    ----------
    station_number: string
        CDIP station number of interest
    data_type: string
        'historic' or 'realtime'
   
    Returns
    -------
    nc: netCDF Object
        netCDF data for the given station number and data type
    '''
    assert isinstance(station_number, str), (f'station_number must be ' + 
                                              f'of type string. Got: {station_number}')
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

    
def request_parse_workflow(nc=None, station_number=None, parameters=None, 
               years=None, start_date=None, end_date=None, 
               data_type='historic', all_2D_variables=False):
    '''
    Parses a passed CDIP netCDF file or requests a station number 
    from http://cdip.ucsd.edu/) and parses. This function can return specific 
    parameters is passed. Years may be non-consecutive e.g. [2001, 2010].
    Time may be sliced by dates (start_date or end date in YYYY-MM-DD).
    data_type defaults to historic but may also be set to 'realtime'.
    By default 2D variables are not parsed if all 2D varaibles are needed. See
    the MHKiT CDiP example Jupyter notbook for information on available parameters. 
    
    
    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type. Can be the output of 
        request_netCDF   
    station_number: string
        Station number of CDIP wave buoy
    parameters: string or list of stings
        Parameters to return. If None will return all varaibles except
        2D-variables.        
    years: int or list of int
        Year date, e.g. 2001 or [2001, 2010]        
    start_date: string 
        Start date in YYYY-MM-DD, e.g. '2012-04-01'
    end_date: string 
        End date in YYYY-MM-DD, e.g. '2012-04-30'
    data_type: string
        Either 'historic' or 'realtime'   
    all_2D_variables: boolean
        Will return all 2D data. Enabling this will add significant 
        processing time. If all 2D variables are not needed it is
        recomended to pass 2D parameters of interest using the 
        'parameters' keyword and leave this set to False. Default False.
    
    Returns
    -------
    data: dictionary
        'vars1D': DataFrame
            1D variables indexed by time    
        'metadata': dictionary
            Anything not of length time
        'vars2D': dictionary of DataFrames, optional
            If 2D-vars are passed in the 'parameters key' or if run 
            with all_2D_variables=True, then this key will appear 
            with a dictonary of DataFrames of 2D variables.     
    '''
    assert isinstance(station_number, (str, type(None))), (f'station_number must be '+     
                                              'of type string')
    assert isinstance(parameters, (str, type(None), list)), ('parameters' /
        'must be of type str or list of strings')
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

  
    if not any([nc, station_number]):
        raise Exception('Must provide either a CDIP netCDF file or a station '+ 
            'number')
   
    if not nc:
        nc = request_netCDF(station_number, data_type)
    
    buoy_name = nc.variables['metaStationName'][:].compressed().tobytes().decode("utf-8")
    
    
    multiyear=False
    if years:
        if isinstance(years,int):
            start_date = f'{years}-01-01'
            end_date = f'{years}-12-31'            
        elif isinstance(years,list):
            if len(years)==1:
                start_date = f'{years[0]}-01-01'
                end_date = f'{years[0]}-12-31' 
            else:
                multiyear=True

    if not multiyear:
        data = get_netcdf_variables(nc, 
                       start_date=start_date, end_date=end_date, 
                       parameters=parameters, 
                       all_2D_variables=all_2D_variables)  

    elif multiyear:
        data={'data':{},'metadata':{}}
        multiyear_data={}
        multiyear_data_2D={}
        for year in years: 
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'   
            
            year_data = get_netcdf_variables(nc, 
                       start_date=start_date, end_date=end_date,  
                       parameters=parameters, 
                       all_2D_variables=all_2D_variables) 
            multiyear_data[year] = year_data['data']
          
        for data_key in year_data['data'].keys():
            if data_key.endswith('2D'):
                data['data'][data_key]={}
                for data_key2D in year_data['data'][data_key].keys():
                    data_list=[]
                    for year in years:    
                        data2D = multiyear_data[year][data_key][data_key2D]
                        data_list.append(data2D)
                    data['data'][data_key][data_key2D]=pd.concat(data_list)
            else:                
                data_list = [multiyear_data[year][data_key] for year in years]
                data['data'][data_key] = pd.concat(data_list)


                

        data['metadata'] = year_data['metadata']
    data['metadata']['name'] = buoy_name    

    return data
    
    
def get_netcdf_variables(nc, start_date=None, end_date=None, 
                         parameters=None, all_2D_variables=False):
    '''
    Iterates over and extracts variables from CDIP bouy data. See
    the MHKiT CDiP example Jupyter notbook for information on available 
    parameters. 
    
    
    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type
    start_stamp: float
        Data of interest start in seconds since epoch
    end_stamp: float
        Data of interest end in seconds since epoch  
    parameters: string or list of stings
        Parameters to return. If None will return all varaibles except
        2D-variables. Default None.
    all_2D_variables: boolean
        Will return all 2D data. Enabling this will add significant 
        processing time. If all 2D variables are not needed it is
        recomended to pass 2D parameters of interest using the 
        'parameters' keyword and leave this set to False. Default False.

    Returns
    -------
    results: dictionary
        'vars1D': DataFrame
            1D variables indexed by time    
        'metadata': dictionary
            Anything not of length time
        'vars2D': dictionary of DataFrames, optional
            If 2D-vars are passed in the 'parameters key' or if run 
            with all_2D_variables=True, then this key will appear 
            with a dictonary of DataFrames of 2D variables.
    '''
    
    assert isinstance(nc, netCDF4.Dataset), 'nc must be netCDF4 dataset'
    assert isinstance(start_date, (str, type(None))), ('start_date' /
        'must be of type str')
    assert isinstance(end_date, (str, type(None))), ('end_date must be' / 
        'of type str')
    assert isinstance(parameters, (str, type(None), list)), ('parameters' /
        'must be of type str or list of strings')        
    assert isinstance(all_2D_variables, bool), ('all_2D_variables'/
        'must be a boolean')

    if parameters:
        if isinstance(parameters,str):
            parameters = [parameters]        
        assert all([isinstance(param , str) for param in parameters]), ('All'/
           'elements of parameters must be strings')


    buoy_name = nc.variables['metaStationName'][:].compressed().tobytes().decode("utf-8")           
    allVariables = [var for var in nc.variables]
    
    include_2D_variables=False
    twoDimensionalVars = [ 'waveEnergyDensity', 'waveMeanDirection', 
                           'waveA1Value', 'waveB1Value', 'waveA2Value', 
                           'waveB2Value', 'waveCheckFactor', 'waveSpread', 
                           'waveM2Value', 'waveN2Value']  
    
    if parameters:
        params = set(parameters)
        include_params = params.intersection(set(allVariables))            
        if params != include_params:
           not_found = params.difference(include_params)
           print(f'WARNING: {not_found} was not found in data.\n' \
                 f'Possible parameters are:\n {allVariables}')
                
        include_params_2D = include_params.intersection(
                                set(twoDimensionalVars))                
        include_params = include_params.difference(include_params_2D)
        
        if include_params_2D:
            include_2D_variables=True
            include_params.add('waveFrequency')
            include_2D_vars = sorted(include_params_2D)
        
        include_vars = sorted(include_params)
            
    else:
        include_vars = allVariables
        
        for var in twoDimensionalVars:
            include_vars.remove(var)
            
        if all_2D_variables:
            include_2D_variables=True
            include_2D_vars = twoDimensionalVars                 

    
    start_stamp, end_stamp =_dates_to_timestamp(nc, start_date=start_date, 
                                                 end_date=end_date)
    
    variables_by_type={}       
    prefixs = ['wave', 'sst', 'gps', 'dwr', 'meta']
    remainingVariables = set(include_vars)
    for prefix in prefixs:
        variables_by_type[prefix] = [var for var in include_vars 
            if var.startswith(prefix)]
        remainingVariables -= set(variables_by_type[prefix])
        if not variables_by_type[prefix]:
            del variables_by_type[prefix]

    results={'data':{}, 'metadata':{}}
    for prefix in variables_by_type:
        var_results={}
        time_variables={}
        metadata={}
        
        if prefix != 'meta':
            prefixTime = nc.variables[f'{prefix}Time'][:]
            
            masked_time = np.ma.masked_outside(prefixTime, start_stamp,
            end_stamp)
            mask = masked_time.mask                               
            var_time = masked_time.compressed() 
            N_time = masked_time.size
        else:
            N_time= np.nan
    
        for var in variables_by_type[prefix]:   
            variable = np.ma.filled(nc.variables[var])
            if variable.size == N_time:              
                variable = np.ma.masked_array(variable, mask).astype(float)
                time_variables[var] = variable.compressed()
            else:
                metadata[var] = nc.variables[var][:].compressed()

        time_slice = pd.to_datetime(var_time, unit='s')
        data = pd.DataFrame(time_variables, index=time_slice)        
         
        if prefix != 'meta':      
            results['data'][prefix] = data
            results['data'][prefix].name = buoy_name
        results['metadata'][prefix] = metadata
            
        if (prefix == 'wave') and (include_2D_variables):
            
            print('Processing 2D Variables:')
            vars2D={}
            columns=metadata['waveFrequency']
            N_time= len(time_slice)
            N_frequency = len(columns)
            try:
                l = len(mask)
            except:
                mask = np.array([False] * N_time)
                
            mask2D= np.tile(mask, (len(columns),1)).T
            for var in include_2D_vars:
                variable2D = nc.variables[var][:].data
                variable2D = np.ma.masked_array(variable2D, mask2D)
                variable2D = variable2D.compressed().reshape(N_time, N_frequency)            
                variable = pd.DataFrame(variable2D,index=time_slice,
                                        columns=columns)
                vars2D[var] = variable
            results['data']['wave2D'] = vars2D
    results['metadata']['name'] = buoy_name
        
    return results
