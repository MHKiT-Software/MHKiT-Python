from io import BytesIO
import pandas as pd
import numpy as np
import requests
import bs4
import zlib


def ndbc_read_file(file_name, missing_values=['MM',9999,999,99]):
    """
    Reads a NDBC wave buoy data file (from https://www.ndbc.noaa.gov).
    
    Realtime and historical data files can be loaded with this function.  
    
    Note: With realtime data, missing data is denoted by "MM".  With historical 
    data, missing data is denoted using a variable number of 
    # 9's, depending on the data type (for example: 9999.0 999.0 99.0).
    'N/A' is automatically converted to missing data.
    
    Data values are converted to float/int when possible. Column names are 
    also converted to float/int when possible (this is useful when column 
    names are frequency).
    
    Parameters
    ------------
    file_name : string
        Name of NDBC wave buoy data file
    
    missing_value : list of values
        List of values that denote missing data    
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to header row 
        
    metadata: dict or None
        Dictionary with {column name: units} key value pairs when the NDBC file  
        contains unit information, otherwise None is returned
    """
    assert isinstance(file_name, str), 'file_name must be of type str'
    assert isinstance(missing_values, list), 'missing_values must be of type list'
    
    # Open file and get header rows
    f = open(file_name,"r")
    header = f.readline().rstrip().split()  # read potential headers
    units = f.readline().rstrip().split()   # read potential units
    f.close()
    
    # If first line is commented, remove comment sign #
    if header[0].startswith("#"):
        header[0] = header[0][1:]
        header_commented = True
    else:
        header_commented = False
        
    # If second line is commented, indicate that units exist
    if units[0].startswith("#"):
        units_exist = True
    else:
        units_exist = False
    
    # Check if the time stamp contains minutes, and create list of column names 
    # to parse for date
    if header[4] == 'mm':
        parse_vals = header[0:5]
        date_format = '%Y %m %d %H %M'
        units = units[5:]   #remove date columns from units
    else:
        parse_vals = header[0:4]
        date_format = '%Y %m %d %H'
        units = units[4:]   #remove date columns from units
    
    # If first line is commented, manually feed in column names
    if header_commented:
        data = pd.read_csv(file_name, sep='\s+', header=None, names = header,
                           comment = "#", parse_dates=[parse_vals]) 
    # If first line is not commented, then the first row can be used as header                        
    else:
        data = pd.read_csv(file_name, sep='\s+', header=0,
                           comment = "#", parse_dates=[parse_vals])
                             
    # Convert index to datetime
    date_column = "_".join(parse_vals)
    data['Time'] = pd.to_datetime(data[date_column], format=date_format)
    data.index = data['Time'].values
    # Remove date columns
    del data[date_column]
    del data['Time']
    
    # If there was a row of units, convert to dictionary
    if units_exist:
        metadata = {column:unit for column,unit in zip(data.columns,units)}
    else:
        metadata = None

    # Convert columns to numeric data if possible, otherwise leave as string
    for column in data:
        data[column] = pd.to_numeric(data[column], errors='ignore')
        
    # Convert column names to float if possible (handles frequency headers)
    # if there is non-numeric name, just leave all as strings.
    try:
        data.columns = [float(column) for column in data.columns]
    except:
        data.columns = data.columns
    
    # Replace indicated missing values with nan
    data.replace(missing_values, np.nan, inplace=True)
    
    return data, metadata


def ndbc_available_data(parameter,
                        buoy_number=None, 
                        proxy=None):  
    '''
    For a given parameter this will return a DataFrame of years, 
    station IDs and file names that contain that parameter data.              
    
    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'
        'stdmet':   'Standard Meteorological Current Year Historical Data'
    buoy_number: string (optional)
        Buoy Number.  5-character alpha-numeric station identifier        
    proxy: string (optional)
        proxy url
        
    Returns
    -------
    available_data: DataFrame
        DataFrame with station ID, years, and NDBC file names. 
    '''
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(buoy_number, (str, type(None), list)), ('If ' 
        'specified the buoy number must be a string or list of strings')
    assert isinstance(proxy , (str, type(None))), 'If specified proxy must be a string'
    supported =_ndbc_supported_params(parameter)
    if isinstance(buoy_number, str):        
        assert len(buoy_number) == 5, ('Buoy must be 5-character'
        f'alpha-numeric station identifier got: {buoy_number}')
    elif isinstance(buoy_number, list):
        for buoy in buoy_number:
            assert len(buoy) == 5, ('Each buoy must be a 5-character'
            f'alpha-numeric station identifier got: {buoy}')
    ndbc_data = f'https://www.ndbc.noaa.gov/data/historical/{parameter}/'
    if proxy == None:
        response = requests.get(ndbc_data)
    else:
        response = requests.get(ndbc_data, proxies=proxy)
    
    status = response.status_code 
    if status != 200:
        msg=f"request.get{ndbc_data} failed by returning code of {status}"
        raise Exception(msg)            


    filenames = pd.read_html(response.text)[0].Name.dropna()    
    buoys = _ndbc_parse_filenames(parameter, filenames)

    available_data = buoys.copy(deep=True)
    
    if isinstance(buoy_number, str):        
        available_data = buoys[buoys.id==buoy_number]
    elif isinstance(buoy_number, list):
        available_data = buoys[buoys.id==buoy_number[0]]
        for i in range(1, len(buoy_number)):
            data = buoys[buoys.id==buoy_number[i]]
            available_data = available_data.append(data)                  
        
    return available_data
    
def _ndbc_parse_filenames(parameter, filenames):  
    '''
    Takes a list of available filenames as a series from NDBC then 
    parses out the station ID and year from the file name.
    
    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'
        'stdmet':   'Standard Meteorological Current Year Historical Data'
    filenames: Series
        List of compressed file names from NDBC
     
    Returns
    -------
    buoys: DataFrame
        DataFrame with keys=['id','year','file_name']    
    '''  
    assert isinstance(filenames, pd.Series), 'filenames must be of type pd.Series' 
    assert isinstance(parameter, str), 'parameter must be a string'
    supported =_ndbc_supported_params(parameter)
    
    file_seps = {
                'swden' : 'w',
                'stdmet' : 'h'
               }
    file_sep= file_seps[parameter]
    
    filenames = filenames[filenames.str.contains('.txt.gz')]
    buoy_id_year_str = filenames.str.split('.', expand=True)[0]
    buoy_id_year = buoy_id_year_str.str.split(file_sep, n=1,expand=True)
    buoys = buoy_id_year.rename(columns={0:'id', 1:'year'})
    
    expected_station_id_length = 5
    buoys = buoys[buoys.id.str.len() == expected_station_id_length]
    buoys['filename'] = filenames  
    return buoys    
    
    
def ndbc_request_data(parameter, filenames, proxy=None):
    '''
    Requests data by filenames and returns a dictionary of DataFrames 
    for each filename passed. The Dictionary is indexed by buoy and year.
        
    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'
        'stdmet':   'Standard Meteorological Current Year Historical Data'
    filenames: DataFrame
	    Data filenames on https://www.ndbc.noaa.gov/data/historical/{parameter}/
	proxy: string
	    Proxy URL   
        
    Returns
    -------
    ndbc_data: dict
        Dictionary of DataFrames indexed by buoy and year.
    '''
    assert isinstance(filenames, pd.Series), 'filenames must be of type pd.Series' 
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(proxy, (str, type(None))), 'If specified proxy must be a string'    
    supported =_ndbc_supported_params(parameter)

    buoy_data = _ndbc_parse_filenames(parameter, filenames)
        
    parameter_url = f'https://www.ndbc.noaa.gov/data/historical/{parameter}'
    ndbc_data = {}    
    
    for year, filename in zip(buoy_data.year, buoy_data.filename):
        file_url = f'{parameter_url}/{filename}'
        response =  requests.get(file_url)
        data = zlib.decompress(response.content, 16+zlib.MAX_WBITS)
        df = pd.read_csv(BytesIO(data), sep='\s+', low_memory=False)
        ndbc_data[year] = df

    return ndbc_data

    

def _ndbc_supported_params(parameter):
    '''
    There is a significant number of datasets provided by NDBC. There is
    specific data processing required for each type. Therefore this 
    function is thrown for any data type not currently covered.
    
    Available Data: https://www.ndbc.noaa.gov/data/ 
    https://www.ndbc.noaa.gov/historical_data.shtml
    Decription of Measurements: https://www.ndbc.noaa.gov/measdes.shtml
    Changes made to historical data: https://www.ndbc.noaa.gov/mods.shtml  
    
    Parameters
    ----------
    None
    
    Returns
    -------
    msg: string
        string indicating what is supported and how to request or add
        new functionality    
    '''
    assert isinstance(parameter, str), 'parameter must be a string'
    supported=True
    supported_params = [
                       'swden',
                       'stdmet'
                      ]
    param = [param for param in supported_params if param == parameter]

    if not param:      
        supported=False
        msg = ["Currently parameters 'swden' and 'stdmet'  are supported. \n"+
               "If you would like to see more data types please \n"+
               " open an issue or submit a Pull Request on GitHub"]
        raise Exception(msg[0])
    

    
    # Historical
    parameters = {
    'adcp'	:	'Acoustic Doppler Current Profiler Current Year Historical Data'	,
    'adcp2'	:	'Acoustic Doppler Current Profiler Current Year Historical Data'	,
    'cwind'	:	'Continuous Winds Current Year Historical Data'	,
    'dart'	:	'Water Column Height (DART) Current Year Historical Data'	,
    'mmbcur'	:	'	',
    'ocean'	:	'Oceanographic Current Year Historical Data'	,
    'rain'	:	'Hourly Rain Current Year Historical Data'	,
    'rain10'	:	'10-Minute Rain Current Year Historical Data'	,
    'rain24'	:	'24-Hour Rain Current Year Historical Data'	,
    'srad'	:	'Solar Radiation Current Year Historical Data'	,
    'stdmet'	:	'Standard Meteorological Current Year Historical Data'	,
    'supl'	:	'Supplemental Measurements Current Year Historical Data'	,
    'swden'	:	'Raw Spectral Wave Current Year Historical Data'	,
    'swdir'	:	'Spectral Wave Current Year Historical Data (alpha1)'	,
    'swdir2'	:	'Spectral Wave Current Year Historical Data (alpha2)'	,
    'swr1'	:	'Spectral Wave Current Year Historical Data (r1)'	,
    'swr2'	:	'Spectral Wave Current Year Historical Data (r2)'	,
    'wlevel'	:	'Tide Current Year Historical Data'	,
    }

    return supported	
    