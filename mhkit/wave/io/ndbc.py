from collections import OrderedDict as _OrderedDict
from collections import defaultdict as _defaultdict
from io import BytesIO
import pandas as pd
import numpy as np
import requests
import zlib
import pandas.errors



def read_file(file_name, missing_values=['MM',9999,999,99]):
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
    file_name: string
        Name of NDBC wave buoy data file
    
    missing_value: list of values
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


def available_data(parameter,
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
        
    proxy: dict
	    Proxy dict passed to python requests, 
        (e.g. proxy_dict= {"http": 'http:wwwproxy.yourProxy:80/'})  
        
    Returns
    -------
    available_data: DataFrame
        DataFrame with station ID, years, and NDBC file names. 
    '''
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(buoy_number, (str, type(None), list)), ('If ' 
        'specified the buoy number must be a string or list of strings')
    assert isinstance(proxy, (dict, type(None))), 'If specified proxy must be a dict' 
    supported = _supported_params(parameter)
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
    buoys = _parse_filenames(parameter, filenames)

    available_data = buoys.copy(deep=True)
    
	# Set year to numeric (makes year key non-unique)
    available_data['year']=available_data.year.str.strip('b')
    available_data['year']=pd.to_numeric(available_data.year.str.strip('_old'))

    if isinstance(buoy_number, str):        
        available_data = available_data[available_data.id==buoy_number]
    elif isinstance(buoy_number, list):
        available_data = available_data[available_data.id==buoy_number[0]]
        for i in range(1, len(buoy_number)):
            data = available_data[available_data.id==buoy_number[i]]
            available_data = available_data.append(data)                  
        
    return available_data
    
def _parse_filenames(parameter, filenames):  
    '''
    Takes a list of available filenames as a series from NDBC then 
    parses out the station ID and year from the file name.
    
    Parameters
    ----------
    parameter: string
        'swden'	:    'Raw Spectral Wave Current Year Historical Data'
        
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
    supported = _supported_params(parameter)
    
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
        
def request_data(parameter, filenames, proxy=None):
    '''
    Requests data by filenames and returns a dictionary of DataFrames 
    for each filename passed. If filenames for a sigle buoy are passed 
    then the yearly DataFrames in the returned dictionary (ndbc_data) are 
    indexed by year (e.g. ndbc_data['2014']). If multiple buoy ids are 
    passed then the returned dictionary is indexed by buoy id and year 
    (e.g. ndbc_data['46022']['2014']).
        
    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'       
        'stdmet':   'Standard Meteorological Current Year Historical Data'
        
    filenames: pandas Series or DataFrame
	    Data filenames on https://www.ndbc.noaa.gov/data/historical/{parameter}/
    
    proxy: dict
	    Proxy dict passed to python requests, 
        (e.g. proxy_dict= {"http": 'http:wwwproxy.yourProxy:80/'})  
        
    Returns
    -------
    ndbc_data: dict
        Dictionary of DataFrames indexed by buoy and year.
    '''
    assert isinstance(filenames, (pd.Series,pd.DataFrame)), (
        'filenames must be of type pd.Series') 
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(proxy, (dict, type(None))), ('If specified proxy' 
      'must be a dict')
    
    supported =_supported_params(parameter)
    if isinstance(filenames,pd.DataFrame):
        filenames = pd.Series(filenames.squeeze())
    assert len(filenames)>0, "At least 1 filename must be passed"
       

    buoy_data = _parse_filenames(parameter, filenames)
    parameter_url = f'https://www.ndbc.noaa.gov/data/historical/{parameter}'
    ndbc_data = _defaultdict(dict)    
    
    for buoy_id in buoy_data['id'].unique():        
        buoy = buoy_data[buoy_data['id']== buoy_id]
        years = buoy.year
        filenames = buoy.filename
        for year, filename in zip(years, filenames):
            file_url = f'{parameter_url}/{filename}'
            if proxy == None:
                response = requests.get(file_url)
            else:
                response = requests.get(file_url, proxies=proxy)
            try: 
                data = zlib.decompress(response.content, 16+zlib.MAX_WBITS)
                df = pd.read_csv(BytesIO(data), sep='\s+', low_memory=False)                
            except zlib.error: 
                msg = (f'Issue decompressing the NDBC file {filename}'  
                       f'(id: {buoy_id}, year: {year}). Please request ' 
                       'the data again.')
                print(msg)
            except pandas.errors.EmptyDataError: 
                msg = (f'The NDBC buoy {buoy_id} for year {year} with ' 
                       f'filename {filename} is empty or missing '     
                        'data. Please omit this file from your data '   
                        'request in the future.')
                print(msg)
            else:
                ndbc_data[buoy_id][year] = df    
                             
    if len(ndbc_data) == 1:
        ndbc_data = ndbc_data[buoy_id]

    return ndbc_data

def to_datetime_index(parameter, ndbc_data):
    '''
    Converts the NDBC date and time information reported in separate
    columns into a DateTime index and removed the NDBC date & time 
    columns.        

    Parameters
    ----------
    parameter: string
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'
        'stdmet':   'Standard Meteorological Current Year Historical Data'
    
    ndbc_data: DataFrame
        NDBC data in dataframe with date and time columns to be converted

    Returns
    -------
	df_datetime: DataFrame
	    Dataframe with NDBC date columns removed, and datetime index
    '''

    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(ndbc_data, pd.DataFrame), 'ndbc_data must be of type pd.DataFrame'
    
    df_datetime = ndbc_data.copy(deep=True)
    df_datetime['date'], ndbc_date_cols = dates_to_datetime(parameter, 
                                                          df_datetime, 
                                                          return_date_cols=True)
    df_datetime = df_datetime.drop(ndbc_date_cols, axis=1)
    df_datetime = df_datetime.set_index('date')
    if parameter=='swden':
        df_datetime.columns = df_datetime.columns.astype(float)         
        
    return df_datetime

def dates_to_datetime(parameter, data, 
                      return_date_cols=False, 
                      return_as_dataframe=False):
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
         
    return_as_dataFrame: bool
        Results returned as a DataFrame (useful for MHKiT-MATLAB)         
        
    Returns
    -------
    date: Series
        Series with NDBC dates dropped and new ['date']
        column in DateTime format

    ndbc_date_cols: list (optional)
        List of the DataFrame columns headers for dates as provided by 
        NDBC
    '''
    assert isinstance(data, pd.DataFrame), 'filenames must be of type pd.DataFrame' 
    assert isinstance(parameter, str), 'parameter must be a string'
    assert isinstance(return_date_cols, bool), 'return_date_cols must be of type bool'
    supported = _supported_params(parameter)
      
    df = data.copy(deep=True)     
    cols = df.columns.values.tolist()
    
    try:
        minutes_loc  = cols.index('mm')
        minutes=True
    except:
        df['mm'] = np.zeros(len(df)).astype(int).astype(str)
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
        
    parse_columns = [year_string, 'MM', 'DD', 'hh', 'mm']
    df = _date_string_to_datetime(df, parse_columns, year_fmt)        
    date = df['date']        
    
    if row_0_is_units:
        date = pd.concat([pd.Series([np.nan]),date])               
    del df
    
    if return_as_dataframe:
        date = pd.DataFrame(date)    
    if return_date_cols:
        if minutes:
            ndbc_date_cols = [year_string, 'MM', 'DD', 'hh', 'mm']
        else:
            ndbc_date_cols = [year_string, 'MM', 'DD', 'hh']
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
        Dataframe with columns (e.g. ['YY', 'MM', 'DD', 'hh', 'mm'])
        
    columns: list 
        list of strings for the columns to consider   
        (e.g. ['YY', 'MM', 'DD', 'hh', 'mm'])
        
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

def parameter_units(parameter=''):
    '''
    Returns an ordered dictionary of NDBC parameters with unit values. 
    If no parameter is passed then an ordered dictionary of all NDBC 
    parameterz specified unites is returned. If a parameter is specified
    then only the units associated with that parameter are returned. 
    Note that many NDBC paramters report multiple measurements and in 
    that case the returned dictionary will contain the NDBC measurement
    name and associated unit for all the measurements associated with 
    the specified parameter. Optional parameter values are given below.
    All units are based on https://www.ndbc.noaa.gov/measdes.shtml.
    
    Parameters
    ----------
    parameter: string (optional)        
        'adcp':     'Acoustic Doppler Current Profiler Current Year Historical Data'        
        'cwind':    'Continuous Winds Current Year Historical Data'        
        'dart':     'Water Column Height (DART) Current Year Historical Data'        
        'derived2': 'Derived Met Values'        
        'ocean' :   'Oceanographic Current Year Historical Data'        
        'rain'	:	'Hourly Rain Current Year Historical Data'	        
        'rain10':	'10-Minute Rain Current Year Historical Data'	
        'rain24':	'24-Hour Rain Current Year Historical Data'	        
        'realtime2': 'Detailed Wave Summary (Realtime `.spec` data files only)'        
        'srad':     'Solar Radiation Current Year Historical Data'        
        'stdmet':   'Standard Meteorological Current Year Historical Data'        
        'supl':     'Supplemental Measurements Current Year Historical Data'        
        'swden'	:	'Raw Spectral Wave Current Year Historical Data'

    Returns
    -------
    units: dict
        Dictionary of parameter units
    '''

    assert isinstance(parameter, str), 'parameter must be a string'
    # supported = _supported_params(parameter)
    
    if parameter == 'adcp':
        units = {'DEP01'  : 'm',
                 'DIR01' : 'deg',
                 'SPD01' : 'cm/s',
                }           
    elif parameter == 'cwind':
        units = {'WDIR'  : 'degT',
                 'WSPD' : 'm/s',
                 'GDR'  : 'degT',
                 'GST'  : 'm/s',
                 'GTIME' : 'hhmm'
                }            
    elif parameter == 'dart':
        units = {'T'  : '-',
                 'HEIGHT' : 'm',
                }   
    elif parameter == 'derived2':
        units = {'CHILL'  : 'degC',
                 'HEAT'   : 'degC',
                 'ICE'    : 'cm/hr',
                 'WSPD10' : 'm/s',
                 'WSPD20' : 'm/s'
                }
    elif parameter == 'ocean':
        units = {'DEPTH'  : 'm',
                 'OTMP' : 'degC',
                 'COND'  : 'mS/cm',
                 'SAL'  : 'psu',
                 'O2%'  : '%',
                 'O2PPM' : 'ppm',
                 'CLCON' : 'ug/l',
                 'TURB' : 'FTU',
                 'PH' : '-',
                 'EH' : 'mv',
                }            
    elif parameter == 'rain':
        units = {'ACCUM'  : 'mm',
                }   
    elif parameter == 'rain10':
        units = {'RATE'  : 'mm/h',
                }   
    elif parameter == 'rain24':
        units = {'RATE'  : 'mm/h',
                 'PCT'  : '%',
                 'SDEV'  : '-',
                }          
    elif parameter == 'realtime2':
        units = {'WVHT': 'm',
                 'SwH' : 'm',
                 'SwP' : 'sec',
                 'WWH' : 'm',
                 'WWP' : 'sec',
                 'SwD' : '-',
                 'WWD' : 'degT',
                 'STEEPNESS' : '-',
                 'APD' : 'sec',
                 'MWD' : 'degT',
                }     
    elif parameter == 'srad':
        units = {'SRAD1'  : 'w/m2',
                 'SRAD2' : 'w/m2',
                 'SRAD3' : 'w/m2',
                }                   
    elif parameter == 'stdmet':
        units = {'WDIR' : 'degT',
                 'WSPD' : 'm/s',
                 'GST'  : 'm/s',
                 'WVHT' : 'm' ,
                 'DPD'  : 'sec',
                 'APD'  : 'sec',
                 'MWD'  : 'degT',
                 'PRES' : 'hPa',
                 'ATMP' : 'degC',
                 'WTMP' : 'degC',
                 'DEWP' : 'degC',
                 'VIS'  : 'nmi',
                 'PTDY' : 'hPa',
                 'TIDE' : 'ft'}
    elif parameter == 'supl':
        units = {'PRES'  : 'hPa',
                 'PTIME' : 'hhmm',
                 'WSPD'  : 'm/s',
                 'WDIR'  : 'degT',
                 'WTIME' : 'hhmm'
                }        
    elif parameter == 'swden':
        units = {'swden' : '(m*m)/Hz'}
    else :
        units = {'swden' : '(m*m)/Hz',
                 'PRES'  : 'hPa',
                 'PTIME' : 'hhmm',
                 'WDIR'  : 'degT',
                 'WTIME' : 'hhmm',
                 'GST'  : 'm/s',
                 'WVHT' : 'm' ,
                 'DPD'  : 'sec',
                 'APD'  : 'sec',
                 'MWD'  : 'degT',
                 'ATMP' : 'degC',
                 'WTMP' : 'degC',
                 'DEWP' : 'degC',
                 'VIS'  : 'nmi',
                 'PTDY' : 'hPa',
                 'TIDE' : 'ft',
                 'SRAD1'  : 'w/m2',
                 'SRAD2' : 'w/m2',
                 'SRAD3' : 'w/m2',
                 'WVHT': 'm',
                 'SwH' : 'm',
                 'SwP' : 'sec',
                 'WWH' : 'm',
                 'WWP' : 'sec',
                 'SwD' : '-',
                 'WWD' : 'degT',
                 'STEEPNESS' : '-',
                 'APD' : 'sec',
                 'RATE'  : 'mm/h',
                 'PCT'  : '%',
                 'SDEV'  : '-',
                 'ACCUM'  : 'mm',
                 'DEPTH'  : 'm',
                 'OTMP' : 'degC',
                 'COND'  : 'mS/cm',
                 'SAL'  : 'psu',
                 'O2%'  : '%',
                 'O2PPM' : 'ppm',
                 'CLCON' : 'ug/l',
                 'TURB' : 'FTU',
                 'PH' : '-',
                 'EH' : 'mv',
                 'CHILL'  : 'degC',
                 'HEAT'   : 'degC',
                 'ICE'    : 'cm/hr',
                 'WSPD'  : 'm/s',
                 'WSPD10' : 'm/s',
                 'WSPD20' : 'm/s',
                 'T'  : '-',
                 'HEIGHT' : 'm',
                 'GDR'  : 'degT',
                 'GST'  : 'm/s',
                 'GTIME' : 'hhmm',
                 'DEP01'  : 'm',
                 'DIR01' : 'deg',
                 'SPD01' : 'cm/s',
                 }

        
    units = _OrderedDict(sorted(units.items()))
        
    return units


def _supported_params(parameter):
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
    'mmbcur':	'Marsh-McBirney Current Measurements',
    'ocean'	:	'Oceanographic Current Year Historical Data'	,
    'rain'	:	'Hourly Rain Current Year Historical Data'	,
    'rain10':	'10-Minute Rain Current Year Historical Data'	,
    'rain24':	'24-Hour Rain Current Year Historical Data'	,
    'srad'	:	'Solar Radiation Current Year Historical Data'	,
    'stdmet':	'Standard Meteorological Current Year Historical Data'	,
    'supl'	:	'Supplemental Measurements Current Year Historical Data'	,
    'swden'	:	'Raw Spectral Wave Current Year Historical Data'	,
    'swdir'	:	'Spectral Wave Current Year Historical Data (alpha1)'	,
    'swdir2':	'Spectral Wave Current Year Historical Data (alpha2)'	,
    'swr1'	:	'Spectral Wave Current Year Historical Data (r1)'	,
    'swr2'	:	'Spectral Wave Current Year Historical Data (r2)'	,
    'wlevel':	'Tide Current Year Historical Data'	,
    }

    return supported	
    
