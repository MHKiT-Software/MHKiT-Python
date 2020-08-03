from io import StringIO
import pandas as pd
import numpy as np
import requests
import bs4


def read_NDBC_file(file_name, missing_values=['MM',9999,999,99]):
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


def get_available_ndbc_data(number, 
                            data="Spectral Wave Data", 
                            proxy=None):  
    '''
    Returns a dictionary of links indexed by hyperlink reference 
    e.g. year.
    
    Parameters
    ----------
    number: string
        Buoy Number
    proxy: string
        proxy url
        
    Returns
    -------
    links: Dict
        Links to NDBC data indexed by href key
    '''
    
    if data != "Spectral Wave Data":
        msg = __supported_ndbc_params()
        return msg
        
    ndbc_buoy_url = f'https://www.ndbc.noaa.gov/station_history.php?station={number}'
    if proxy == None:
        ndbcURL = requests.get(ndbc_buoy_url)
    else:
        ndbcURL = requests.get(ndbc_buoy_url, proxies=proxy)

    ndbcURL.raise_for_status()
    ndbcHTML = bs4.BeautifulSoup(ndbcURL.text, "lxml")
    headers = ndbcHTML.findAll("b", text="Spectral wave density data: ")
    
    #checks for headers in differently formatted webpages
    if len(headers) == 0:
        msg=f"Spectral wave density data for buoy {number} not found"
        raise Exception(msg)

    if len(headers) == 2:
        headers = headers[1]
    else:
        headers = headers[0]

    links = {a.string: a["href"] for a in headers.find_next_siblings("a",
        href=True)}
        
    return links    
    
    
def fetch_ndbc(links, data="Spectral Wave Data", proxy=None):
    '''
    Returns a DataFrame for each {key: link}  element passed in the 
	links dictionary.
        
    Parameters
    ----------
    links: Dict
	    Data link dict from `get_available_ndbc_data`
	data: string
	    NDBC data product
	proxy: string
	    Proxy URL   
        
    Returns
    -------
    ndbc_data: dict
        Dictionary of NDBC data 
    '''

    if data != "Spectral Wave Data":
        msg = __supported_ndbc_params()
        return msg
	              
    ndbc_data = {}
    for key in links:
        key_URL = f'https://ndbc.noaa.gov{links[key]}'
        file_name = key_URL.replace('download_data', 'view_text_file')
        response =  requests.get(file_name)
        df = pd.read_csv(StringIO(response.text), sep='\s+')
        ndbc_data[key] = df
    return ndbc_data
        

def ndbc_dates_to_datetime(dataframe, data="Spectral Wave Data", 
                           return_date_cols=False):
    '''
    Takes a DataFrame and converts the NDBC date columns 
	(e.g. "#YY  MM DD hh mm") to datetime. Returns a DataFrame with the 
	removed NDBC date columns a new ['date'] columns with DateTime Format.
    
    Parameters
    ----------
    dataframe: DataFrame
        Dataframe with headers (e.g. ['YY', 'MM', 'DD', 'hh', {'mm'}])
    data: string
        Specifies the type of NDBC data
    return_date_col: Bool
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

    if data != "Spectral Wave Data":
        msg = __supported_ndbc_params()
        return msg
    
    # Remove frequency columns    
    df = dataframe.copy(deep=True)     
    times_only = __remove_columns(df, starts_with='.')
       
    ndbc_date_cols = times_only.columns.values.tolist()
    if len(ndbc_date_cols) == 4:
        minutes = False
    elif len(ndbc_date_cols) ==5:
        minutes = True
    else:
        msg:f"Error unexpected length of time. Return: {ndbc_date_cols} "         
    
    # So far these have been consistiently names
    months_loc = ndbc_date_cols.index('MM')
    days_loc   = ndbc_date_cols.index('DD')
    hours_loc  = ndbc_date_cols.index('hh')
    if minutes:
        minutes_loc  = ndbc_date_cols.index('mm')
        index_exclude_year = [months_loc, days_loc, hours_loc, minutes_loc]
    else:
        index_exclude_year = [months_loc, days_loc, hours_loc]
        
    years_loc = [*set([*range(len(ndbc_date_cols))]) - set(index_exclude_year)][0]          
    year_string = ndbc_date_cols[years_loc]
    
    if ('#' in year_string) or (year_string == 'YYYY'):
        year_fmt = '%Y'
    elif year_string =='YY':
        year_fmt = '%y' 
               
    df = __date_string_to_datetime(df, ndbc_date_cols, year_fmt)        
    date = df['date']       
    del df
    
    if return_date_cols:
        return date, ndbc_date_cols
    return date

    
def __supported_ndbc_params():
    '''
    There is a significant number of datasets provided by NDBC. There is
    specific data processing required for each type. Therefore this 
    function is thrown for any data type not currently covered.
    Parameters
    ----------
    None
    
    Returns
    -------
    msg: string
        string indicating what is supported and how to request or add
        new functionality    
    '''
    
    msg = ["Currently only Historical Spectral Wave Data is "+
    "supported. If you would like to see more data types please"+
    " open an issue or submit a Pull Request on GitHub"]
    print(msg)
    return msg	
	

def __date_string_to_datetime(df, columns, year_fmt):
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
    
    # Convert to str and zero pad
    for key in columns:
        df[key] = df[key].astype(str).str.zfill(2)
    
    df['date_string'] = df[columns[0]]
    for column in columns[1:]:
        df['date_string'] = df[['date_string', column]].apply(lambda x: ''.join(x), axis=1)
    df['date'] = pd.to_datetime(df['date_string'], format=f'{year_fmt}%m%d%H%M')
    del df['date_string']
    
    return df
    

def __remove_columns(df, starts_with='.'):
    '''
    Removes column names that start with '.' and returns the modified 
    DataFrame.
    
    Parameters
    ----------
    df: Dataframe
        Dataframe with columns that start_with=pattern to be removed
    starts_with: str
        Removes all columns that start with the specified pattern
    '''  
    for column in df:
        if column.startswith(starts_with):
            del df[column]    
    return df
    