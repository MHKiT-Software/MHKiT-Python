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


def fetch_NDBC(number, data="Spectral Wave Data", proxy=None):
    '''
    Historical data: Spectral wave density on ndbc.noaa.gov
    for the given buoy number.  
        
    Parameters
    ----------
    saveType: string
        If set to to "h5", the data will be saved in a compressed .h5
        file
        If set to "txt", the data will be stored in a raw .txt file
        Otherwise, a file will not be created
        NOTE: Only applies 
    savePath : string
        Relative path to place directory with data files.       
    '''
            
    # Returns link for each year of historical Spectral wave density data
    links = _get_links(number, proxy=proxy)
    
    noaa_data = {}
    for key in links:
        key_URL = f'https://ndbc.noaa.gov{links[key]}'
        file_name = key_URL.replace('download_data', 'view_text_file')
        response =  requests.get(file_name)
        df = pd.read_csv(StringIO(response.text), sep='\s+')
        noaa_data[key] = df
        #import ipdb; ipdb.set_trace()
    return noaa_data
    
    
def _get_links(number, data="Spectral Wave Data", proxy=None):  
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
        links to NOAA data indexed by href key
    '''
    
    if data != "Spectral Wave Data":
        "Currently only Historical Spectral Wave Data is supported. If you would like to see more data types please open an issue or submit a Pull Request on GitHub"
        #break
        
    #prepares to pull the data from the NDBC website
    url = f'https://www.ndbc.noaa.gov/station_history.php?station={number}'
    if proxy == None:
        ndbcURL = requests.get(url)
    else:
        ndbcURL = requests.get(url, proxies=proxy)

    ndbcURL.raise_for_status()
    ndbcHTML = bs4.BeautifulSoup(ndbcURL.text, "lxml")
    headers = ndbcHTML.findAll("b", text="Spectral wave density data: ")
    
    #checks for headers in differently formatted webpages
    if len(headers) == 0:
        raise Exception("Spectral wave density data for buoy #%s not found" % number)

    if len(headers) == 2:
        headers = headers[1]
    else:
        headers = headers[0]

    links = {a.string: a["href"] for a in headers.find_next_siblings("a", href=True)}
        
    return links    
    
    
def noaa_dates_to_DateTime(dataframe, data="Spectral Wave Data"):
    '''
    Analyzes a DataFrame header to convert #YY  MM DD hh mm
    to return a DataFrame with the NOAA DateColumns removed
    and a new ['date'] columns with DateTime Format
    
    Parameters
    ----------
    dataframe: DataFrame
        NOAA Datafrmae with headers ['YY', 'MM', 'DD', 'hh', {'mm'}]
    data: string
        Specifies the type of NOAA data
        
    Returns
    -------
    df: DataFrame
        DateFrame with NOAA dates dropped and new ['date']
        column in DateTime format
    '''

    if data != "Spectral Wave Data":
        "Currently only Historical Spectral Wave Data is supported. If you would like to see more data types please open an issue or submit a Pull Request on GitHub"
        #break
    
    df = dataframe.copy(deep=True)
    hours_loc = np.where(df.columns.values == 'hh')[0][0]       
    minutes = df.columns[hours_loc+1] == 'mm'

    all_columns = df.columns.values.tolist()
    year_col = all_columns[0]
    
    is_pound = ('#' in year_col)
    if is_pound: 
        df = df.rename(columns={year_col: year_col.split('#')[1]})
        all_columns = df.columns.values.tolist()
        year_col = all_columns[0]
        year_fmt = '%Y'
        
    elif year_col == 'YYYY':
        year_fmt = '%Y'
    elif year_col =='YY':
        year_fmt = '%y' 

    if minutes:
        columns = all_columns[0:hours_loc+2]
            
        # Convert to str and zero pad
        for key in columns:
            df[key] = df[key].astype(str).str.zfill(2)
        
        df['date_string'] =  [str(a)+str(b)+str(c)+str(d)+str(e) for a,b,c,d,e in zip(df[columns[0]], df[columns[1]], df[columns[2]], df[columns[3]], df[columns[4]])]  
        df['date'] = pd.to_datetime(df['date_string'], format=f'{year_fmt}%m%d%H%M')
        del df['date_string']
        
        df.drop(columns[0:hours_loc+2], axis=1, inplace=True)
        
    else:
        columns = all_columns[0:hours_loc+1]
        # Convert to str and zero pad
        for key in columns[0:hours_loc+1]:          
            df[key] = df[key].astype(str).str.zfill(2)
        
        df['date_string'] =  [str(a)+str(b)+str(c)+str(d) for a,b,c,d in zip(df[columns[0]], df[columns[1]], df[columns[2]], df[columns[3]])]
        df['date'] = pd.to_datetime(df['date_string'], format=f'{year_fmt}%m%d%H')
        del df['date_string']
        
        df.drop(columns[0:hours_loc+1], axis=1, inplace=True)
    
    return df
    