import pandas as pd
import numpy as np
from rex import WaveX, MultiYearWaveX


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




def read_US_wave_dataset(wave_path, parameter, lat_lon, tree=None, 
                                 unscale=True, str_decode=True, hsds=True):
    
        """
        Reads data from the WPTO wave hindcast data hosted on AWS. 

        Note: To access the WPTO hindcast data, you will need to configure h5pyd for data access on HSDS. 
        To get your own API key, visit https://developer.nrel.gov/signup/. 

        To configure h5phd type 
        hsconfigure
        and enter at the prompt:
        hs_endpoint = https://developer.nrel.gov/api/hsds
        hs_username = None
        hs_password = None
        hs_api_key = {your key}
        Parameters
        ----------
        wave_path : string
            Path to US_Wave .h5 files
            Available formats:
                /nrel/US_wave/US_wave$_{year}.h5
                /nrel/US_wave/virtual_buoy/US_virtual_buoy_{year}.h5
        parameter: string
            dataset parameter to be downloaded
            spatial dataset options: directionality_coefficient, energy_period, maximum_energy_direction
                mean_absolute_period, mean_zero-crossing_period, omni-directional_wave_power, peak_period
                significant_wave_height, spectral_width, water_depth 
            virtual buoy options: directional_wave_spectrum, directionality_coefficient
                energy_period, maximum_energy_direction, mean_absolute_period, mean_wave_direction
                mean_zero-crossing_period, omni_directional_wave_power, peak_period, significant_wave_height
                spectral_width, water_depth
        lat_lon: tuple or list of tuples
            latitude longitude pairs at which to extract data 
        tree : str | cKDTree (optional)
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool (optional)
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool (optional)
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool (optional)
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        ---------
        data: pandas DataFrame 
            Data indexed by datetime with columns named according to header row 
        """
        
        assert isinstance(parameter, str), 'parameter must be of type string'
        assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'

        waveKwargs = {'tree':tree,'unscale':unscale,'str_decode':str_decode, 'hsds':hsds}
        
        
        if isinstance(wave_path,list) or '*' in wave_path:
            rex_accessor = MultiYearWaveX
        else: 
            rex_accessor = WaveX
            
        with rex_accessor(wave_path, **waveKwargs) as waves:
            data = waves.get_lat_lon_df(parameter,lat_lon)
        
        if data.shape[-1] == 1:
            return data.squeeze('columns')
        else:
            return data
            
            
        
        
        
        