import xml.etree.ElementTree as ET
import datetime
import json
import math
import pandas as pd
import requests


def request_noaa_data(station, parameter, start_date, end_date,
                      proxy=None, write_json=None):
    """
    Loads NOAA current data directly from https://tidesandcurrents.noaa.gov/api/ using a 
    get request into a pandas DataFrame. NOAA sets max of 31 days between start and end date.
    See https://co-ops.nos.noaa.gov/api/ for options. All times are reported as GMT and metric
    units are returned for data.

    The request URL prints to the screen.

    Parameters
    ----------
    station : str
        NOAA current station number (e.g. 'cp0101')
    parameter : str
        NOAA paramter (e.g. '' for Discharge, cubic feet per second)
    start_date : str
        Start date in the format yyyyMMdd
    end_date : str
        End date in the format yyyyMMdd 
    proxy : dict or None
         To request data from behind a firewall, define a dictionary of proxy settings, 
         for example {"http": 'localhost:8080'}
    write_json : str or None
        Name of json file to write data

    Returns
    -------
    data : pandas DataFrame 
        Data indexed by datetime with columns named according to the parameter's 
        variable description

    """
    # Convert start and end dates to datetime objects
    begin = datetime.datetime.strptime(start_date, '%Y%m%d').date()
    end = datetime.datetime.strptime(end_date, '%Y%m%d').date()
    # Convert to datetime
    begin = datetime.date(year0, month0, day0)
    end = datetime.date(year1, month1, day1)
    # Determine the number of 30 day intervals
    delta = 30
    interval = math.ceil(((end - begin).days)/delta)
    # Intialize date list
    date_list = []
    # Create 30 day intervals
    for i in range(interval + 1):
        date_list.append(
            (begin+i*datetime.timedelta(days=delta)).strftime('%Y%m%d'))
    # Replace last entry in date list with end time
    date_list[-1] = end_date
    # Intialize dictionary to hold responses
    dataFrames = {}
    # Iterate over date_list (30 day intervals)
    for i in range(len(date_list)-1):
        start_date = date_list[i]
        end_date = date_list[i+1]
        data_url = 'https://tidesandcurrents.noaa.gov/api/datagetter?'
        api_query = 'begin_date='+start_date + \
                    '&end_date='+end_date + \
                    '&station='+station + \
                    '&product='+parameter + \
                    '&units=metric&' +  \
                    'time_zone=gmt&' +\
                    'application=web_services&' + \
                    'format=xml'
        print('Data request URL: ', data_url+api_query)
        # Get response
        response = requests.get(url=data_url+api_query, proxies=proxy)
        # Connvert to DataFrame and save in Dictionary
        dataFrames[date_list[i]], metadata = _xml_to_dataframe(response)
        # Future TODO: Add option to request data as json
        # dataFrames[date_list[i]], metadata = _json_to_dataframe (response)
    # Get first DataFrame
    data = dataFrames[date_list[0]]
    # Append all remaining DataFrames
    if len(dataFrames) > 1:
        for i in range(1, len(dataFrames)):
            data = data.append(dataFrames[date_list[i]])
    # Remove duplicated date values
    data = data[~data.index.duplicated()]
    # Write json if specified
    if write_json is not None:
        with open(write_json, 'w') as outfile:
            # Convert DataFrame to json
            jsonData = data.to_json()
            # Convert to python object data
            pyData = json.loads(jsonData)
            # Add metadata to pyData
            pyData['metadata'] = metadata
            # Wrtie the pyData to a json file
            json.dump(pyData, outfile)
    # import ipdb; ipdb.set_trace()
    return data, metadata


def _json_to_dataframe(response):
    '''
    Returns a dataframe  and metadata from a NOAA
    response.
    TODO: This function currently does not return the 
      full dataset requested.
    '''
    text = json.loads(response.text)
    metadata = text['metadata']
    # import ipdb; ipdb.set_trace()
    # Initialize DataFrame
    data = pd.DataFrame.from_records(
        text['data'][1], index=[text['data'][1]['t']])
    # Append all times to DataFrame
    for i in range(1, len(text['data'])):
        data.append(pd.DataFrame.from_records(text['data'][i],
                                              index=[text['data'][i]['t']]))
    # Convert index to DataFram
    data.index = pd.to_datetime(data.index)
    # Remove 't' becuase it is the index
    del data['t']
    # List of columns which are string
    cols = data.columns[data.dtypes.eq('object')]
    # Convert columns to float
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    return data, metadata


def _xml_to_dataframe(response):
    '''
    Returns a dataframe from an xml response
    '''
    root = ET.fromstring(response.text)
    metadata = None
    data = None

    for child in root:
        # Save meta data dictionary
        if child.tag == 'metadata':
            metadata = child.attrib
        elif child.tag == 'observations':
            data = child
        elif child.tag == 'error':
            print('***ERROR: Response returned error')
            return None

    if data is None:
        print('***ERROR: No observations found')
        return None

    # Create a list of DataFrames then Concatenate
    df = pd.concat([pd.DataFrame(obs.attrib, index=[0])
                   for obs in data], ignore_index=True)

    # Convert time to datetime
    df['t'] = pd.to_datetime(df.t)
    df = df.set_index('t')
    df.drop_duplicates(inplace=True)

    # Convert data to float
    df[['d', 's']] = df[['d', 's']].apply(pd.to_numeric)

    return df, metadata


def read_noaa_json(filename):
    '''
    Returns site DataFrame and metadata from a json saved from the 
    request_noaa_data
    Parameters
    ----------
    filename: string
        filename with path of json file to load
    Returns
    -------
    data: DataFrame
        Timeseries Site data of direction and speed 
    metadata: dictionary
        Site metadata
    '''
    with open(filename) as outfile:
        jsonData = json.load(outfile)
    # Get the metadata
    metadata = jsonData['metadata']
    # Remove metadata entry
    del jsonData['metadata']
    # Remainder is DataFrame
    data = pd.DataFrame.from_dict(jsonData)
    # Convert from epoch to date time
    data.index = pd.to_datetime(data.index, unit='ms')
    return data, metadata
