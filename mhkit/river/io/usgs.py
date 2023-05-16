import os
import pandas as pd
import json
import requests
import shutil
from mhkit.utils.cache_utils import handle_caching


def _read_usgs_json(text):

    data = pd.DataFrame()
    for i in range(len(text['value']['timeSeries'])):
        try:
            # text['value']['timeSeries'][i]['sourceInfo']['siteName']
            site_name = text['value']['timeSeries'][i]['variable']['variableDescription']
            site_data = pd.DataFrame(
                text['value']['timeSeries'][i]['values'][0]['value'])
            site_data.set_index('dateTime', drop=True, inplace=True)
            site_data.index = pd.to_datetime(site_data.index, utc=True)
            site_data.rename(columns={'value': site_name}, inplace=True)
            site_data[site_name] = pd.to_numeric(site_data[site_name])
            site_data.index.name = None
            del site_data['qualifiers']
            data = data.combine_first(site_data)
        except:
            pass

    return data  # we could also extract metadata and return that here


def read_usgs_file(file_name):
    """
    Reads a USGS JSON data file (from https://waterdata.usgs.gov/nwis)

    Parameters
    ----------
    file_name : str
        Name of USGS JSON data file

    Returns
    -------
    data : pandas DataFrame 
        Data indexed by datetime with columns named according to the parameter's 
        variable description
    """
    with open(file_name) as json_file:
        text = json.load(json_file)

    data = _read_usgs_json(text)

    return data


def request_usgs_data(
        station,
        parameter,
        start_date,
        end_date,
        data_type='Daily',
        proxy=None,
        write_json=None,
        clear_cache=False):
    """
    Loads USGS data directly from https://waterdata.usgs.gov/nwis using a 
    GET request

    The request URL prints to the screen.

    Parameters
    ----------
    station : str
        USGS station number (e.g. '08313000')
    parameter : str
        USGS paramter ID (e.g. '00060' for Discharge, cubic feet per second)
    start_date : str
        Start date in the format 'YYYY-MM-DD' (e.g. '2018-01-01')
    end_date : str
        End date in the format 'YYYY-MM-DD' (e.g. '2018-12-31')
    data_type : str
        Data type, options include 'Daily' (return the mean daily value) and 
        'Instantaneous'.
    proxy : dict or None
         To request data from behind a firewall, define a dictionary of proxy settings, 
         for example {"http": 'localhost:8080'}
    write_json : str or None
        Name of json file to write data
    clear_cache : bool
        If True, the cache for this specific request will be cleared.         

    Returns
    -------
    data : pandas DataFrame 
        Data indexed by datetime with columns named according to the parameter's 
        variable description
    """
    assert data_type in [
        'Daily', 'Instantaneous'], 'data_type must be Daily or Instantaneous'

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"),
                             ".cache", "mhkit", "usgs")

    # Create a unique filename based on the function parameters
    hash_params = f"{station}_{parameter}_{start_date}_{end_date}_{data_type}"

    # Use handle_caching to manage cache
    cached_data, metadata, cache_filepath = handle_caching(
        hash_params, cache_dir, write_json, clear_cache)

    if cached_data is not None:
        return cached_data

    # If no cached data, proceed with the API request
    if data_type == 'Daily':
        data_url = 'https://waterservices.usgs.gov/nwis/dv'
        api_query = '/?format=json&sites='+station + \
                    '&startDT='+start_date+'&endDT='+end_date + \
                    '&statCd=00003' + \
                    '&parameterCd='+parameter+'&siteStatus=all'
    else:
        data_url = 'https://waterservices.usgs.gov/nwis/iv'
        api_query = '/?format=json&sites='+station + \
                    '&startDT='+start_date+'&endDT='+end_date + \
                    '&parameterCd='+parameter+'&siteStatus=all'

    print('Data request URL: ', data_url+api_query)

    response = requests.get(url=data_url+api_query, proxies=proxy)
    text = json.loads(response.text)

    data = _read_usgs_json(text)

    # After making the API request and processing the response, write the
    #  response to a cache file
    handle_caching(hash_params, cache_dir, data=data, clear_cache=clear_cache)

    if write_json:
        shutil.copy(cache_filepath, write_json)

    return data
