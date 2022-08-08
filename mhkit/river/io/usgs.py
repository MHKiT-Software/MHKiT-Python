import pandas as pd
import xarray as xr
import numpy as np
import json
import requests
from datetime import datetime

def _read_usgs_json(text):
    
    data = pd.DataFrame()
    for i in range(len(text['value']['timeSeries'])):
        try:
            site_name = text['value']['timeSeries'][i]['variable']['variableDescription'] #text['value']['timeSeries'][i]['sourceInfo']['siteName']
            site_data = pd.DataFrame(text['value']['timeSeries'][i]['values'][0]['value'])
            site_data.set_index('dateTime', drop=True, inplace=True)
            site_data.index = pd.to_datetime(site_data.index, utc=True)
            site_data.rename(columns={'value': site_name}, inplace=True)
            site_data[site_name] = pd.to_numeric(site_data[site_name])
            site_data.index.name = None
            del site_data['qualifiers']
            data = data.combine_first(site_data)
        except:
            pass
     
    return data # we could also extract metadata and return that here

def _read_usgs_json_to_xr(text):
    
    ds = xr.Dataset()
    for i in range(len(text['value']['timeSeries'])):
        site_name = text['value']['timeSeries'][i]['variable']['variableDescription']
        n_samples = len(text['value']['timeSeries'][i]['values'][0]['value'])
        data = np.zeros(n_samples)
        cols = np.empty(n_samples,dtype=object)
        try:           
            for qq in range(n_samples):
                data[qq] = text['value']['timeSeries'][i]['values'][0]['value'][qq]['value']
                cols[qq] = datetime.fromisoformat(
                    text['value']['timeSeries'][i]['values'][0]['value'][qq]['dateTime'])            
            attrs={
                'siteName': text['value']['timeSeries'][i]['sourceInfo']['siteName'],
                'siteCode': text['value']['timeSeries'][i]['sourceInfo']['siteCode'][0]['value'],
                'siteNetwork': text['value']['timeSeries'][i]['sourceInfo']['siteCode'][0]['network'],
                'siteAgencyCode': text['value']['timeSeries'][i]['sourceInfo']['siteCode'][0]['agencyCode'],
                'TimeZoneOffset': text['value']['timeSeries'][i]['sourceInfo']['timeZoneInfo']['defaultTimeZone']['zoneOffset'],
                'TimeZoneAbbreviation': text['value']['timeSeries'][i]['sourceInfo']['timeZoneInfo']['defaultTimeZone']['zoneAbbreviation'],
                'SiteUsesDaylightSavings': text['value']['timeSeries'][i]['sourceInfo']['timeZoneInfo']['siteUsesDaylightSavingsTime'],
                'DaylightSavingsTimeZoneOffset': text['value']['timeSeries'][i]['sourceInfo']['timeZoneInfo']['daylightSavingsTimeZone']['zoneOffset'],
                'latitude': text['value']['timeSeries'][i]['sourceInfo']['geoLocation']['geogLocation']['latitude'],
                'longitude': text['value']['timeSeries'][i]['sourceInfo']['geoLocation']['geogLocation']['longitude'],
                'variableName': text['value']['timeSeries'][i]['variable']['variableName'],
                'valueType': text['value']['timeSeries'][i]['variable']['valueType'],
                'units': text['value']['timeSeries'][i]['variable']['unit']['unitCode'],
                'USGS_Code': text['value']['timeSeries'][i]['name'],
                'queryURL': text['value']['queryInfo']['queryURL']} 
            if i == 0:
                ds[site_name] = xr.DataArray(data,
                                    attrs=attrs, 
                                    coords={'dateTime': cols},
                                    dims=['dateTime'])
            else:
                ds[site_name].combine_first(xr.DataArray(data,
                                    attrs=attrs, 
                                    coords={'dateTime': cols},
                                    dims=['dateTime']))
        except:
            pass
     
    return ds

def read_usgs_file(file_name, xarray=False):
    """
    Reads a USGS JSON data file (from https://waterdata.usgs.gov/nwis)

    Parameters
    ----------
    file_name : str
        Name of USGS JSON data file
    xarray : bool
        If true, return data will be xarray Dataset rather than pandas DataFrame.
        Default value is false.
        
    Returns
    -------
    data : pandas DataFrame 
        Data indexed by datetime with columns named according to the parameter's 
        variable description
    """
    with open(file_name) as json_file:
        text = json.load(json_file)

    if not xarray:
        data = _read_usgs_json(text)
    else:
        data = _read_usgs_json_to_xr(text)    
    
    return data 


def request_usgs_data(station, parameter, start_date, end_date, 
                      data_type='Daily', xarray=False, proxy=None, write_json=None):
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
    xarray : bool
        If true, return data will be xarray Dataset rather than pandas DataFrame.
        Default value is false.
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
    assert data_type in ['Daily', 'Instantaneous'], 'data_type must be Daily or Instantaneous'
    
    if data_type == 'Daily':
        data_url = 'https://waterservices.usgs.gov/nwis/dv'
        api_query = '/?format=json&sites='+station+ \
                    '&startDT='+start_date+'&endDT='+end_date+ \
                    '&statCd=00003'+ \
                    '&parameterCd='+parameter+'&siteStatus=all'
    else:
        data_url = 'https://waterservices.usgs.gov/nwis/iv'
        api_query = '/?format=json&sites='+station+ \
                    '&startDT='+start_date+'&endDT='+end_date+ \
                    '&parameterCd='+parameter+'&siteStatus=all'
            
    print('Data request URL: ', data_url+api_query)
    
    response = requests.get(url=data_url+api_query,proxies=proxy)
    text = json.loads(response.text)
    
    if write_json is not None:
        with open(write_json, 'w') as outfile:
            json.dump(text, outfile)
    
    if not xarray:
        data = _read_usgs_json(text)
    else:
        data = _read_usgs_json_to_xr(text)
    
    return data 
