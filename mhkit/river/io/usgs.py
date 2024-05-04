import os
import json
import requests
import shutil
import pandas as pd
from mhkit.utils.cache import handle_caching


def _read_usgs_json(text, to_pandas=True):
    data = pd.DataFrame()
    for i in range(len(text["value"]["timeSeries"])):
        try:
            site_name = text["value"]["timeSeries"][i]["variable"][
                "variableDescription"
            ]
            site_data = pd.DataFrame(
                text["value"]["timeSeries"][i]["values"][0]["value"]
            )
            site_data.set_index("dateTime", drop=True, inplace=True)
            site_data.index = pd.to_datetime(site_data.index, utc=True)
            site_data.rename(columns={"value": site_name}, inplace=True)
            site_data[site_name] = pd.to_numeric(site_data[site_name])
            site_data.index.name = None
            del site_data["qualifiers"]
            data = data.combine_first(site_data)
        except:
            pass

    if not to_pandas:
        data = data.to_dataset()

    return data


def read_usgs_file(file_name, to_pandas=True):
    """
    Reads a USGS JSON data file (from https://waterdata.usgs.gov/nwis)

    Parameters
    ----------
    file_name : str
        Name of USGS JSON data file
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named according to the parameter's
        variable description
    """
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    with open(file_name) as json_file:
        text = json.load(json_file)

    data = _read_usgs_json(text, to_pandas)

    return data


def request_usgs_data(
    station,
    parameter,
    start_date,
    end_date,
    data_type="Daily",
    proxy=None,
    write_json=None,
    clear_cache=False,
    to_pandas=True,
):
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
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named according to the parameter's
        variable description
    """
    if not data_type in ["Daily", "Instantaneous"]:
        raise ValueError(f"data_type must be Daily or Instantaneous. Got: {data_type}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "usgs")

    # Create a unique filename based on the function parameters
    hash_params = f"{station}_{parameter}_{start_date}_{end_date}_{data_type}"

    # Use handle_caching to manage cache
    cached_data, metadata, cache_filepath = handle_caching(
        hash_params, cache_dir, write_json, clear_cache
    )

    if cached_data is not None:
        return cached_data

    # If no cached data, proceed with the API request
    if data_type == "Daily":
        data_url = "https://waterservices.usgs.gov/nwis/dv"
        api_query = (
            "/?format=json&sites="
            + station
            + "&startDT="
            + start_date
            + "&endDT="
            + end_date
            + "&statCd=00003"
            + "&parameterCd="
            + parameter
            + "&siteStatus=all"
        )
    else:
        data_url = "https://waterservices.usgs.gov/nwis/iv"
        api_query = (
            "/?format=json&sites="
            + station
            + "&startDT="
            + start_date
            + "&endDT="
            + end_date
            + "&parameterCd="
            + parameter
            + "&siteStatus=all"
        )

    print("Data request URL: ", data_url + api_query)

    response = requests.get(url=data_url + api_query, proxies=proxy)
    text = json.loads(response.text)

    # handle_caching is only set-up for pandas, so force this data to output as pandas for now
    data = _read_usgs_json(text, True)

    # After making the API request and processing the response, write the
    #  response to a cache file
    handle_caching(hash_params, cache_dir, data=data, clear_cache_file=clear_cache)

    if write_json:
        shutil.copy(cache_filepath, write_json)

    if not to_pandas:
        data = data.to_dataset()

    return data
