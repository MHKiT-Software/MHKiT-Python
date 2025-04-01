"""
usgs.py

This module provides functions for retrieving and processing data from the United States
Geological Survey (USGS) National Water Information System (NWIS). It enables access to
river flow data and related measurements useful for hydrokinetic resource assessment.

Functions:
----------
- read_usgs_file: Read data from USGS data files
- request_usgs_data: Fetch data directly from USGS web services
- process_usgs_data: Process and validate USGS data formats

"""

import os
import json
import shutil
import requests
import pandas as pd
from mhkit.utils.cache import handle_caching


def _read_usgs_json(text, to_pandas=True):
    """
    Process USGS JSON response into a pandas DataFrame or xarray Dataset.

    Parameters
    ----------
    text : dict
        JSON response from USGS API containing time series data
    to_pandas : bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data : pandas.DataFrame or xarray.Dataset
        Processed time series data
    """
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
        except (KeyError, ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
            print(f"Warning: Failed to process time series {i}: {str(e)}")
            continue

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

    with open(file_name, encoding="utf-8") as json_file:
        text = json.load(json_file)

    data = _read_usgs_json(text, to_pandas)

    return data


# pylint: disable=too-many-locals
def request_usgs_data(
    station,
    parameter,
    start_date,
    end_date,
    options=None,
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
        USGS parameter ID (e.g. '00060' for Discharge, cubic feet per second)
    start_date : str
        Start date in the format 'YYYY-MM-DD' (e.g. '2018-01-01')
    end_date : str
        End date in the format 'YYYY-MM-DD' (e.g. '2018-12-31')
    options : dict, optional
        Dictionary containing optional parameters:
        - data_type: str
            Data type, options include 'Daily' (return the mean daily value) and
            'Instantaneous'. Default = 'Daily'
        - proxy: dict or None
            Proxy settings for the request. Default = None
        - write_json: str or None
            Name of json file to write data. Default = None
        - clear_cache: bool
            If True, the cache for this specific request will be cleared. Default = False
        - to_pandas: bool
            Flag to output pandas instead of xarray. Default = True
        - timeout: int
            Timeout in seconds for the HTTP request. Default = 30

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named according to the parameter's
        variable description
    """
    # Set default options
    options = options or {}
    data_type = options.get("data_type", "Daily")
    proxy = options.get("proxy", None)
    write_json = options.get("write_json", None)
    clear_cache = options.get("clear_cache", False)
    to_pandas = options.get("to_pandas", True)
    timeout = options.get("timeout", 30)  # 30 seconds default timeout

    if data_type not in ["Daily", "Instantaneous"]:
        raise ValueError(f"data_type must be Daily or Instantaneous. Got: {data_type}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number. Got: {timeout}")

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "usgs")

    # Create a unique filename based on the function parameters
    hash_params = f"{station}_{parameter}_{start_date}_{end_date}_{data_type}"

    cached_data, _, cache_filepath = handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": None, "metadata": None, "write_json": write_json},
        clear_cache_file=clear_cache,
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

    response = requests.get(url=data_url + api_query, proxies=proxy, timeout=timeout)
    text = json.loads(response.text)

    # handle_caching is only set-up for pandas, so force this data to output as pandas for now
    data = _read_usgs_json(text, True)

    # After making the API request and processing the response, write the
    #  response to a cache file
    handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": data, "metadata": None, "write_json": None},
        clear_cache_file=clear_cache,
    )

    if write_json:
        shutil.copy(cache_filepath, write_json)

    if not to_pandas:
        data = data.to_dataset()

    return data
