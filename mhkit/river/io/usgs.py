"""
This module provides functions for retrieving and processing data from the United States
Geological Survey (USGS) Water Data APIs (https://api.waterdata.usgs.gov). These
APIs follow the Open Geospatial Consortium (OGC) API standards
(https://ogcapi.ogc.org) and are referred to here as the OGC APIs. The module enables
access to river flow data and related measurements useful for hydrokinetic resource
assessment.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import json
import shutil
import warnings
import zoneinfo
import requests
import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from mhkit.utils.cache import handle_caching
from mhkit.utils.type_handling import convert_to_dataset


def _read_waterml_json(text: Dict) -> DataFrame:
    """
    Process a legacy USGS WaterServices (WaterML JSON) response into a
    pandas DataFrame.

    Parameters
    ----------
    text : dict
        WaterML JSON response containing time series data

    Returns
    -------
    data : pandas.DataFrame
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

    return data


def _read_ogc_json(
    features: List[Dict],
    column_name: Optional[str] = None,
    include_status: bool = False,
) -> DataFrame:
    """
    Process USGS Water Data OGC API features into a pandas DataFrame.

    Parameters
    ----------
    features : list of dict
        GeoJSON features, one per observation
    column_name : str, optional
        Name of the value column. Defaults to the features' parameter code.
    include_status : bool, optional
        If True, include the 'approval_status' and 'qualifier' columns.
        Default = False.

    Returns
    -------
    data : pandas.DataFrame
        Processed time series data
    """
    columns = ["time", "value"] + (
        ["approval_status", "qualifier"] if include_status else []
    )
    data = pd.DataFrame(
        [feature["properties"] for feature in features], columns=columns
    ).set_index("time")
    data.index = pd.to_datetime(data.index, utc=True, format="ISO8601")
    data.index.name = None
    data["value"] = pd.to_numeric(data["value"])
    # The API does not return features in time order
    data = data.sort_index(kind="stable")
    data = data[~data.index.duplicated(keep="first")]

    if column_name is None:
        codes = [feature["properties"].get("parameter_code") for feature in features]
        column_name = next((code for code in codes if code), "value")

    return data.rename(columns={"value": column_name})


def _read_usgs_json(
    text: Dict,
    to_pandas: bool = True,
    column_name: Optional[str] = None,
    include_status: bool = False,
) -> Union[DataFrame, xr.Dataset]:
    """
    Process a USGS JSON response into a pandas DataFrame or xarray Dataset.

    Both the legacy WaterServices (WaterML JSON) format and the Water Data
    OGC API (GeoJSON FeatureCollection) format are supported.

    Parameters
    ----------
    text : dict
        JSON response from a USGS API containing time series data
    to_pandas : bool, optional
        Flag to output pandas instead of xarray. Default = True.
    column_name : str, optional
        Name of the value column for OGC API data. Defaults to the parameter code.
    include_status : bool, optional
        If True, include the 'approval_status' and 'qualifier' columns for
        OGC API data. Default = False.

    Returns
    -------
    data : pandas.DataFrame or xarray.Dataset
        Processed time series data
    """
    if "timeSeries" in text.get("value", {}):
        data = _read_waterml_json(text)
    elif text.get("type") == "FeatureCollection":
        data = _read_ogc_json(text["features"], column_name, include_status)
    else:
        raise ValueError(
            "Unrecognized USGS JSON format. Expected WaterML JSON or a GeoJSON "
            "FeatureCollection."
        )

    if not to_pandas:
        data = convert_to_dataset(data)

    return data


def read_usgs_file(
    file_name: str, to_pandas: bool = True
) -> Union[DataFrame, xr.Dataset]:
    """
    Reads a USGS JSON data file, either from the legacy WaterServices
    (https://waterservices.usgs.gov) or the Water Data Open Geospatial
    Consortium (OGC) APIs (https://api.waterdata.usgs.gov)

    Parameters
    ----------
    file_name : str
        Name of USGS JSON data file
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by datetime. Legacy files have columns named according
        to the parameter's variable description. OGC API files have a column
        named according to the parameter code (e.g. '00060').
    """
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    with open(file_name, encoding="utf-8") as json_file:
        text = json.load(json_file)

    data = _read_usgs_json(text, to_pandas)

    return data


def _get_ogc_pages(
    url: str,
    params: Dict,
    headers: Dict,
    proxy: Optional[Dict],
    timeout: Union[int, float],
) -> List[Dict]:
    """
    Sends GET requests to a USGS Water Data OGC API, following the API's
    pagination links, and returns each JSON response. SSL errors are retried.

    Parameters
    ----------
    url : str
        Request URL
    params : dict
        Query parameters
    headers : dict
        Request headers
    proxy : dict or None
        Proxy settings for the request
    timeout : int or float
        Timeout in seconds for the HTTP request

    Returns
    -------
    pages : list of dict
        JSON responses
    """
    print(
        "Data request URL: ", requests.Request("GET", url, params=params).prepare().url
    )

    pages = []
    max_retries = 3
    while url:
        retry_count = 0
        while True:
            try:
                response = requests.get(
                    url=url,
                    params=params,
                    headers=headers,
                    proxies=proxy,
                    timeout=timeout,
                    verify=True,
                )
                break
            except requests.exceptions.SSLError as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                print(
                    f"SSL Error occurred, retrying... (Attempt {retry_count}/{max_retries})"
                )

        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                "USGS API rate limit exceeded (HTTP 429). Request an API key at "
                "https://api.waterdata.usgs.gov/signup/ and pass it with "
                "options={'api_key': ...} or the USGS_API_KEY environment variable.",
                response=response,
            )
        response.raise_for_status()

        page = json.loads(response.text)
        pages.append(page)
        # The next link already contains the query parameters
        url = next(
            (link["href"] for link in page.get("links", []) if link["rel"] == "next"),
            None,
        )
        params = None

    return pages


def _time_windows(
    start_date: str, end_date: str, data_type: str, timezone: str = "UTC"
) -> List[Tuple[str, str]]:
    """
    Builds the time envelope(s) for a request. Instantaneous dates are local
    days in the timezone, and requests are split into windows within the API's
    1100 day limit for that collection. If local midnight does not exist or is
    ambiguous due to a daylight saving time change, a warning is issued and
    pandas shifts to the first valid time.

    Parameters
    ----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD', inclusive
    data_type : str
        'Daily' or 'Instantaneous'
    timezone : str, optional
        Timezone of the Instantaneous start and end dates. Default = 'UTC'

    Returns
    -------
    windows : list of tuple of str
        (start, end) pairs formatted for the API's 'time' parameter
    """
    if data_type == "Daily":
        return [(start_date, end_date)]

    # The continuous collection rejects time ranges longer than 1100 days.
    # Leave a day of margin so daylight saving time changes cannot push the
    # envelope past the limit.
    starts = pd.date_range(start_date, end_date, freq="1099D")
    # Each window ends just before local midnight after its last day
    ends = starts[1:].append(
        pd.DatetimeIndex([pd.Timestamp(end_date) + pd.DateOffset(days=1)])
    )
    midnights = pd.DatetimeIndex(starts.append(ends))

    local = midnights.tz_localize(timezone, nonexistent="NaT", ambiguous="NaT")
    if local.isna().any():
        warnings.warn(
            "Local midnight does not exist or is ambiguous due to daylight saving "
            f"time in {timezone} on "
            f"{', '.join(midnights[local.isna()].strftime('%Y-%m-%d'))}. "
            "Using the first valid time.",
            UserWarning,
        )
        local = midnights.tz_localize(
            timezone,
            nonexistent="shift_forward",
            ambiguous=np.ones(len(midnights), dtype=bool),
        )
    utc = local.tz_convert("UTC")

    time_format = "%Y-%m-%dT%H:%M:%SZ"
    return [
        (
            start.strftime(time_format),
            (end - pd.Timedelta(seconds=1)).strftime(time_format),
        )
        for start, end in zip(utc[: len(starts)], utc[len(starts) :])
    ]


# pylint: disable=too-many-arguments, too-many-locals
def _request_ogc_data(
    station: str,
    parameter: str,
    start_date: str,
    end_date: str,
    data_type: str,
    request_options: Dict,
    proxy: Optional[Dict],
    timeout: Union[int, float],
) -> DataFrame:
    """
    Requests data from the USGS Water Data OGC APIs.

    Parameters
    ----------
    station : str
        USGS station number
    parameter : str
        USGS parameter code
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD', inclusive
    data_type : str
        'Daily' or 'Instantaneous'
    request_options : dict
        'api_key' (str or None), 'include_status' (bool) and 'timezone' (str)
    proxy : dict or None
        Proxy settings for the request
    timeout : int or float
        Timeout in seconds for the HTTP request

    Returns
    -------
    data : pandas.DataFrame
        Processed time series data
    """
    api_key = request_options["api_key"]
    base_url = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
    headers = {"X-Api-Key": api_key} if api_key else {}
    location_id = station if station.startswith("USGS-") else f"USGS-{station}"
    params = {
        "f": "json",
        "monitoring_location_id": location_id,
        "parameter_code": parameter,
        # Maximum number of features the API returns per page
        "limit": 50000,
    }
    if data_type == "Daily":
        data_url = f"{base_url}/daily/items"
        params["statistic_id"] = "00003"
    else:
        data_url = f"{base_url}/continuous/items"

    features = []
    for window_start, window_end in _time_windows(
        start_date, end_date, data_type, request_options["timezone"]
    ):
        params["time"] = f"{window_start}/{window_end}"
        for page in _get_ogc_pages(data_url, dict(params), headers, proxy, timeout):
            features.extend(page.get("features", []))

    # Name the column with the parameter description, as the legacy service did
    try:
        column_name = _get_ogc_pages(
            f"{base_url}/parameter-codes/items/{parameter}",
            {"f": "json"},
            headers,
            proxy,
            timeout,
        )[0]["properties"]["parameter_description"]
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Warning: Failed to look up parameter {parameter}: {str(e)}")
        column_name = parameter

    return _read_usgs_json(
        {"type": "FeatureCollection", "features": features},
        True,
        column_name,
        request_options["include_status"],
    )


# pylint: disable=too-many-arguments
def _request_waterservices_data(
    station: str,
    parameter: str,
    start_date: str,
    end_date: str,
    data_type: str,
    proxy: Optional[Dict],
    timeout: Union[int, float],
) -> DataFrame:
    """
    Requests data from the legacy USGS WaterServices, which USGS is
    decommissioning in Q1 2027. Instantaneous dates are station local days.

    Parameters
    ----------
    station : str
        USGS station number
    parameter : str
        USGS parameter code
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    data_type : str
        'Daily' or 'Instantaneous'
    proxy : dict or None
        Proxy settings for the request
    timeout : int or float
        Timeout in seconds for the HTTP request

    Returns
    -------
    data : pandas.DataFrame
        Processed time series data
    """
    service = "dv" if data_type == "Daily" else "iv"
    params = {
        "format": "json",
        "sites": station,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": parameter,
        "siteStatus": "all",
    }
    if data_type == "Daily":
        params["statCd"] = "00003"
    url = f"https://waterservices.usgs.gov/nwis/{service}/"

    print(
        "Data request URL: ", requests.Request("GET", url, params=params).prepare().url
    )

    max_retries = 3
    retry_count = 0
    while True:
        try:
            response = requests.get(
                url=url, params=params, proxies=proxy, timeout=timeout, verify=True
            )
            break
        except requests.exceptions.SSLError as e:
            retry_count += 1
            if retry_count == max_retries:
                raise e
            print(
                f"SSL Error occurred, retrying... (Attempt {retry_count}/{max_retries})"
            )
    response.raise_for_status()

    return _read_usgs_json(json.loads(response.text), True)


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def request_usgs_data(
    station: str,
    parameter: str,
    start_date: str,
    end_date: str,
    options: Optional[Dict] = None,
) -> Union[DataFrame, xr.Dataset]:
    """
    Loads USGS data directly from the USGS Water Data Open Geospatial
    Consortium (OGC) APIs (https://api.waterdata.usgs.gov) using GET requests

    The request URL prints to the screen. Instantaneous requests longer than
    1100 days are split into multiple requests.

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
        - api_key: str or None
            USGS Water Data API key (https://api.waterdata.usgs.gov/signup/),
            which raises the API rate limit. If None, the USGS_API_KEY
            environment variable is used if set. Default = None
        - include_status: bool
            If True, include the 'approval_status' ('Approved' or 'Provisional')
            and 'qualifier' columns. Default = False
        - use_legacy_api: bool
            If True, request data from the legacy USGS WaterServices
            (https://waterservices.usgs.gov), which USGS is decommissioning in
            Q1 2027, instead of the OGC APIs. Deprecated and will be removed
            by 2027-01-31. The timezone and include_status options are not
            supported and api_key is ignored. Default = False
        - timezone: str
            Timezone of start_date and end_date for Instantaneous data, as an
            IANA name (e.g. 'America/Anchorage'), so that each date covers that
            local day. Daily values are already local calendar days, so this is
            not used for Daily data. The returned data is always indexed in
            UTC. Default = 'UTC'

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by UTC datetime with a column named according to the
        parameter's description (e.g. 'Discharge, cubic feet per second'), or
        the parameter code if the description cannot be looked up. If
        include_status is True, also includes the 'approval_status' and
        'qualifier' columns.
    """
    # Set default options
    options = options or {}
    data_type = options.get("data_type", "Daily")
    proxy = options.get("proxy", None)
    write_json = options.get("write_json", None)
    clear_cache = options.get("clear_cache", False)
    to_pandas = options.get("to_pandas", True)
    timeout = options.get("timeout", 30)  # 30 seconds default timeout
    api_key = options.get("api_key", None) or os.environ.get("USGS_API_KEY")
    include_status = options.get("include_status", False)
    timezone = options.get("timezone", "UTC")
    use_legacy_api = options.get("use_legacy_api", False)

    if data_type not in ["Daily", "Instantaneous"]:
        raise ValueError(f"data_type must be Daily or Instantaneous. Got: {data_type}")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number. Got: {timeout}")

    if api_key is not None and not isinstance(api_key, str):
        raise TypeError(f"api_key must be of type str. Got: {type(api_key)}")

    if not isinstance(include_status, bool):
        raise TypeError(
            f"include_status must be of type bool. Got: {type(include_status)}"
        )

    if not isinstance(timezone, str) or timezone not in zoneinfo.available_timezones():
        raise ValueError(
            "timezone must be an IANA timezone name (e.g. 'America/Anchorage'). "
            f"Got: {timezone}"
        )

    if not isinstance(use_legacy_api, bool):
        raise TypeError(
            f"use_legacy_api must be of type bool. Got: {type(use_legacy_api)}"
        )

    if use_legacy_api:
        if include_status or timezone != "UTC":
            raise ValueError(
                "The include_status and timezone options are not supported with "
                "use_legacy_api."
            )
        warnings.warn(
            "The use_legacy_api option is deprecated. USGS is decommissioning the "
            "legacy WaterServices (waterservices.usgs.gov) in Q1 2027 "
            "(https://waterdata.usgs.gov/blog/api-waterservices-decom/) and MHKiT "
            "will remove support by 2027-01-31. To use the new USGS Water Data "
            "APIs, remove use_legacy_api from the options passed to "
            "request_usgs_data.",
            FutureWarning,
            stacklevel=2,
        )

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "usgs")

    # Create a unique filename based on the function parameters
    # The "ogc" prefix keeps results cached from the legacy WaterServices, which
    # used station local time, from being returned
    hash_params = f"ogc_{station}_{parameter}_{start_date}_{end_date}_{data_type}"
    if use_legacy_api:
        hash_params = f"{station}_{parameter}_{start_date}_{end_date}_{data_type}"
    if include_status:
        hash_params += "_status"
    if data_type == "Instantaneous" and timezone != "UTC":
        hash_params += f"_{timezone}"

    cached_data, _, cache_filepath = handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": None, "metadata": None, "write_json": write_json},
        clear_cache_file=clear_cache,
    )

    if cached_data is not None:
        # The cache stores UTC timestamps without a timezone
        if cached_data.index.tz is None:
            cached_data.index = cached_data.index.tz_localize("UTC")
        if not to_pandas:
            cached_data = convert_to_dataset(cached_data)
        return cached_data

    # If no cached data, proceed with the API request
    if use_legacy_api:
        data = _request_waterservices_data(
            station, parameter, start_date, end_date, data_type, proxy, timeout
        )
    else:
        request_options = {
            "api_key": api_key,
            "include_status": include_status,
            "timezone": timezone,
        }
        data = _request_ogc_data(
            station,
            parameter,
            start_date,
            end_date,
            data_type,
            request_options,
            proxy,
            timeout,
        )

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
        data = convert_to_dataset(data)

    return data
