"""
This module provides functions for retrieving and processing data from the United States
Geological Survey (USGS) National Water Information System (NWIS). It enables access to
river flow data and related measurements useful for hydrokinetic resource assessment.

Requests go to the USGS Water Data APIs (https://api.waterdata.usgs.gov/ogcapi/v1/), which
replace the legacy WaterServices endpoints (waterservices.usgs.gov) that USGS is
decommissioning. Files saved from either service can still be read.

"""

from typing import Dict, Union, Optional
import os
import json
import shutil
import requests
import pandas as pd
import xarray as xr
from pandas import DataFrame
from mhkit.utils.cache import handle_caching
from mhkit.utils.type_handling import convert_to_dataset


WATER_DATA_API = "https://api.waterdata.usgs.gov/ogcapi/v1"


def _read_water_data_features(text: Dict) -> DataFrame:
    """
    Process a USGS Water Data API response (GeoJSON features) into a DataFrame
    with the same layout as the legacy WaterServices reader: a UTC datetime
    index and one numeric column per parameter, named by its description.

    Parameters
    ----------
    text : dict
        GeoJSON FeatureCollection from the Water Data API. The parameter
        description used for the column name is read from
        ``text["parameter_description"]`` when present, else the parameter code.

    Returns
    -------
    data : pandas.DataFrame
        Processed time series data
    """
    rows = [feature["properties"] for feature in text.get("features", [])]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    name = text.get("parameter_description") or str(frame["parameter_code"].iloc[0])
    data = pd.DataFrame(
        {name: pd.to_numeric(frame["value"], errors="coerce").values},
        index=pd.to_datetime(frame["time"], utc=True),
    )
    data.index.name = None
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data


def _read_usgs_json(text: Dict, to_pandas: bool = True) -> Union[DataFrame, xr.Dataset]:
    """
    Process USGS JSON response into a pandas DataFrame or xarray Dataset.

    Parameters
    ----------
    text : dict
        JSON response from USGS API containing time series data, either
        legacy WaterServices JSON or Water Data API GeoJSON
    to_pandas : bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data : pandas.DataFrame or xarray.Dataset
        Processed time series data
    """
    if "features" in text:
        data = _read_water_data_features(text)
        return data if to_pandas else convert_to_dataset(data)

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
        data = convert_to_dataset(data)

    return data


def read_usgs_file(
    file_name: str, to_pandas: bool = True
) -> Union[DataFrame, xr.Dataset]:
    """
    Reads a USGS JSON data file: legacy WaterServices JSON (from
    https://waterdata.usgs.gov/nwis), or Water Data API GeoJSON saved from
    https://api.waterdata.usgs.gov/ogcapi/v1/

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
    station: str,
    parameter: str,
    start_date: str,
    end_date: str,
    options: Optional[Dict] = None,
) -> Union[DataFrame, xr.Dataset]:
    """
    Loads USGS data from the USGS Water Data APIs
    (https://api.waterdata.usgs.gov/ogcapi/v1/) using GET requests: the
    ``daily`` collection for 'Daily' and ``continuous`` for 'Instantaneous'.

    The request URLs print to the screen. Instantaneous requests are split
    into one query per year, as the ``continuous`` collection serves at most
    three years per query.

    Parameters
    ----------
    station : str
        USGS station number (e.g. '08313000')
    parameter : str
        USGS parameter ID (e.g. '00060' for Discharge, cubic feet per second)
    start_date : str
        Start date in the format 'YYYY-MM-DD' (e.g. '2018-01-01')
    end_date : str
        End date in the format 'YYYY-MM-DD' (e.g. '2018-12-31'). Both dates
        are inclusive. For 'Instantaneous' data they bound the request in UTC;
        the legacy WaterServices endpoint read them in the site's local time.
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
            USGS Water Data API key for higher rate limits
            (https://api.waterdata.usgs.gov/signup/). Default = the
            API_USGS_PAT environment variable, else None

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
    text = _request_water_data(
        station,
        parameter,
        start_date,
        end_date,
        {
            "data_type": data_type,
            "proxy": proxy,
            "timeout": timeout,
            "api_key": options.get("api_key", os.environ.get("API_USGS_PAT")),
        },
    )

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
        data = convert_to_dataset(data)

    return data


def _get_json(url: str, params: Optional[Dict], proxy, timeout) -> Dict:
    """GET a URL and parse JSON, retrying SSL errors up to three times."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url=url, params=params, proxies=proxy, timeout=timeout, verify=True
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError:
            if attempt == max_retries:
                raise
            print(f"SSL Error occurred, retrying... (Attempt {attempt}/{max_retries})")
    raise RuntimeError("unreachable")


def _year_windows(start_date: str, end_date: str):
    """Split [start_date, end_date] into consecutive windows of at most one year."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    while start < end:
        stop = min(start + pd.DateOffset(years=1), end)
        yield start, stop
        start = stop


def _request_water_data(
    station: str, parameter: str, start_date: str, end_date: str, settings: Dict
) -> Dict:
    """
    Request all features for one station and parameter from the Water Data
    API, following "next" links, and return them as one FeatureCollection with
    the parameter's description attached for column naming.
    """
    proxy, timeout, api_key = (
        settings["proxy"],
        settings["timeout"],
        settings["api_key"],
    )
    base = {
        "f": "json",
        "monitoring_location_id": f"USGS-{station}",
        "parameter_code": parameter,
        "limit": 50000,
    }
    if api_key:
        base["api_key"] = api_key

    if settings["data_type"] == "Daily":
        collection = "daily"
        windows = [(start_date, end_date)]
        base["statistic_id"] = "00003"
    else:
        collection = "continuous"
        windows = [
            (a.strftime("%Y-%m-%dT%H:%M:%SZ"), b.strftime("%Y-%m-%dT%H:%M:%SZ"))
            for a, b in _year_windows(start_date, end_date)
        ]
    url = f"{WATER_DATA_API}/collections/{collection}/items"

    features = []
    for begin, finish in windows:
        params = dict(base, time=f"{begin}/{finish}")
        print(
            "Data request URL: ",
            requests.Request("GET", url, params=params).prepare().url,
        )
        page = _get_json(url, params, proxy, timeout)
        features += page.get("features", [])
        next_link = _next_link(page)
        while next_link:
            if api_key and "api_key=" not in next_link:
                next_link += f"&api_key={api_key}"
            page = _get_json(next_link, None, proxy, timeout)
            features += page.get("features", [])
            next_link = _next_link(page)

    description = None
    try:
        code = _get_json(
            f"{WATER_DATA_API}/collections/parameter-codes/items/{parameter}",
            {"f": "json"},
            proxy,
            timeout,
        )
        description = code.get("properties", {}).get("parameter_description")
    except (requests.exceptions.RequestException, ValueError):
        description = None

    return {
        "type": "FeatureCollection",
        "features": features,
        "parameter_description": description,
    }


def _next_link(page: Dict) -> Optional[str]:
    """Return the href of a page's "next" link, if any."""
    for link in page.get("links", []):
        if link.get("rel") == "next":
            return link.get("href")
    return None
