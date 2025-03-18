"""
noaa.py

This module provides functions to fetch, process, and read NOAA (National
Oceanic and Atmospheric Administration) current data directly from the
NOAA Tides and Currents API (https://api.tidesandcurrents.noaa.gov/api/prod/). It
supports loading data into a pandas DataFrame, handling data in XML and
JSON formats, and writing data to a JSON file.

Functions:
----------
request_noaa_data(station, parameter, start_date, end_date, options=None):
    Loads NOAA current data from the API into a pandas DataFrame,
    with optional support for proxy settings and writing data to a JSON
    file.

_xml_to_dataframe(response):
    Converts NOAA response data in XML format into a pandas DataFrame
    and returns metadata.

read_noaa_json(filename):
    Reads a JSON file containing NOAA data saved from the request_noaa_data
    function and returns a DataFrame with timeseries site data and metadata.
"""

import os
import xml.etree.ElementTree as ET
import datetime
import json
import math
import shutil
import warnings
import pandas as pd
import requests
from mhkit.utils.cache import handle_caching


def request_noaa_data(
    station: str,
    parameter: str,
    start_date: str,
    end_date: str,
    options: dict = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Loads NOAA current data directly from https://api.tidesandcurrents.noaa.gov/api/prod/
    into a pandas DataFrame. NOAA sets max of 31 days between start and end date.
    See https://api.tidesandcurrents.noaa.gov/api/prod/ for options. All times are reported as
    GMT and metric units are returned for data. Uses cached data if available.

    The request URL prints to the screen.

    Parameters
    ----------
    station : str
        NOAA current station number (e.g. 'cp0101', "s08010", "9446484")
    parameter : str
        NOAA parameter (e.g. "currents", "salinity", "water_level", "water_temperature",
        "air_temperature", "wind", "air_pressure")
        https://api.tidesandcurrents.noaa.gov/api/prod/
    start_date : str
        Start date in the format yyyyMMdd
    end_date : str
        End date in the format yyyyMMdd
    options : dict, optional
        Dictionary containing optional parameters:
        - proxy: dict or None
            Proxy settings for the request.
        - write_json: str or None
            Path to write the data as a JSON file.
        - clear_cache: bool
            Whether to clear cached data.
        - to_pandas: bool
            Whether to return the data as a pandas DataFrame.

    Returns
    -------
    data : pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named according to the parameter's
        variable description
    metadata : dict or None
        Request metadata. If returning xarray, metadata is instead attached to
        the data's attributes.
    """
    if kwargs:
        warnings.warn(
            f"Unexpected keyword arguments: {', '.join(kwargs.keys())}. "
            "Please pass options as a dictionary.",
            UserWarning,
        )

    options = options or {}
    proxy = options.get("proxy", None)
    write_json = options.get("write_json", None)
    clear_cache = options.get("clear_cache", False)
    to_pandas = options.get("to_pandas", True)

    _validate_inputs(
        station,
        parameter,
        start_date,
        end_date,
        options,
    )

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "noaa")
    hash_params = f"{station}_{parameter}_{start_date}_{end_date}"

    cached_data, cached_metadata, cache_filepath = handle_caching(
        hash_params,
        cache_dir,
        {"data": None, "metadata": None, "write_json": write_json},
        clear_cache,
    )

    if cached_data is not None:
        return _handle_cached_data(
            cached_data, cached_metadata, write_json, cache_filepath, to_pandas
        )

    return _fetch_noaa_data(
        station,
        parameter,
        start_date,
        end_date,
        {
            "proxy": proxy,
            "cache_dir": cache_dir,
            "hash_params": hash_params,
            "write_json": write_json,
            "clear_cache": clear_cache,
            "to_pandas": to_pandas,
        },
    )


def _validate_inputs(
    station: str, parameter: str, start_date: str, end_date: str, options: dict
) -> None:
    if not isinstance(station, str):
        raise TypeError(
            f"Expected 'station' to be of type str, but got {type(station)}"
        )
    if not isinstance(parameter, str):
        raise TypeError(
            f"Expected 'parameter' to be of type str, but got {type(parameter)}"
        )
    if not isinstance(start_date, str):
        raise TypeError(
            f"Expected 'start_date' to be of type str, but got {type(start_date)}"
        )
    if not isinstance(end_date, str):
        raise TypeError(
            f"Expected 'end_date' to be of type str, but got {type(end_date)}"
        )

    proxy = options.get("proxy", None)
    write_json = options.get("write_json", None)
    clear_cache = options.get("clear_cache", False)
    to_pandas = options.get("to_pandas", True)

    if proxy and not isinstance(proxy, dict):
        raise TypeError(
            f"Expected 'proxy' to be of type dict or None, but got {type(proxy)}"
        )
    if write_json and not isinstance(write_json, str):
        raise TypeError(
            f"Expected 'write_json' to be of type str or None, but got {type(write_json)}"
        )
    if not isinstance(clear_cache, bool):
        raise TypeError(
            f"Expected 'clear_cache' to be of type bool, but got {type(clear_cache)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")


def _handle_cached_data(
    cached_data: pd.DataFrame,
    cached_metadata: dict,
    write_json: str,
    cache_filepath: str,
    to_pandas: bool,
) -> tuple[pd.DataFrame, dict]:
    """
    Handles cached data by optionally writing it to a JSON file and returning it.

    Parameters
    ----------
    cached_data : pd.DataFrame
        The cached data to be returned.
    cached_metadata : dict
        Metadata associated with the cached data.
    write_json : str
        Path to write the cached data as a JSON file, if specified.
    cache_filepath : str
        Filepath of the cached data.
    to_pandas : bool
        Flag indicating whether to return the data as a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The cached data and its metadata.
    """
    if write_json:
        shutil.copy(cache_filepath, write_json)
    if to_pandas:
        return cached_data, cached_metadata

    cached_data = cached_data.to_xarray()
    cached_data.attrs = cached_metadata
    return cached_data


def _fetch_noaa_data(
    station: str, parameter: str, start_date: str, end_date: str, options: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Fetches NOAA data from the API, processes it, and returns it along with metadata.

    Parameters
    ----------
    station : str
        NOAA current station number.
    parameter : str
        NOAA parameter to fetch.
    start_date : str
        Start date for data retrieval in yyyyMMdd format.
    end_date : str
        End date for data retrieval in yyyyMMdd format.
    options : dict
        Dictionary of options for data retrieval:
        - proxy: dict or None
            Proxy settings for the request.
        - cache_dir: str
            Directory for caching data.
        - hash_params: str
            Parameters used for caching.
        - write_json: str or None
            Path to write the data as a JSON file.
        - clear_cache: bool
            Whether to clear cached data.
        - to_pandas: bool
            Whether to return the data as a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The fetched data and its metadata.
    """
    begin, end = _parse_dates(start_date, end_date)
    date_list = _create_date_ranges(begin, end)

    data_frames = []
    metadata = None  # Initialize metadata
    for i in range(len(date_list) - 1):
        start_date = date_list[i].strftime("%Y%m%d")
        end_date = date_list[i + 1].strftime("%Y%m%d")
        data_url = _build_data_url(station, parameter, start_date, end_date)

        print(f"Data request URL: {data_url}\n")
        response = _make_request(data_url, options["proxy"])
        df, metadata = _xml_to_dataframe(response)
        data_frames.append(df)

    return _process_data_frames(data_frames, metadata, options)


def _process_data_frames(
    data_frames: list[pd.DataFrame], metadata: dict, options: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Processes a list of data frames by concatenating them and handling caching.

    Parameters
    ----------
    data_frames : list[pd.DataFrame]
        List of data frames to process.
    metadata : dict
        Metadata associated with the data.
    options : dict
        Options for processing, including caching and output format:
        - hash_params: str
            Parameters used for caching.
        - cache_dir: str
            Directory for caching data.
        - write_json: str or None
            Path to write the data as a JSON file.
        - clear_cache: bool
            Whether to clear cached data.
        - to_pandas: bool
            Whether to return the data as a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The processed data and its metadata.
    """
    data = _concatenate_data_frames(data_frames)
    cache_filepath = handle_caching(
        options["hash_params"],
        options["cache_dir"],
        {"data": data, "metadata": metadata, "write_json": None},
        options["clear_cache"],
    )

    if options["write_json"]:
        shutil.copy(cache_filepath, options["write_json"])

    if options["to_pandas"]:
        return data, metadata

    data = data.to_xarray()
    data.attrs = metadata
    return data


def _parse_dates(start_date: str, end_date: str) -> tuple[datetime.date, datetime.date]:
    begin = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    return begin, end


def _create_date_ranges(
    begin: datetime.date, end: datetime.date
) -> list[datetime.date]:
    delta = 30
    interval = math.ceil(((end - begin).days) / delta)
    date_list = [
        begin + datetime.timedelta(days=i * delta) for i in range(interval + 1)
    ]
    date_list[-1] = end
    return date_list


def _build_data_url(
    station: str, parameter: str, start_date: str, end_date: str
) -> str:
    api_query = (
        f"begin_date={start_date}&end_date={end_date}&station={station}&product={parameter}"
        "&units=metric&time_zone=gmt&application=web_services&format=xml"
    )
    if parameter == "water_level":
        api_query += "&datum=MLLW"
    return f"https://tidesandcurrents.noaa.gov/api/datagetter?{api_query}"


def _make_request(data_url: str, proxy: dict) -> requests.Response:
    try:
        response = requests.get(url=data_url, proxies=proxy, timeout=60)
        response.raise_for_status()
        if "error" in response.content.decode():
            raise requests.exceptions.RequestException(response.content.decode())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Error message: {response.content.decode()}\n")
        raise
    except requests.exceptions.RequestException as req_err:
        print(f"Requests error occurred: {req_err}")
        print(f"Error message: {response.content.decode()}\n")
        raise
    return response


def _concatenate_data_frames(data_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates a list of data frames into a single data frame, removing duplicates.

    Parameters
    ----------
    data_frames : list[pd.DataFrame]
        List of data frames to concatenate.

    Returns
    -------
    pd.DataFrame
        The concatenated data frame with duplicates removed.
    """
    if data_frames:
        data = pd.concat(data_frames, ignore_index=False)
    else:
        raise ValueError("No data retrieved.")
    return data.loc[~data.index.duplicated()]


def _xml_to_dataframe(response: requests.Response) -> tuple[pd.DataFrame, dict]:
    """
    Returns a dataframe from an xml response
    """
    root = ET.fromstring(response.text)
    metadata = None
    data = None

    for child in root:
        if child.tag == "metadata":
            metadata = child.attrib
        elif child.tag == "observations":
            data = child
        elif child.tag == "error":
            print("***ERROR: Response returned error")
            return None, {}

    if data is None:
        print("***ERROR: No observations found")
        return None, {}

    df = pd.concat(
        [pd.DataFrame(obs.attrib, index=[0]) for obs in data], ignore_index=True
    )

    df["t"] = pd.to_datetime(df.t)
    df = df.set_index("t")
    df.drop_duplicates(inplace=True)

    cols = list(df.columns)
    for var in cols:
        try:
            df[var] = df[var].apply(pd.to_numeric)
        except ValueError:
            pass

    return df, metadata or {}


def read_noaa_json(filename: str, to_pandas: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Returns site DataFrame and metadata from a json saved from the
    request_noaa_data
    Parameters
    ----------
    filename: string
        filename with path of json file to load
    to_pandas : bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    data: DataFrame
        Timeseries Site data of direction and speed
    metadata : dictionary or None
        Site metadata. If returning xarray, metadata is instead attached to
        the data's attributes.
    """
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    with open(filename, encoding="utf-8") as outfile:
        json_data = json.load(outfile)
    try:  # original MHKiT format (deprecate in future)
        # Get the metadata
        metadata = json_data["metadata"]
        # Remove metadata entry
        del json_data["metadata"]
        # Remainder is DataFrame
        data = pd.DataFrame.from_dict(json_data)
        # Convert from epoch to date time
        data.index = pd.to_datetime(data.index, unit="ms")

    except ValueError:  # using cache.py format
        if "metadata" in json_data:
            metadata = json_data.pop("metadata", None)
        data = pd.DataFrame(
            json_data["data"],
            index=pd.to_datetime(json_data["index"]),
            columns=json_data["columns"],
        )

    if to_pandas:
        return data, metadata

    data = data.to_xarray()
    data.attrs = metadata
    return data
