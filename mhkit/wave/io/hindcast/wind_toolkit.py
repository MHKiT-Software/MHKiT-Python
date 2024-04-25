"""
Wind Toolkit Data Utility Functions
===================================

This module contains a collection of utility functions designed to facilitate 
the extraction, caching, and visualization of wind data from the WIND Toolkit 
hindcast dataset hosted on AWS. This dataset includes offshore wind hindcast data 
with various parameters like wind speed, direction, temperature, and pressure.

Key Functions:
--------------
- `region_selection`: Determines which predefined wind region a given latitude 
  and longitude fall within.
  
- `get_region_data`: Retrieves latitude and longitude data points for a specified 
  wind region. Uses caching to speed up repeated requests.
  
- `plot_region`: Plots the geographical extent of a specified wind region and 
  can overlay a given latitude-longitude point.
  
- `elevation_to_string`: Converts a parameter (e.g., 'windspeed') and elevation 
  values (e.g., [20, 40, 120]) to the formatted strings used in the WIND Toolkit.
  
- `request_wtk_point_data`: Fetches specified wind data parameters for given 
  latitude-longitude points and years from the WIND Toolkit hindcast dataset. 
  Supports caching for faster repeated data retrieval.

Dependencies:
-------------
- rex: Library to handle renewable energy datasets.
- pandas: Data manipulation and analysis.
- os, hashlib, pickle: Used for caching functionality.
- matplotlib: Used for plotting.

Notes:
------
- To access the WIND Toolkit hindcast data, users need to configure `h5pyd` 
  for data access on HSDS (see the metocean_example or WPTO_hindcast_example
  notebook for more details).
  
- While some functions perform basic checks (e.g., verifying that latitude 
  and longitude are within a predefined region), it's essential to understand 
  the boundaries of each region and the available parameters and elevations in the dataset.

Author: 
-------
akeeste
ssolson

Date:
-----
2023-09-26

"""

import os
import hashlib
import pickle
import pandas as pd

from rex import MultiYearWindX
import matplotlib.pyplot as plt
from mhkit.utils.cache import handle_caching
from mhkit.utils.type_handling import convert_to_dataset


def region_selection(lat_lon, preferred_region=""):
    """
    Returns the name of the predefined region in which the given coordinates reside.
    Can be used to check if the passed lat/lon pair is within the WIND Toolkit hindcast dataset.

    Parameters
    ----------
    lat_lon : tuple
        Latitude and longitude coordinates as floats or integers

    preferred_region : string (optional)
        Latitude and longitude coordinates as floats or integers

    Returns
    -------
    region : string
        Name of predefined region for given coordinates
    """
    if not isinstance(lat_lon, tuple):
        raise TypeError(f"lat_lon must be of type tuple, got {type(lat_lon).__name__}")

    if len(lat_lon) != 2:
        raise ValueError(f"lat_lon must be of length 2, got length {len(lat_lon)}")

    if not isinstance(lat_lon[0], (float, int)):
        raise TypeError(
            f"lat_lon values must be floats or ints, got {type(lat_lon[0]).__name__}"
        )

    if not isinstance(lat_lon[1], (float, int)):
        raise TypeError(
            f"lat_lon values must be floats or ints, got {type(lat_lon[1]).__name__}"
        )

    if not isinstance(preferred_region, str):
        raise TypeError(
            f"preferred_region must be a string, got {type(preferred_region).__name__}"
        )

    # Note that this check is fast, but not robust because region are not
    # rectangular on a lat-lon grid
    rDict = {
        "CA_NWP_overlap": {"lat": [41.213, 42.642], "lon": [-129.090, -121.672]},
        "Offshore_CA": {"lat": [31.932, 42.642], "lon": [-129.090, -115.806]},
        "Hawaii": {"lat": [15.565, 26.221], "lon": [-164.451, -151.278]},
        "NW_Pacific": {"lat": [41.213, 49.579], "lon": [-130.831, -121.672]},
        "Mid_Atlantic": {"lat": [37.273, 42.211], "lon": [-76.427, -64.800]},
    }

    def region_search(x):
        return all(
            (
                True if rDict[x][dk][0] <= d <= rDict[x][dk][1] else False
                for dk, d in {"lat": lat_lon[0], "lon": lat_lon[1]}.items()
            )
        )

    region = [key for key in rDict if region_search(key)]

    if region[0] == "CA_NWP_overlap":
        if preferred_region == "Offshore_CA":
            region[0] = "Offshore_CA"
        elif preferred_region == "NW_Pacific":
            region[0] = "NW_Pacific"
        else:
            raise TypeError(
                f"Preferred_region ({preferred_region}) must be 'Offshore_CA' or 'NW_Pacific' when lat_lon {lat_lon} falls in the overlap region"
            )

    if len(region) == 0:
        raise TypeError(f"Coordinates {lat_lon} out of bounds. Must be within {rDict}")
    else:
        return region[0]


def get_region_data(region):
    """
    Retrieves the latitude and longitude data points for the specified region
    from the cache if available; otherwise, fetches the data and caches it for
    subsequent calls.

    The function forms a unique identifier from the `region` parameter and checks
    whether the corresponding data is available in the cache. If the data is found,
    it's loaded and returned. If not, the data is fetched, cached, and then returned.

    Parameters
    ----------
    region : str
        Name of the predefined region in the WIND Toolkit for which to
        retrieve latitude and longitude data points. It is case-sensitive.
        Examples: 'Offshore_CA','Hawaii','Mid_Atlantic','NW_Pacific'

    Returns
    -------
    lats : numpy.ndarray
        A 1D array containing the latitude coordinates of data points
        in the specified region.

    lons : numpy.ndarray
        A 1D array containing the longitude coordinates of data points
        in the specified region.

    Example
    -------
    >>> lats, lons = get_region_data('Offshore_CA')
    """
    if not isinstance(region, str):
        raise TypeError("region must be of type string")
    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "hindcast")

    # Create a unique identifier for this function call
    hash_id = hashlib.md5(region.encode()).hexdigest()

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create a path to the cache file for this function call
    cache_file = os.path.join(cache_dir, f"{hash_id}.pkl")

    if os.path.isfile(cache_file):
        # If the cache file exists, load the data from the cache
        with open(cache_file, "rb") as f:
            lats, lons = pickle.load(f)
        return lats, lons
    else:
        wind_path = "/nrel/wtk/" + region.lower() + "/" + region + "_*.h5"
        windKwargs = {
            "tree": None,
            "unscale": True,
            "str_decode": True,
            "hsds": True,
            "years": [2019],
        }

        # Get the latitude and longitude list from the region in rex
        rex_wind = MultiYearWindX(wind_path, **windKwargs)
        lats = rex_wind.lat_lon[:, 0]
        lons = rex_wind.lat_lon[:, 1]

        # Save data to cache
        with open(cache_file, "wb") as f:
            pickle.dump((lats, lons), f)

        return lats, lons


def plot_region(region, lat_lon=None, ax=None):
    """
    Visualizes the area that a given region covers. Can help users understand
    the extent of a region since they are not all rectangular.

    Parameters
    ----------
    region : string
        Name of predefined region in the WIND Toolkit
        Options: 'Offshore_CA','Hawaii','Mid_Atlantic','NW_Pacific'
    lat_lon : couple (optional)
        Latitude and longitude pair to plot on top of the chosen region. Useful
        to inform accurate latitude-longitude selection for data analysis.
    ax : matplotlib axes object (optional)
        Axes for plotting.  If None, then a new figure is created.

    Returns
    ---------
    ax : matplotlib pyplot axes
    """
    if not isinstance(region, str):
        raise TypeError("region must be of type string")

    supported_regions = ["Offshore_CA", "Hawaii", "Mid_Atlantic", "NW_Pacific"]
    if region not in supported_regions:
        raise ValueError(
            f'{region} not in list of supported regions: {", ".join(supported_regions)}'
        )

    lats, lons = get_region_data(region)

    # Plot the latitude longitude pairs
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(lons, lats, "o", label=f"{region} region")
    if lat_lon is not None:
        ax.plot(lat_lon[1], lat_lon[0], "o", label="Specified lat-lon point")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid()
    ax.set_title(f"Extent of the WIND Toolkit {region} region")
    ax.legend()

    return ax


def elevation_to_string(parameter, elevations):
    """
    Takes in a parameter (e.g. 'windspeed') and elevations (e.g. [20, 40, 120])
    and returns the formatted strings that are input to WIND Toolkit (e.g. windspeed_10m).
    Does not check parameter against the elevation levels. This is done in request_wtk_point_data.

    Parameters
    ----------
    parameter: string
        Name of the WIND toolkit parameter.
        Options: 'windspeed', 'winddirection', 'temperature', 'pressure'
    elevations : list
        List of elevations (float).
        Values can range from approxiamtely 20 to 200 in increments of 20, depending
        on the parameter in question. See Documentation for request_wtk_point_data
        for the full list of available parameters.

    Returns
    ---------
    parameter_list: list
        Formatted List of WIND Toolkit parameter strings

    """

    if not isinstance(parameter, str):
        raise TypeError(f"parameter must be a string, got {type(parameter)}")

    if not isinstance(elevations, (float, list)):
        raise TypeError(f"elevations must be a float or list, got {type(elevations)}")

    if parameter not in ["windspeed", "winddirection", "temperature", "pressure"]:
        raise ValueError(f"Invalid parameter: {parameter}")

    parameter_list = []
    for e in elevations:
        parameter_list.append(parameter + "_" + str(e) + "m")

    return parameter_list


def request_wtk_point_data(
    time_interval,
    parameter,
    lat_lon,
    years,
    preferred_region="",
    tree=None,
    unscale=True,
    str_decode=True,
    hsds=True,
    clear_cache=False,
    to_pandas=True,
):
    """
    Returns data from the WIND Toolkit offshore wind hindcast hosted on
    AWS at the specified latitude and longitude point(s), or the closest
    available point(s).Visit https://registry.opendata.aws/nrel-pds-wtk/
    for more information about the dataset and available locations and years.

    Calls with multiple parameters must have the same time interval. Calls
    with multiple locations must use the same region (use the plot_region function).

    Note: To access the WIND Toolkit hindcast data, you will need to
    configure h5pyd for data access on HSDS. Please see the
    metocean_example or WPTO_hindcast_example notebook for more information.

    Parameters
    ----------
    time_interval : string
        Data set type of interest
        Options: '1-hour' '5-minute'
    parameter : string or list of strings
        Dataset parameter to be downloaded. Other parameters may be available.
        This list is limited to those available at both 5-minute and 1-hour
        time intervals for all regions.
        Options:
            'precipitationrate_0m', 'inversemoninobukhovlength_2m',
            'relativehumidity_2m', 'surface_sea_temperature',
            'pressure_0m', 'pressure_100m', 'pressure_200m',
            'temperature_10m', 'temperature_20m', 'temperature_40m',
            'temperature_60m', 'temperature_80m', 'temperature_100m',
            'temperature_120m', 'temperature_140m', 'temperature_160m',
            'temperature_180m', 'temperature_200m',
            'winddirection_10m', 'winddirection_20m', 'winddirection_40m',
            'winddirection_60m', 'winddirection_80m', 'winddirection_100m',
            'winddirection_120m', 'winddirection_140m', 'winddirection_160m',
            'winddirection_180m', 'winddirection_200m',
            'windspeed_10m', 'windspeed_20m', 'windspeed_40m',
            'windspeed_60m', 'windspeed_80m', 'windspeed_100m',
            'windspeed_120m', 'windspeed_140m', 'windspeed_160m',
            'windspeed_180m', 'windspeed_200m'
    lat_lon : tuple or list of tuples
        Latitude longitude pairs at which to extract data. Use plot_region() or
        region_selection() to see the corresponding region for a given location.
    years : list
        Year(s) to be accessed. The years 2000-2019 available (up to 2020
        for Mid-Atlantic). Examples: [2015] or [2004,2006,2007]
    preferred_region : string (optional)
        Region that the lat_lon belongs to ('Offshore_CA' or 'NW_Pacific').
        Required when a lat_lon point falls in both the Offshore California
        and NW Pacific regions. Overlap region defined by
        latitude = (41.213, 42.642) and longitude = (-129.090, -121.672).
        Default = ''
    tree : str | cKDTree (optional)
        cKDTree or path to .pkl file containing pre-computed tree
        of lat, lon coordinates, default = None
    unscale : bool (optional)
        Boolean flag to automatically unscale variables on extraction
        Default = True
    str_decode : bool (optional)
        Boolean flag to decode the bytestring meta data into normal
        strings. Setting this to False will speed up the meta data read.
        Default = True
    hsds : bool (optional)
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS. Setting to False will indicate to look for files on
        local machine, not AWS. Default = True
    clear_cache : bool (optional)
        Boolean flag to clear the cache related to this specific request.
        Default is False.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    data: DataFrame
        Data indexed by datetime with columns named for parameter and
        cooresponding metadata index
    meta: DataFrame
        Location metadata for the requested data location
    """

    if not isinstance(parameter, (str, list)):
        raise TypeError("parameter must be of type string or list")
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError("lat_lon must be of type list or tuple")
    if not isinstance(time_interval, str):
        raise TypeError("time_interval must be a string")
    if not isinstance(years, list):
        raise TypeError("years must be a list")
    if not isinstance(preferred_region, str):
        raise TypeError("preferred_region must be a string")
    if not isinstance(tree, (str, type(None))):
        raise TypeError("tree must be a string or None")
    if not isinstance(unscale, bool):
        raise TypeError("unscale must be bool type")
    if not isinstance(str_decode, bool):
        raise TypeError("str_decode must be bool type")
    if not isinstance(hsds, bool):
        raise TypeError("hsds must be bool type")
    if not isinstance(clear_cache, bool):
        raise TypeError("clear_cache must be of type bool")

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "hindcast")

    # Construct a string representation of the function parameters
    hash_params = f"{time_interval}_{parameter}_{lat_lon}_{years}_{preferred_region}_{tree}_{unscale}_{str_decode}_{hsds}"

    # Use handle_caching to manage caching.
    data, meta, _ = handle_caching(hash_params, cache_dir, clear_cache_file=clear_cache)

    if data is not None and meta is not None:
        if not to_pandas:
            data = convert_to_dataset(data)
            data.attrs = meta

        return data, meta  # Return cached data and meta if available
    else:
        # check for multiple region selection
        if isinstance(lat_lon[0], float):
            region = region_selection(lat_lon, preferred_region)
        else:
            reglist = []
            for loc in lat_lon:
                reglist.append(region_selection(loc))
            if reglist.count(reglist[0]) == len(lat_lon):
                region = reglist[0]
            else:
                raise TypeError("Coordinates must be within the same region!")

        if time_interval == "1-hour":
            wind_path = f"/nrel/wtk/{region.lower()}/{region}_*.h5"
        elif time_interval == "5-minute":
            wind_path = f"/nrel/wtk/{region.lower()}-5min/{region}_*.h5"
        else:
            raise TypeError(
                f"Invalid time_interval '{time_interval}', must be '1-hour' or '5-minute'"
            )
        windKwargs = {
            "tree": tree,
            "unscale": unscale,
            "str_decode": str_decode,
            "hsds": hsds,
            "years": years,
        }
        data_list = []
        with MultiYearWindX(wind_path, **windKwargs) as rex_wind:
            if isinstance(parameter, list):
                for p in parameter:
                    temp_data = rex_wind.get_lat_lon_df(p, lat_lon)
                    col = temp_data.columns[:]
                    for i, c in zip(range(len(col)), col):
                        temp = f"{p}_{i}"
                        temp_data = temp_data.rename(columns={c: temp})

                    data_list.append(temp_data)
                data = pd.concat(data_list, axis=1)

            else:
                data = rex_wind.get_lat_lon_df(parameter, lat_lon)
                col = data.columns[:]

                for i, c in zip(range(len(col)), col):
                    temp = f"{parameter}_{i}"
                    data = data.rename(columns={c: temp})

            meta = rex_wind.meta.loc[col, :]
            meta = meta.reset_index(drop=True)

        # Save the retrieved data and metadata to cache.
        handle_caching(hash_params, cache_dir, data=data, metadata=meta)

        if not to_pandas:
            data = convert_to_dataset(data)
            data.attrs = meta

        return data, meta
