"""
This module provides functions to access and process WPTO wave hindcast data
hosted on AWS at specified latitude and longitude points or the closest
available points. It includes functions to retrieve data for predefined
regions, request point data for various parameters, and request directional
spectrum data.

Functions:
    - region_selection(lat_lon): Returns the name of the predefined region for
      given latitude and longitude coordinates.
    - request_wpto_point_data(data_type, parameter, lat_lon, years, tree=None,
      unscale=True, str_decode=True, hsds=True): Returns data from the WPTO wave
      hindcast hosted on AWS at the specified latitude and longitude point(s) for
      the requested data type, parameter, and years.
    - request_wpto_directional_spectrum(lat_lon, year, tree=None, unscale=True,
      str_decode=True, hsds=True): Returns directional spectra data from the WPTO
      wave hindcast hosted on AWS at the specified latitude and longitude point(s)
      for the given year.

Dependencies:
    - sys
    - time.sleep
    - pandas
    - xarray
    - numpy
    - rex.MultiYearWaveX, rex.WaveX

Author: rpauly, aidanbharath, ssolson
Date: 2023-09-26
"""

import os
import sys
from time import sleep
import pandas as pd
import xarray as xr
import numpy as np
from rex import MultiYearWaveX, WaveX
from mhkit.utils.cache import handle_caching
from mhkit.utils.type_handling import convert_to_dataset


def region_selection(lat_lon):
    """
    Returns the name of the predefined region in which the given
    coordinates reside. Can be used to check if the passed lat/lon
    pair is within the WPTO hindcast dataset.

    Parameters
    ----------
    lat_lon : list or tuple
        Latitude and longitude coordinates as floats or integers

    Returns
    -------
    region : string
        Name of predefined region for given coordinates
    """
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")

    if not all(isinstance(coord, (float, int)) for coord in lat_lon):
        raise TypeError(
            f"lat_lon values must be of type float or int. Got: {type(lat_lon[0])}"
        )

    regions = {
        "Hawaii": {"lat": [15.0, 27.000002], "lon": [-164.0, -151.0]},
        "West_Coast": {"lat": [30.0906, 48.8641], "lon": [-130.072, -116.899]},
        "Atlantic": {"lat": [24.382, 44.8247], "lon": [-81.552, -65.721]},
    }

    def region_search(lat_lon, region, regions):
        return all(
            regions[region][dk][0] <= d <= regions[region][dk][1]
            for dk, d in {"lat": lat_lon[0], "lon": lat_lon[1]}.items()
        )

    region = [region for region in regions if region_search(lat_lon, region, regions)]

    if not region:
        raise ValueError("ERROR: coordinates out of bounds.")

    return region[0]


def request_wpto_point_data(
    data_type,
    parameter,
    lat_lon,
    years,
    tree=None,
    unscale=True,
    str_decode=True,
    hsds=True,
    path=None,
    to_pandas=True,
):
    """
    Returns data from the WPTO wave hindcast hosted on AWS at the
    specified latitude and longitude point(s), or the closest
    available point(s).
    Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more
    information about the dataset and available locations and years.

    Note: To access the WPTO hindcast data, you will need to configure
    h5pyd for data access on HSDS. Please see the WPTO_hindcast_example
    notebook for setup instructions.

    Parameters
    ----------
    data_type : string
        Data set type of interest
        Options: '3-hour' '1-hour'
    parameter : string or list of strings
        Dataset parameter to be downloaded
        3-hour dataset options: 'directionality_coefficient',
            'energy_period', 'maximum_energy_direction'
            'mean_absolute_period', 'mean_zero-crossing_period',
            'omni-directional_wave_power', 'peak_period'
            'significant_wave_height', 'spectral_width', 'water_depth'
        1-hour dataset options: 'directionality_coefficient',
            'energy_period', 'maximum_energy_direction'
            'mean_absolute_period', 'mean_zero-crossing_period',
            'omni-directional_wave_power', 'peak_period',
            'significant_wave_height', 'spectral_width',
            'water_depth', 'maximim_energy_direction',
            'mean_wave_direction', 'frequency_bin_edges'
    lat_lon : tuple or list of tuples
        Latitude longitude pairs at which to extract data
    years : list
        Year(s) to be accessed. The years 1979-2010 available.
        Examples: [1996] or [2004,2006,2007]
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
    path : string (optional)
        Optionally override with a custom .h5 filepath. Useful when setting
        `hsds=False`.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    data: pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named for parameter
        and cooresponding metadata index
    meta: DataFrame
        Location metadata for the requested data location
    """
    if not isinstance(parameter, (str, list)):
        raise TypeError(
            f"parameter must be of type string or list. Got: {type(parameter)}"
        )
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")
    if not isinstance(data_type, str):
        raise TypeError(f"data_type must be a string. Got: {type(data_type)}")
    if not isinstance(years, list):
        raise TypeError(f"years must be a list. Got: {type(years)}")
    if not isinstance(tree, (str, type(None))):
        raise TypeError(f"If specified, tree must be a string. Got: {type(tree)}")
    if not isinstance(unscale, bool):
        raise TypeError(
            f"If specified, unscale must be bool type. Got: {type(unscale)}"
        )
    if not isinstance(str_decode, bool):
        raise TypeError(
            f"If specified, str_decode must be bool type. Got: {type(str_decode)}"
        )
    if not isinstance(hsds, bool):
        raise TypeError(f"If specified, hsds must be bool type. Got: {type(hsds)}")
    if not isinstance(path, (str, type(None))):
        raise TypeError(f"If specified, path must be a string. Got: {type(path)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f"If specified, to_pandas must be bool type. Got: {type(to_pandas)}"
        )

    # Attempt to load data from cache
    # Construct a string representation of the function parameters
    hash_params = f"{data_type}_{parameter}_{lat_lon}_{years}_{tree}_{unscale}_{str_decode}_{hsds}_{path}_{to_pandas}"
    cache_dir = _get_cache_dir()
    data, meta, _ = handle_caching(hash_params, cache_dir)

    if data is not None:
        return data, meta
    else:
        if "directional_wave_spectrum" in parameter:
            sys.exit("This function does not support directional_wave_spectrum output")

        # Check for multiple region selection
        if isinstance(lat_lon[0], float):
            region = region_selection(lat_lon)
        else:
            region_list = []
            for loc in lat_lon:
                region_list.append(region_selection(loc))
            if region_list.count(region_list[0]) == len(lat_lon):
                region = region_list[0]
            else:
                sys.exit("Coordinates must be within the same region!")

        if path:
            wave_path = path
        elif data_type == "3-hour":
            wave_path = f"/nrel/US_wave/{region}/{region}_wave_*.h5"
        elif data_type == "1-hour":
            wave_path = (
                f"/nrel/US_wave/virtual_buoy/{region}/{region}_virtual_buoy_*.h5"
            )
        else:
            print("ERROR: invalid data_type")

        wave_kwargs = {
            "tree": tree,
            "unscale": unscale,
            "str_decode": str_decode,
            "hsds": hsds,
            "years": years,
        }
        data_list = []

        with MultiYearWaveX(wave_path, **wave_kwargs) as rex_waves:
            if isinstance(parameter, list):
                for param in parameter:
                    temp_data = rex_waves.get_lat_lon_df(param, lat_lon)
                    gid = rex_waves.lat_lon_gid(lat_lon)
                    cols = temp_data.columns[:]
                    for i, col in zip(range(len(cols)), cols):
                        temp = f"{param}_{gid}"
                        temp_data = temp_data.rename(columns={col: temp})

                    data_list.append(temp_data)
                data = pd.concat(data_list, axis=1)

            else:
                data = rex_waves.get_lat_lon_df(parameter, lat_lon)
                cols = data.columns[:]

                for i, col in zip(range(len(cols)), cols):
                    temp = f"{parameter}_{i}"
                    data = data.rename(columns={col: temp})

            meta = rex_waves.meta.loc[cols, :]
            meta = meta.reset_index(drop=True)
            gid = rex_waves.lat_lon_gid(lat_lon)
            meta["gid"] = gid

            if not to_pandas:
                data = convert_to_dataset(data)
                data["time_index"] = pd.to_datetime(data.time_index)

                if isinstance(parameter, list):
                    param_coords = [f"{param}_{gid}" for param in parameter]
                    data.coords["parameter"] = xr.DataArray(
                        param_coords, dims="parameter"
                    )

                data.coords["year"] = xr.DataArray(years, dims="year")

                meta_ds = meta.to_xarray()
                data = xr.merge([data, meta_ds])

                # Remove the 'index' coordinate
                data = data.drop_vars("index")

        # save_to_cache(hash_params, data, meta)
        handle_caching(hash_params, cache_dir, data, meta)

        return data, meta


def request_wpto_directional_spectrum(
    lat_lon,
    year,
    tree=None,
    unscale=True,
    str_decode=True,
    hsds=True,
    path=None,
):
    """
    Returns directional spectra data from the WPTO wave hindcast hosted
    on AWS at the specified latitude and longitude point(s),
    or the closest available point(s). The data is returned as an
    xarray Dataset with keys indexed by a graphical identifier (gid).
    `gid`s are integers which represent a lat, long on which data is
    stored. Requesting an array of `lat_lons` will return a dataset
    with multiple `gids` representing the data closest to each requested
    `lat`, `lon`.

    Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more
    information about the dataset and available
    locations and years.

    Note: To access the WPTO hindcast data, you will need to configure
    h5pyd for data access on HSDS.
    Please see the WPTO_hindcast_example notebook for more information.

    Parameters
    ----------
    lat_lon: tuple or list of tuples
        Latitude longitude pairs at which to extract data
    year : string
        Year to be accessed. The years 1979-2010 available.
        Only one year can be requested at a time.
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
    path : string (optional)
        Optionally override with a custom .h5 filepath. Useful when setting
        `hsds=False`

    Returns
    ---------
    data: xarray Dataset
        Coordinates as datetime, frequency, and direction for data at
        specified location(s)
    meta: DataFrame
        Location metadata for the requested data location
    """
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")
    if not isinstance(year, str):
        raise TypeError(f"year must be a string. Got: {type(year)}")
    if not isinstance(tree, (str, type(None))):
        raise TypeError(f"If specified, tree must be a string. Got: {type(tree)}")
    if not isinstance(unscale, bool):
        raise TypeError(
            f"If specified, unscale must be bool type. Got: {type(unscale)}"
        )
    if not isinstance(str_decode, bool):
        raise TypeError(
            f"If specified, str_decode must be bool type. Got: {type(str_decode)}"
        )
    if not isinstance(hsds, bool):
        raise TypeError(f"If specified, hsds must be bool type. Got: {type(hsds)}")
    if not isinstance(path, (str, type(None))):
        raise TypeError(f"If specified, path must be a string. Got: {type(path)}")

    # check for multiple region selection
    if isinstance(lat_lon[0], float):
        region = region_selection(lat_lon)
    else:
        reglist = [region_selection(loc) for loc in lat_lon]
        if reglist.count(reglist[0]) == len(lat_lon):
            region = reglist[0]
        else:
            sys.exit("Coordinates must be within the same region!")

    # Attempt to load data from cache
    hash_params = f"{lat_lon}_{year}_{tree}_{unscale}_{str_decode}_{hsds}_{path}"
    cache_dir = _get_cache_dir()
    data, meta, _ = handle_caching(hash_params, cache_dir)

    if data is not None:
        return data, meta

    wave_path = path or (
        f"/nrel/US_wave/virtual_buoy/{region}/{region}_virtual_buoy_{year}.h5"
    )
    parameter = "directional_wave_spectrum"
    wave_kwargs = {
        "tree": tree,
        "unscale": unscale,
        "str_decode": str_decode,
        "hsds": hsds,
    }

    with WaveX(wave_path, **wave_kwargs) as rex_waves:
        # Get graphical identifier
        gid = rex_waves.lat_lon_gid(lat_lon)

        # Setup index and columns
        columns = [gid] if isinstance(gid, (int, np.integer)) else gid
        time_index = rex_waves.time_index
        frequency = rex_waves["frequency"]
        direction = rex_waves["direction"]
        index = pd.MultiIndex.from_product(
            [time_index, frequency, direction],
            names=["time_index", "frequency", "direction"],
        )

        # Create bins for multiple smaller API dataset requests
        N = 6
        length = len(rex_waves)
        quotient, remainder = divmod(length, N)
        bins = [i * quotient for i in range(N + 1)]
        bins[-1] += remainder
        index_bins = (np.array(bins) * len(frequency) * len(direction)).tolist()

        # Request multiple datasets and add to dictionary
        datas = {}
        for i in range(len(bins) - 1):
            idx = index[index_bins[i] : index_bins[i + 1]]

            # Request with exponential back off wait time
            sleep_time = 2
            num_retries = 4
            for _ in range(num_retries):
                try:
                    data_array = rex_waves[parameter, bins[i] : bins[i + 1], :, :, gid]
                    str_error = None
                except Exception as err:
                    str_error = str(err)

                if str_error:
                    sleep(sleep_time)
                    sleep_time *= 2
                else:
                    break

            ax1 = np.product(data_array.shape[:3])
            ax2 = data_array.shape[-1] if len(data_array.shape) == 4 else 1
            datas[i] = pd.DataFrame(
                data_array.reshape(ax1, ax2), columns=columns, index=idx
            )

        data_raw = pd.concat(datas.values())
        data = data_raw.to_xarray()
        data["time_index"] = pd.to_datetime(data.time_index)

        # Get metadata
        meta = rex_waves.meta.loc[columns, :]
        meta = meta.reset_index(drop=True)
        meta["gid"] = gid

        # Convert gid to integer or list of integers
        gid_list = (
            [int(g) for g in gid] if isinstance(gid, (list, np.ndarray)) else [int(gid)]
        )

        data_var_concat = xr.concat([data[g] for g in gid_list], dim="gid")

        # Create a new DataArray with the correct dimensions and coordinates
        spectral_density = xr.DataArray(
            data_var_concat.data.reshape(
                -1, len(frequency), len(direction), len(gid_list)
            ),
            dims=["time_index", "frequency", "direction", "gid"],
            coords={
                "time_index": data["time_index"],
                "frequency": data["frequency"],
                "direction": data["direction"],
                "gid": gid_list,
            },
        )

        # Create the new dataset
        data = xr.Dataset(
            {"spectral_density": spectral_density},
            coords={
                "time_index": data["time_index"],
                "frequency": data["frequency"],
                "direction": data["direction"],
                "gid": gid_list,
            },
        )

    handle_caching(hash_params, cache_dir, data, meta)

    return data, meta


def _get_cache_dir():
    """
    Returns the path to the cache directory.
    """
    return os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "hindcast")
