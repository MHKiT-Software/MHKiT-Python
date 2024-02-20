"""
This module provides functionality for managing cache files to optimize
network requests and computations for handling data. The module focuses
on enabling users to read from and write to cache files, as well as 
perform cache clearing operations. Cache files are utilized to store data 
temporarily, mitigating the need to re-fetch or recompute the same data multiple 
times, which can be especially useful in network-dependent tasks.

The module consists of two main functions:

1. `handle_caching`:
   This function manages the caching of data. It provides options to read from 
   and write to cache files, depending on whether the data is already provided 
   or if it needs to be fetched from the cache. If a cache file corresponding 
   to the given parameters already exists, the function can either load data 
   from it or clear it based on the parameters passed. It also offers the ability 
   to store associated metadata along with the data and supports both JSON and 
   pickle file formats for caching. This function returns the loaded data and 
   metadata from the cache file, along with the cache file path.

2. `clear_cache`:
   This function enables the clearing of either specific sub-directories or the 
   entire cache directory, depending on the parameter passed. It removes the 
   specified directory and then recreates it to ensure future caching tasks can 
   be executed without any issues. If the specified directory does not exist, 
   the function prints an indicative message.

Module Dependencies:
--------------------
    - hashlib: For creating unique filenames based on hashed parameters.
    - json: For reading and writing JSON formatted cache files.
    - os: For performing operating system dependent tasks like directory creation.
    - re: For regular expression operations to match datetime formatted strings.
    - shutil: For performing high-level file operations like copying and removal.
    - pickle: For reading and writing pickle formatted cache files.
    - pandas: For handling data in DataFrame format.

Author: ssolson
Date: 2023-09-26
"""

import hashlib
import json
import os
import re
import shutil
import pickle
import pandas as pd


def handle_caching(
    hash_params,
    cache_dir,
    data=None,
    metadata=None,
    write_json=None,
    clear_cache_file=False,
):
    """
    Handles caching of data to avoid redundant network requests or
    computations.

    The function checks if a cache file exists for the given parameters.
    If it does, the function will load data from the cache file, unless
    the `clear_cache_file` parameter is set to `True`, in which case the
    cache file is cleared. If the cache file does not exist and the
    `data` parameter is not `None`, the function will store the
    provided data in a cache file.

    Parameters
    ----------
    hash_params : str
        The parameters to be hashed and used as the filename for the cache file.
    cache_dir : str
        The directory where the cache files are stored.
    data : pandas DataFrame or None
        The data to be stored in the cache file. If `None`, the function
        will attempt to load data from the cache file.
    metadata : dict or None
        Metadata associated with the data. This will be stored in the
        cache file along with the data.
    write_json : str or None
        If specified, the cache file will be copied to a file with this name.
    clear_cache_file : bool
        If `True`, the cache file for the given parameters will be cleared.

    Returns
    -------
    data : pandas DataFrame or None
        The data loaded from the cache file. If data was provided as a
        parameter, the same data will be returned. If the cache file
        does not exist and no data was provided, `None` will be returned.
    metadata : dict or None
        The metadata loaded from the cache file. If metadata was provided
        as a parameter, the same metadata will be returned. If the cache
        file does not exist and no metadata was provided, `None` will be
        returned.
    cache_filepath : str
        The path to the cache file.
    """

    # Check if 'cdip' is in cache_dir, then use .pkl instead of .json
    file_extension = (
        ".pkl"
        if "cdip" in cache_dir or "hindcast" in cache_dir or "ndbc" in cache_dir
        else ".json"
    )

    # Make cache directory if it doesn't exist
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    # Create a unique filename based on the function parameters
    cache_filename = (
        hashlib.md5(hash_params.encode("utf-8")).hexdigest() + file_extension
    )
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # If clear_cache_file is True, remove the cache file for this request
    if clear_cache_file and os.path.isfile(cache_filepath):
        os.remove(cache_filepath)
        print(f"Cleared cache for {cache_filepath}")

    # If a cached file exists, load and return the data from the file
    if os.path.isfile(cache_filepath) and data is None:
        if file_extension == ".json":
            with open(cache_filepath, encoding="utf-8") as f:
                jsonData = json.load(f)

            # Extract metadata if it exists
            if "metadata" in jsonData:
                metadata = jsonData.pop("metadata", None)

            # Check if index is datetime formatted
            if all(
                re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(dt))
                for dt in jsonData["index"]
            ):
                data = pd.DataFrame(
                    jsonData["data"],
                    index=pd.to_datetime(jsonData["index"]),
                    columns=jsonData["columns"],
                )
            else:
                data = pd.DataFrame(
                    jsonData["data"],
                    index=jsonData["index"],
                    columns=jsonData["columns"],
                )

            # Convert the rest to DataFrame
            data = pd.DataFrame(
                jsonData["data"],
                index=pd.to_datetime(jsonData["index"]),
                columns=jsonData["columns"],
            )

        elif file_extension == ".pkl":
            with open(cache_filepath, "rb") as f:
                data, metadata = pickle.load(f)

        if write_json:
            shutil.copy(cache_filepath, write_json)

        return data, metadata, cache_filepath

    # If a cached file does not exist and data is provided,
    # store the data in a cache file
    elif data is not None:
        if file_extension == ".json":
            # Convert DataFrame to python dict
            pyData = data.to_dict(orient="split")
            # Add metadata to pyData
            pyData["metadata"] = metadata
            # Check if index is datetime indexed
            if isinstance(data.index, pd.DatetimeIndex):
                pyData["index"] = [
                    dt.strftime("%Y-%m-%d %H:%M:%S") for dt in pyData["index"]
                ]
            else:
                pyData["index"] = list(data.index)
            with open(cache_filepath, "w", encoding="utf-8") as f:
                json.dump(pyData, f)

        elif file_extension == ".pkl":
            with open(cache_filepath, "wb") as f:
                pickle.dump((data, metadata), f)

        if write_json:
            shutil.copy(cache_filepath, write_json)

        return data, metadata, cache_filepath
    # If data is not provided and the cache file doesn't exist, return cache_filepath
    return None, None, cache_filepath


def clear_cache(specific_dir=None):
    """
    Clears the cache.

    The function checks if a specific directory or the entire cache directory
    exists. If it does, the function will remove the directory and recreate it.
    If the directory does not exist, a message indicating is printed.

    Parameters
    ----------
    specific_dir : str or None, optional
        Specific sub-directory to clear. If None, the entire cache is cleared.
        Default is None.

    Returns
    -------
    None
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit")

    # Consider generating this from a system folder search
    folders = {
        "river": "river",
        "tidal": "tidal",
        "wave": "wave",
        "usgs": os.path.join("river", "usgs"),
        "noaa": os.path.join("tidal", "noaa"),
        "ndbc": os.path.join("wave", "ndbc"),
        "cdip": os.path.join("wave", "cdip"),
        "hindcast": os.path.join("wave", "hindcast"),
    }

    # If specific_dir is provided and matches a key in the folders dictionary,
    # use its corresponding value
    if specific_dir and specific_dir in folders:
        specific_dir = folders[specific_dir]

    # Construct the path to the directory to be cleared
    path_to_clear = os.path.join(cache_dir, specific_dir) if specific_dir else cache_dir

    # Check if the directory exists
    if os.path.exists(path_to_clear):
        # Clear the directory
        shutil.rmtree(path_to_clear)
        # Recreate the directory after deletion
        os.makedirs(path_to_clear)
    else:
        print(f"The directory {path_to_clear} does not exist.")
