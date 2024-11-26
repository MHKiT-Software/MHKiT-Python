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

from typing import Optional, Tuple, Dict, Any
import hashlib
import json
import os
import shutil
import pickle
import pandas as pd


def handle_caching(
    hash_params: str,
    cache_dir: str,
    cache_content: Optional[Dict[str, Any]] = None,
    clear_cache_file: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], str]:
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
        Parameters to generate the cache file hash.
    cache_dir : str
        Directory where cache files are stored.
    cache_content : Optional[Dict[str, Any]], optional
        Content to be cached. Should contain 'data', 'metadata', and 'write_json'.
    clear_cache_file : bool
        Whether to clear the existing cache.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], str]
        Cached data, metadata, and cache file path.
    """

    data = None
    metadata = None

    def _generate_cache_filepath():
        """Generates the cache file path based on the hashed parameters."""
        file_extension = (
            ".pkl"
            if "cdip" in cache_dir or "hindcast" in cache_dir or "ndbc" in cache_dir
            else ".json"
        )
        cache_filename = (
            hashlib.md5(hash_params.encode("utf-8")).hexdigest() + file_extension
        )
        return os.path.join(cache_dir, cache_filename), file_extension

    def _clear_cache(cache_filepath):
        """Clear the cache file if requested."""
        if clear_cache_file and os.path.isfile(cache_filepath):
            os.remove(cache_filepath)
            print(f"Cleared cache for {cache_filepath}")

    def _load_cache(file_extension, cache_filepath):
        """Load data from the cache file based on its extension."""
        nonlocal data, metadata  # Specify that these are outer variables
        if file_extension == ".json":
            with open(cache_filepath, encoding="utf-8") as f:
                json_data = json.load(f)

            metadata = json_data.pop("metadata", None)

            data = pd.DataFrame(
                json_data["data"],
                index=pd.to_datetime(json_data["index"]),
                columns=json_data["columns"],
            )
        elif file_extension == ".pkl":
            with open(cache_filepath, "rb") as f:
                data, metadata = pickle.load(f)

        return data, metadata

    def _write_cache(data, metadata, file_extension, cache_filepath):
        """Store data in the cache file based on the extension."""
        if file_extension == ".json":
            py_data = data.to_dict(orient="split")
            py_data["metadata"] = metadata
            if isinstance(data.index, pd.DatetimeIndex):
                py_data["index"] = [
                    dt.strftime("%Y-%m-%d %H:%M:%S") for dt in py_data["index"]
                ]
            else:
                py_data["index"] = list(data.index)
            with open(cache_filepath, "w", encoding="utf-8") as f:
                json.dump(py_data, f)
        elif file_extension == ".pkl":
            with open(cache_filepath, "wb") as f:
                pickle.dump((data, metadata), f)

    # Create the cache directory if it doesn't exist
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    # Generate cache filepath and extension
    cache_filepath, file_extension = _generate_cache_filepath()

    # Clear cache if requested
    _clear_cache(cache_filepath)

    # If cache file exists and cache_content["data"] is None, load from cache
    if os.path.isfile(cache_filepath) and (
        cache_content is None or cache_content["data"] is None
    ):
        return _load_cache(file_extension, cache_filepath) + (cache_filepath,)

    # Store data in cache if provided
    if cache_content and cache_content["data"] is not None:
        _write_cache(
            cache_content["data"],
            cache_content["metadata"],
            file_extension,
            cache_filepath,
        )
        if cache_content["write_json"]:
            shutil.copy(cache_filepath, cache_content["write_json"])

        return cache_content["data"], cache_content["metadata"], cache_filepath

    return None, None, cache_filepath


def clear_cache(specific_dir: Optional[str] = None) -> None:
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
