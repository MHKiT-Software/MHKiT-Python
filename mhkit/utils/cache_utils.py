import hashlib
import json
import os
import shutil
import pickle
import pandas as pd


def handle_caching(hash_params, cache_dir, data=None, metadata=None, write_json=None,
                   clear_cache=False):
    """
    Handles caching of data to avoid redundant network requests or
    computations.

    The function checks if a cache file exists for the given parameters.
    If it does, the function will load data from the cache file, unless
    the `clear_cache` parameter is set to `True`, in which case the
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
    clear_cache : bool
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

    # Make cache directory if it doesn't exist
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    cache_filename = hashlib.md5(
        hash_params.encode('utf-8')).hexdigest() + ".json"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # If clear_cache is True, remove the cache file for this request
    if clear_cache and os.path.isfile(cache_filepath):
        os.remove(cache_filepath)
        print(f"Cleared cache for {cache_filepath}")

    # If a cached file exists, load and return the data from the file
    if os.path.isfile(cache_filepath) and data is None:
        with open(cache_filepath, "r") as f:
            jsonData = json.load(f)

        # Extract metadata if it exists
        if 'metadata' in jsonData:
            metadata = jsonData.pop('metadata', None)

        # Convert the rest to DataFrame
        data = pd.DataFrame(
            jsonData['data'],
            index=pd.to_datetime(jsonData['index']),
            columns=jsonData['columns']
        )

        if write_json:
            shutil.copy(cache_filepath, write_json)

        return data, metadata, cache_filepath

    elif data is not None:
        # Convert DataFrame to python dict
        pyData = data.to_dict(orient='split')
        # Add metadata to pyData
        pyData['metadata'] = metadata
        # Write the pyData to a json file
        pyData['index'] = [dt.strftime('%Y-%m-%d %H:%M:%S')
                           for dt in pyData['index']]
        with open(cache_filepath, "w") as outfile:
            json.dump(pyData, outfile)

        if write_json:
            shutil.copy(cache_filepath, write_json)

        return data, metadata, cache_filepath

    return None, None, cache_filepath


def cache_cdip(hash_params, cache_dir, data=None):
    # Create a unique identifier for this function call
    hash_id = hashlib.md5(hash_params.encode()).hexdigest()

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create a path to the cache file for this function call
    cache_file = os.path.join(cache_dir, f"{hash_id}.pkl")

    if data is not None:
        # If data is provided, store it in the cache
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return data, None, None

    elif os.path.isfile(cache_file):
        # If data is not provided and the cache file exists, load the data from the cache
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data, None, None

    else:
        # If data is not provided and the cache file doesn't exist, return None
        return None, None, None
