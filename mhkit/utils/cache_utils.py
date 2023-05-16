import os
import pandas as pd
import json
import requests
import hashlib
import shutil


def handle_caching(hash_params, cache_dir, write_json=None, clear_cache=False):
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
    if os.path.isfile(cache_filepath):
        with open(cache_filepath, "r") as f:
            jsonData = json.load(f)

        # Extract metadata if it exists
        metadata = None
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

    return None, None, cache_filepath
