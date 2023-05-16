import os
import pandas as pd
import json
import requests
import hashlib
import shutil


def handle_caching(hash_params, cache_dir, data=None, metadata=None, write_json=None,
                   clear_cache=False):
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
