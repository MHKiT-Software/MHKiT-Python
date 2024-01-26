"""
Unit Testing for MHKiT Cache Utilities

This module provides unit tests for the caching utilities present in the MHKiT library. 
These utilities help in caching and retrieving data, ensuring efficient and repeatable 
data access without redundant computations or network requests.

The tests cover:
1. Creation of cache files with the correct file naming based on provided parameters.
2. Proper retrieval of data from the cache, ensuring data integrity.
3. Usage of appropriate file extensions based on the type of data being cached.
4. Clearing of cache directories as specified.

By running these tests, one can validate that the caching utilities of MHKiT are functioning 
as expected, ensuring that users can rely on cached data and metadata when using the MHKiT library.

Usage:
    python -m unittest test_cache.py

Requirements:
    - pandas
    - hashlib
    - tempfile
    - shutil
    - os
    - unittest
    - MHKiT library functions (from mhkit.utils.cache)

Author: ssolson
Date: 2023-08-18
"""

import unittest
import hashlib
import tempfile
import shutil
import os
import pandas as pd
from mhkit.utils.cache import handle_caching, clear_cache


class TestCacheUtils(unittest.TestCase):
    """
    Unit tests for cache utility functions.

    This test class provides a suite of tests to validate the functionality of caching utilities,
    ensuring data is correctly cached, retrieved, and cleared. It specifically tests:

    1. The creation of cache files by the `handle_caching` function.
    2. The correct retrieval of data from the cache.
    3. The appropriate file extension used when caching CDIP data.
    4. The effective clearing of specified cache directories.

    During the setup phase, a test cache directory is created, and sample data is prepared.
    Upon completion of tests, the teardown phase ensures the test cache directory is removed,
    leaving the environment clean.

    Attributes:
    -----------
    cache_dir : str
        Directory path where the test cache files will be stored.
    hash_params : str
        Sample parameters to be hashed for cache file naming.
    data : pandas DataFrame
        Sample data to be used for caching in tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "mhkit", "test_cache"
        )
        cls.hash_params = "test_params"
        cls.data = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=pd.date_range("20220101", periods=3)
        )

    @classmethod
    def tearDownClass(cls):
        # Remove the test_cache directory
        if os.path.exists(cls.cache_dir):
            shutil.rmtree(cls.cache_dir)

    def test_handle_caching_creates_cache(self):
        """
        Test if the `handle_caching` function correctly creates a cache file.

        The method tests the following scenario:
        1. Invokes the `handle_caching` function to cache a sample DataFrame.
        2. Constructs the expected cache file path based on provided `hash_params`.
        3. Checks if the cache file exists at the expected location.

        Asserts:
        - The cache file is successfully created at the expected file path.
        """
        handle_caching(self.hash_params, self.cache_dir, data=self.data)

        cache_filename = (
            hashlib.md5(self.hash_params.encode("utf-8")).hexdigest() + ".json"
        )
        cache_filepath = os.path.join(self.cache_dir, cache_filename)

        assert os.path.isfile(cache_filepath)

    def test_handle_caching_retrieves_data(self):
        """
        Test if the `handle_caching` function retrieves the correct data from cache.

        The method tests the following scenario:
        1. Invokes the `handle_caching` function to cache a sample DataFrame.
        2. Retrieves the data from the cache using the `handle_caching` function.
        3. Compares the retrieved data to the original sample DataFrame.

        Asserts:
        - The retrieved data matches the original sample DataFrame.
        """
        handle_caching(self.hash_params, self.cache_dir, data=self.data)
        retrieved_data, _, _ = handle_caching(self.hash_params, self.cache_dir)
        pd.testing.assert_frame_equal(self.data, retrieved_data, check_freq=False)

    def test_handle_caching_cdip_file_extension(self):
        """
        Test if the `handle_caching` function uses the correct file extension for CDIP caching.

        The method tests the following scenario:
        1. Specifies the cache directory to include "cdip", signaling CDIP-related caching.
        2. Invokes the `handle_caching` function to cache a sample DataFrame in the CDIP directory.
        3. Constructs the expected cache file path using a ".pkl" extension based on provided `hash_params`.
        4. Checks if the cache file with the ".pkl" extension exists at the expected location.

        Asserts:
        - The cache file with a ".pkl" extension is successfully created at the expected file path.
        """
        cache_dir = os.path.join(self.cache_dir, "cdip")
        handle_caching(self.hash_params, cache_dir, data=self.data)

        cache_filename = (
            hashlib.md5(self.hash_params.encode("utf-8")).hexdigest() + ".pkl"
        )
        cache_filepath = os.path.join(cache_dir, cache_filename)

        assert os.path.isfile(cache_filepath)

    def test_clear_cache(self):
        """
        Test if the `clear_cache` function correctly clears the specified cache directory.

        The method tests the following scenario:
        1. Moves the contents of the directory to be cleared to a temporary location.
        2. Invokes the `clear_cache` function to clear the specified directory.
        3. Checks if the directory has been cleared.
        4. Restores the original contents of the directory from the temporary location.

        Asserts:
        - The specified directory is successfully cleared by the `clear_cache` function.
        """
        specific_dir = "wave"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit")
        path_to_clear = os.path.join(cache_dir, specific_dir)

        # Step 1: Move contents to temporary directory
        temp_dir = tempfile.mkdtemp()
        if os.path.exists(path_to_clear):
            shutil.move(path_to_clear, temp_dir)

        # Step 2: Run clear_cache and test
        clear_cache(specific_dir)
        assert not os.path.exists(path_to_clear)

        # Step 3: Move contents back to original location, if they exist in the temporary directory
        if os.path.exists(os.path.join(temp_dir, specific_dir)):
            shutil.move(os.path.join(temp_dir, specific_dir), cache_dir)
        shutil.rmtree(temp_dir)  # Clean up temporary directory


if __name__ == "__main__":
    unittest.main()
