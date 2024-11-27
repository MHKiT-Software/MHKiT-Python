"""
This module provides utility functions for converting various data types
to xarray structures such as xarray.DataArray and xarray.Dataset. It also
includes functions for handling nested dictionaries containing pandas 
DataFrames by converting them to xarray Datasets.

Functions:
----------
- to_numeric_array: Converts input data to a numeric NumPy array.
- convert_to_dataset: Converts pandas or xarray data structures to xarray.Dataset.
- convert_to_dataarray: Converts various data types to xarray.DataArray.
- convert_nested_dict_and_pandas: Recursively converts pandas DataFrames 
  in nested dictionaries to xarray Datasets.
"""

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import xarray as xr


def to_numeric_array(
    data: Union[list, np.ndarray, pd.Series, xr.DataArray], name: str
) -> np.ndarray:
    """
    Convert input data to a numeric array, ensuring all elements are numeric.
    """
    if isinstance(data, (list, np.ndarray, pd.Series, xr.DataArray)):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                (f"{name} must contain numeric data." + f" Got data type: {data.dtype}")
            )
    else:
        raise TypeError(
            (
                f"{name} must be a list, np.ndarray, pd.Series,"
                + f" or xr.DataArray. Got: {type(data)}"
            )
        )
    return data


def convert_to_dataset(
    data: Union[pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset], name: str = "data"
) -> xr.Dataset:
    """
    Converts the given data to an xarray.Dataset.

    This function is designed to handle inputs that can be either a
    pandas DataFrame, a pandas Series, an xarray DataArray, or an
    xarray Dataset. It ensures that the output is consistently an
    xarray.Dataset.

    Parameters
    ----------
    data: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        The data to be converted.

    name: str (Optional)
        The name to assign to the data variable in case the input is an
        xarray DataArray without a name.
        Default value is 'data'.

    Returns
    -------
    xarray.Dataset
        The input data converted to an xarray.Dataset. If the input is
        already an xarray.Dataset, it is returned as is.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> ds = convert_to_dataset(df)
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>

    >>> series = pd.Series([1, 2, 3], name='C')
    >>> ds = convert_to_dataset(series)
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>

    >>> data_array = xr.DataArray([1, 2, 3])
    >>> ds = convert_to_dataset(data_array, name='D')
    >>> type(ds)
    <class 'xarray.core.dataset.Dataset'>
    """
    if not isinstance(data, (pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset)):
        raise TypeError(
            "Input data must be of type pandas.DataFrame, pandas.Series, "
            "xarray.DataArray, or xarray.Dataset."
            f"Got {type(data)}."
        )

    if not isinstance(name, str):
        raise TypeError("The 'name' parameter must be a string" f"Got {type(name)}.")

    # Takes data that could be pd.DataFrame, pd.Series, xr.DataArray, or
    # xr.Dataset and converts it to xr.Dataset
    if isinstance(data, pd.DataFrame):
        # xr.Dataset(data) is drastically faster (1e1 - 1e2x faster)
        #  than using pd.DataFrame.to_xarray()
        data = xr.Dataset(data)

    if isinstance(data, pd.Series):
        # Converting to a DataArray then to a dataset makes the variable and
        # dimension naming cleaner than going straight to a Dataset with
        # xr.Dataset(pd.Series)
        data = xr.DataArray(data)

    if isinstance(data, xr.DataArray):
        # xr.DataArray.to_dataset() breaks if the data variable is unnamed
        if data.name is None:
            data.name = name
        data = data.to_dataset()

    return data


def convert_to_dataarray(
    data: Union[np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset],
    name: str = "data",
) -> xr.DataArray:
    """
    Converts the given data to an xarray.DataArray.

    This function takes in a numpy ndarray, pandas Series, pandas
    Dataframe, or xarray Dataset and outputs an equivalent xarray
    DataArray. DataArrays can be passed through with no changes.

    Xarray datasets can only be input when all variable have the same
    dimensions.

    Multivariate pandas Dataframes become 2D DataArrays, which is
    especially useful when IO functions return Dataframes with an
    extremely large number of variable. Use the function
    convert_to_dataset to change a multivariate Dataframe into a
    multivariate Dataset.

    Parameters
    ----------
    data: numpy ndarray, pandas DataFrame, pandas Series, xarray
    DataArray, or xarray Dataset
        The data to be converted.

    name: str (Optional)
        The name to overwrite the name of the input data variable for pandas or xarray input.
        Default value is 'data'.

    Returns
    -------
    xarray.DataArray
        The input data converted to an xarray.DataArray. If the input
        is already an xarray.DataArray, it is returned as is.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> da = convert_to_dataarray(df)
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>

    >>> series = pd.Series([1, 2, 3], name='C')
    >>> da = convert_to_dataarray(series)
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>

    >>> data_array = xr.DataArray([1, 2, 3])
    >>> da = convert_to_dataarray(data_array, name='D')
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>
    """
    if not isinstance(
        data, (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "Input data must be of type np.ndarray, pandas.Series, pandas.DataFrame, "
            f"xarray.DataArray, or xarray.Dataset. Got {type(data)}"
        )

    if not isinstance(name, str):
        raise TypeError(f"The 'name' parameter must be a string. Got {type(name)}")

    # Checks pd.DataFrame input and converts to pd.Series if possible
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            # Convert the 1D, univariate case to a Series, which will
            # be caught by the Series conversion below. This eliminates
            # an unnecessary variable dimension and names the DataArray
            # with the DataFrame variable name.
            #
            # Use iloc instead of squeeze. For DataFrames/Series with only a
            # single value, squeeze returns a scalar which is unexpected.
            # iloc returns a Series with one value as expected.
            data = data.iloc[:, 0]
        else:
            # With this conversion, dataframe columns always become "dim_1".
            # Rename to "variable" to match how multiple Dataset variables get
            # converted into a DataArray dimension
            data = xr.DataArray(data)
            if data.dims[1] == "dim_1":
                data = data.rename({"dim_1": "variable"})
    # Checks xr.Dataset input and converts to xr.DataArray if possible
    if isinstance(data, xr.Dataset):
        keys = list(data.keys())
        if len(keys) == 1:
            # if only one variable, select that variable so reduce the Dataset to a DataArray
            data = data[keys[0]]
        else:
            # Allow multiple variables if they have the same dimensions
            # transpose so that the new "variable dimension" is the last
            # dimension (matches DataFrame to DataArray behavior)
            if all(data[keys[0]].dims == data[key].dims for key in keys):
                data = data.to_array().T
            else:
                raise ValueError(
                    "Multivariate Datasets can only be input if all \
                        variables have the same dimensions."
                )

    # Converts pd.Series to xr.DataArray
    if isinstance(data, pd.Series):
        data = data.to_xarray()

    # Converts np.ndarray to xr.DataArray. Assigns a simple 0-based
    # dimension named index to match how pandas converts to xarray
    if isinstance(data, np.ndarray):
        data = xr.DataArray(
            data=data, dims="index", coords={"index": np.arange(len(data))}
        )

    # If there's no data name, add one to prevent issues calling or
    # converting to a Dataset later on
    data.name = data.name if data.name is not None else name

    return data


def convert_nested_dict_and_pandas(
    data: Dict[str, Union[pd.DataFrame, Dict[str, Any]]]
) -> Dict[str, Union[xr.Dataset, Dict[str, Any]]]:
    """
    Recursively searches inside nested dictionaries for pandas DataFrames to
    convert to xarray Datasets. Typically called by wave.io functions that read
    SWAN, WEC-Sim, CDIP, NDBC data.

    Parameters
    ----------
    data: dictionary of dictionaries and pandas DataFrames

    Returns
    -------
    data : dictionary of dictionaries and xarray Datasets

    """
    for key in data.keys():
        if isinstance(data[key], pd.DataFrame):
            data[key] = convert_to_dataset(data[key])
        elif isinstance(data[key], dict):
            data[key] = convert_nested_dict_and_pandas(data[key])

    return data
