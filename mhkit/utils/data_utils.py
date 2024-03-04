import numpy as np
import pandas as pd
import xarray as xr


def convert_to_dataset(data, name="data"):
    """
    Converts the given data to an xarray.Dataset.

    This function is designed to handle inputs that can be either a pandas DataFrame, a pandas Series,
    an xarray DataArray, or an xarray Dataset. It ensures that the output is consistently an xarray.Dataset.

    Parameters
    ----------
    data: pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        The data to be converted.

    name: str (Optional)
        The name to assign to the data variable in case the input is an xarray DataArray without a name.
        Default value is 'data'.

    Returns
    -------
    xarray.Dataset
        The input data converted to an xarray.Dataset. If the input is already an xarray.Dataset,
        it is returned as is.

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
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_xarray()

    if isinstance(data, xr.DataArray):
        # xr.DataArray.to_dataset() breaks if the data variable is unnamed
        if data.name == None:
            data.name = name
        data = data.to_dataset()

    return data


def convert_to_dataArray(data, name="data"):
    """
    Converts the given data to an xarray.DataArray.

    This function is designed to handle inputs that can be either a numpy ndarray, pandas Series,
    or an xarray DataArray. For convenience, pandas DataFrame and xarray Dataset can also be input
    but may only contain a single variable. The function ensures that the output is consistently
    an xarray.DataArray.

    Parameters
    ----------
    data: numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        The data to be converted.

    name: str (Optional)
        The name to overwrite the name of the input data variable for pandas or xarray input.
        Default value is 'data'.

    Returns
    -------
    xarray.DataArray
        The input data converted to an xarray.DataArray. If the input is already an xarray.DataArray,
        it is returned as is.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> da = convert_to_dataArray(df)
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>

    >>> series = pd.Series([1, 2, 3], name='C')
    >>> da = convert_to_dataArray(series)
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>

    >>> data_array = xr.DataArray([1, 2, 3])
    >>> da = convert_to_dataArray(data_array, name='D')
    >>> type(da)
    <class 'xarray.core.datarray.DataArray'>
    """
    if not isinstance(
        data, (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            "Input data must be of type np.ndarray, pandas.DataFrame, pandas.Series, "
            "xarray.DataArray, or xarray.Dataset"
        )

    if not isinstance(name, str):
        raise TypeError("The 'name' parameter must be a string")

    # Checks pd.DataFrame input and converts to pd.Series if possible
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 1:
            raise ValueError(
                "If the input data is a pd.DataFrame or xr.Dataset, it must contain one variable. Got {data.shape[1]}"
            )
        else:
            data = data.squeeze()

    # Checks xr.Dataset input and converts to xr.DataArray if possible
    if isinstance(data, xr.Dataset):
        if len(data.keys()) > 1:
            raise ValueError(
                "If the input data is a pd.DataFrame or xr.Dataset, it must contain one variable. Got {len(data.keys())}"
            )
        else:
            data = data.to_array()

    # Converts pd.Series to xr.DataArray
    if isinstance(data, pd.Series):
        data = data.to_xarray()

    # Converts np.ndarray to xr.DataArray. Assigns a simple 0-based dimension named index
    if isinstance(data, np.ndarray):
        data = xr.DataArray(
            data=data, dims="index", coords={"index": np.arange(len(data))}
        )

    # If there's not data name, add one to prevent issues calling or converting the dataArray later one
    if data.name == None:
        data.name = name

    return data
