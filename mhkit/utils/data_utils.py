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
            "xarray.DataArray, or xarray.Dataset"
        )

    if not isinstance(name, str):
        raise TypeError("The 'name' parameter must be a string")

    # Takes data that could be pd.DataFrame, pd.Series, xr.DataArray, or
    # xr.Dataset and converts it to xr.Dataset
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_xarray()

    if isinstance(data, xr.DataArray):
        if data.name is None:
            data.name = (
                name  # xr.DataArray.to_dataset() breaks if the data variable is unnamed
            )
        data = data.to_dataset()

    return data
