import numpy as np
import pandas as pd
import xarray as xr


def to_numeric_array(data, name):
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
