import pandas as pd
import xarray as xr 

def _convert_to_dataset(data, name='data'):
    # Takes data that could be pd.DataFrame, pd.Series, xr.DataArray, or 
    # xr.Dataset and converts it to xr.Dataset
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_xarray()

    if isinstance(data, xr.DataArray):
        if data.name is None:
            data.name = name # xr.DataArray.to_dataset() breaks if the data variable is unnamed
        data = data.to_dataset()

    return data
