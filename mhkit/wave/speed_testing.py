import numpy as np
import pandas as pd
import xarray as xr
from mhkit import wave
from mhkit.utils import to_numeric_array, convert_to_dataarray, convert_to_dataset
import timeit

def _transform_dataset(data, name):
    # Converting data from a Dataset into a DataArray will turn the variables
    # columns into a 'variable' dimension.
    # Converting it back to a dataset will keep this concise variable dimension
    # but in the expected xr.Dataset/pd.DataFrame format
    data = data.to_array()
    data = convert_to_dataset(data, name=name)
    data = data.rename({"variable": "index"})
    return data

def frequency_moment_da(S, N, frequency_bins=None, frequency_dimension="", to_pandas=True):
    S = convert_to_dataarray(S)
    if not isinstance(N, int):
        raise TypeError(f"N must be of type int. Got: {type(N)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )
    f = S[frequency_dimension]

    # Eq 8 in IEC 62600-101
    S = S.sel({frequency_dimension: slice(1e-12, f.max())})  # omit frequency of 0
    f = S[frequency_dimension]  # reset frequency_dimension without the 0 frequency

    fn = np.power(f, N)
    if frequency_bins is None:
        delta_f = f.diff(dim=frequency_dimension)
        delta_f0 = f[0]
        delta_f0 = delta_f0.assign_coords({frequency_dimension: f[0]})
        delta_f = xr.concat([delta_f0, delta_f], dim=frequency_dimension)
    else:
        delta_f = xr.DataArray(
            data=convert_to_dataarray(frequency_bins),
            dims=frequency_dimension,
            coords={frequency_dimension: f},
        )

    m = S * fn * delta_f
    m = m.sum(dim=frequency_dimension)

    m.name = "m" + str(N)

    if to_pandas:
        m = m.to_dataframe()

    return m

def energy_period_da(S, frequency_dimension="", frequency_bins=None, to_pandas=True):
    S = convert_to_dataarray(S)
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    mn1 = frequency_moment_da(
        S,
        -1,
        frequency_bins=frequency_bins,
        frequency_dimension=frequency_dimension,
        to_pandas=False,
    )
    m0 = frequency_moment_da(
        S,
        0,
        frequency_bins=frequency_bins,
        frequency_dimension=frequency_dimension,
        to_pandas=False,
    )

    # Eq 13 in IEC 62600-101
    Te = mn1 / m0
    Te.name = "Te"

    if to_pandas:
        Te = Te.to_dataframe()

    return Te


def frequency_moment_np(S, N, frequency_bins=None, frequency_dimension="", to_pandas=True):
    S = convert_to_dataarray(S)
    if not isinstance(N, int):
        raise TypeError(f"N must be of type int. Got: {type(N)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if frequency_dimension == "":
        frequency_dimension = list(S.coords)[0]
    elif frequency_dimension not in list(S.dims):
        raise ValueError(
            f"frequency_dimension is not a dimension of S ({list(S.dims)}). Got: {frequency_dimension}."
        )
    f = S[frequency_dimension].values
    mask = f >= 1e-12
    frequency_index = S.dims.index(frequency_dimension)

    S_np = to_numeric_array(S, name="S")

    # Eq 8 in IEC 62600-101
    mask_nd = mask * np.full(S_np.shape, True)
    S_np = S_np[mask_nd].reshape(S_np.shape)  # omit frequency of 0
    f = f[mask]  # reset frequency_dimension without the 0 frequency

    fn = np.power(f, N)
    if frequency_bins is None:
        delta_f = np.diff(f, frequency_index)
        delta_f0 = f[1] - f[0]
        delta_f = np.concatenate(([delta_f0], delta_f))
    else:
        delta_f = frequency_bins

    data = S_np * fn * delta_f
    data = data.sum(frequency_index)

    m = S[{frequency_dimension:0}]
    m = m.drop_vars(frequency_dimension)
    m.name = "m" + str(N)
    m.values = data

    # newDims = [dim for dim in S.dims if dim not in [frequency_dimension]]
    # newCoords = {dim: coord for dim,coord in newDims if S.}
    if to_pandas:
        m = m.to_pandas()

    return m

def energy_period_np(S, frequency_dimension="", frequency_bins=None, to_pandas=True):
    S = convert_to_dataarray(S)
    S_np = to_numeric_array(S, name="S")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    mn1 = frequency_moment_np(
        S,
        -1,
        frequency_bins=frequency_bins,
        frequency_dimension=frequency_dimension,
        to_pandas=False,
    )
    # mn1 = mn1.rename({"m-1": "Te"})
    m0 = frequency_moment_np(
        S,
        0,
        frequency_bins=frequency_bins,
        frequency_dimension=frequency_dimension,
        to_pandas=False,
    )
    # m0 = m0.rename({"m0": "Te"})

    # Eq 13 in IEC 62600-101
    Te = mn1 / m0
    Te.name = "Te"

    if to_pandas:
        Te = Te.to_pandas()

    return Te

# ndbc.read_file outputs the NDBC file data into two variables.
# raw_ndbc_data is a pandas DataFrame containing the file data.
# meta contains the meta data, if available.
# ndbc_data_file = "../../examples/data/wave/data.txt"
# [raw_ndbc_data, meta] = wave.io.ndbc.read_file(ndbc_data_file)
# raw_ndbc_data.head()
# ndbc_data = raw_ndbc_data.T

parameter = "swden"
ndbc_available_data = wave.io.ndbc.available_data(parameter, '46050')
filenames = ndbc_available_data["filename"]
filenames = filenames.loc[1994:1996] # isolate data for 2018-2019 to speed up script
ndbc_requested_data = wave.io.ndbc.request_data(parameter, filenames)
all_ndbc_data = {}
# Create a Datetime Index and remove NOAA date columns for each year
for year in ndbc_requested_data:
    year_data = ndbc_requested_data[year]
    all_ndbc_data[year] = wave.io.ndbc.to_datetime_index(parameter, year_data)


ndbc_data = all_ndbc_data["2018"]
# ndbc_data = raw_ndbc_data.T


# # Uncomment for format testing
# # How to get the two below cases to work in the same way?
# te_np = energy_period_np(ndbc_data, frequency_dimension="variable") # masking this does work because shapes are compatible
# te_np = energy_period_np(ndbc_data.T, frequency_dimension="index") # masking this does NOT work because shapes are incompatible
# te_da = energy_period_da(ndbc_data, frequency_dimension="variable")
# te_df = wave.resource.energy_period(ndbc_data, frequency_dimension="variable")

ndbc_data_da = convert_to_dataarray(ndbc_data)
ndbc_data_s = ndbc_data.iloc[0,:]
te = wave.resource.energy_period(ndbc_data, frequency_dimension="variable")
te_da = wave.resource.energy_period(ndbc_data_da, frequency_dimension="variable")
te_s = wave.resource.energy_period(ndbc_data_s)

# # Uncomment for speed testing
# # Initial results:
# # current: 43.4233, dataArray: 0.06466, numpy: 0.018468
# # --> dataArray ~700x faster than current
# # --> numpy ~3.5x faster than dataArray
# n = 5
# time = {}
# time["dataset"] = timeit.timeit(lambda: wave.resource.energy_period(ndbc_data), number=n)/n
# time["dataarray"] = timeit.timeit(lambda: energy_period_da(ndbc_data, frequency_dimension="index"), number=n)/n
# time["numpy"] = timeit.timeit(lambda: energy_period_np(ndbc_data, frequency_dimension="index"), number=n)/n

ndbc_data
