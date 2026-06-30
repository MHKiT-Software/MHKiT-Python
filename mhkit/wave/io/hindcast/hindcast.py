"""
This module provides functions to access and process WPTO wave hindcast data
hosted on AWS at specified latitude and longitude points or the closest
available points. It includes functions to retrieve data for predefined
regions, request point data for various parameters, and request directional
spectrum data.

Author: rpauly, aidanbharath, ssolson, simmsa
Date: 2026-06-29
"""

import os
import sys
import warnings
from time import sleep
from typing import List, Tuple, Union, Optional, Dict
import pandas as pd
import xarray as xr
import numpy as np
import fsspec
from scipy.spatial import cKDTree
from rex import MultiYearWaveX, WaveX
from rex.utilities.exceptions import ResourceRuntimeError
from mhkit.utils.cache import handle_caching
from mhkit.utils.type_handling import convert_to_dataset

# Public AWS S3 bucket mirroring the WPTO US Wave hindcast .h5 files served by
# HSDS. Used as a fallback when HSDS is unavailable: the same data is read
# directly from S3. https://registry.opendata.aws/wpto-pds-us-wave/
_WAVE_S3_BUCKET = "wpto-pds-us-wave"

# Retry count for the initial HSDS open in the hybrid read. Kept low so an
# unresponsive HSDS server fails over to the S3 fallback in seconds rather than
# waiting out h5pyd's default of 10 retries with exponential backoff.
_HSDS_OPEN_RETRIES = 2


def _latest_s3_version(region_path: str) -> str:
    """
    Returns the newest version directory in the wave S3 bucket that contains
    the given region path.

    The bucket stores each release under a version directory (e.g. "v1.0.1")
    and not every region exists in every version, so the version is resolved by
    listing the bucket rather than assumed.

    Parameters
    ----------
    region_path : string
        Region path under a version directory, e.g. "West_Coast" or
        "virtual_buoy/West_Coast"

    Returns
    -------
    version : string
        Newest version directory containing the region, e.g. "v1.0.1"
    """
    file_system = fsspec.filesystem("s3", anon=True)
    versions = sorted(
        (
            entry.rsplit("/", 1)[-1]
            for entry in file_system.ls(_WAVE_S3_BUCKET)
            if entry.rsplit("/", 1)[-1].startswith("v")
        ),
        reverse=True,
    )
    for version in versions:
        if file_system.exists(f"{_WAVE_S3_BUCKET}/{version}/{region_path}"):
            return version
    raise FileNotFoundError(
        f"{region_path} not found in any version of s3://{_WAVE_S3_BUCKET}"
    )


def _s3_wave_path(region: str, data_type: str, year: str = "*") -> str:
    """
    Builds the s3:// path for a wave region, mirroring the HSDS domain layout.

    Parameters
    ----------
    region : string
        Region name, e.g. "West_Coast"
    data_type : string
        "3-hour" for the regular wave files or "1-hour" for the virtual_buoy
        files
    year : string or int
        Year to read, or "*" for a multi-year wildcard glob. Default "*".

    Returns
    -------
    s3_path : string
        s3:// path (or wildcard glob) of the underlying .h5 file(s)
    """
    if data_type == "1-hour":
        region_path = f"virtual_buoy/{region}"
        filename = f"{region}_virtual_buoy_{year}.h5"
    else:
        region_path = region
        filename = f"{region}_wave_{year}.h5"
    version = _latest_s3_version(region_path)
    return f"s3://{_WAVE_S3_BUCKET}/{version}/{region_path}/{filename}"


def _open_wave_resource(
    resource_cls,
    hsds_path: str,
    wave_kwargs: dict,
    s3_region: str,
    s3_data_type: str,
    s3_year: str = "*",
    s3_fallback: bool = True,
):
    """
    Opens a rex wave resource on HSDS, falling back to a direct S3 read when the
    HSDS open fails (e.g. the HSDS server is unavailable).

    Returns an open resource. The caller closes it, typically with a `with`
    block. The HSDS and S3 reads return the same data, so the caller's read
    logic does not change with the source.

    Parameters
    ----------
    resource_cls : type
        rex resource class to open, WaveX or MultiYearWaveX
    hsds_path : string
        HSDS domain path to open
    wave_kwargs : dict
        Keyword arguments for the resource, including hsds and (for
        MultiYearWaveX) years
    s3_region : string
        Region name used to build the S3 path for the fallback
    s3_data_type : string
        "3-hour" or "1-hour", used to build the S3 path for the fallback
    s3_year : string or int
        Year used to build the S3 path for the fallback. Default "*".
    s3_fallback : bool
        Whether to fall back to S3 when the HSDS open fails. Disabled when the
        caller is not reading the default HSDS files (custom path or hsds=False).

    Returns
    -------
    rex_waves : WaveX or MultiYearWaveX
        Open rex resource, HSDS-backed when available else S3-backed
    """
    hsds_open_kwargs = dict(wave_kwargs)
    if wave_kwargs.get("hsds"):
        hsds_open_kwargs["hsds_kwargs"] = {"retries": _HSDS_OPEN_RETRIES}
    try:
        return resource_cls(hsds_path, **hsds_open_kwargs)
    except (ResourceRuntimeError, IOError):
        if not s3_fallback:
            raise
        # HSDS is unavailable. Read the same data directly from the public S3
        # bucket. rex reads s3:// paths with h5py and fsspec when hsds=False.
        warnings.warn(
            "HSDS is unavailable; falling back to reading the wave data "
            "directly from S3, which is slower.",
            UserWarning,
        )
        s3_path = _s3_wave_path(s3_region, s3_data_type, s3_year)
        return resource_cls(s3_path, **{**wave_kwargs, "hsds": False})


def _meta_cache_path(region: str, data_type: str) -> str:
    """Local parquet path for a wave dataset's cached coordinate metadata."""
    kind = "virtual_buoy" if data_type == "1-hour" else "wave"
    return os.path.join(_get_cache_dir(), "meta", f"{region}_{kind}.parquet")


def _cached_meta(region: str, data_type: str) -> pd.DataFrame:
    """
    Returns the location metadata (latitude, longitude, ...) for a wave dataset,
    indexed by grid id (gid), from a local parquet cache.

    The coordinates are fixed per region across all years, so they are read from
    S3 once and cached. Every later spatial query reuses the local copy instead
    of re-reading hundreds of thousands of points from S3.

    Parameters
    ----------
    region : string
        Region name, e.g. "West_Coast"
    data_type : string
        "3-hour" for the regular wave files or "1-hour" for the virtual_buoy
        files

    Returns
    -------
    meta : pandas.DataFrame
        Location metadata indexed by gid
    """
    cache_path = _meta_cache_path(region, data_type)
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)

    region_path = f"virtual_buoy/{region}" if data_type == "1-hour" else region
    version = _latest_s3_version(region_path)
    file_system = fsspec.filesystem("s3", anon=True)
    h5_files = [
        entry
        for entry in file_system.ls(f"{_WAVE_S3_BUCKET}/{version}/{region_path}")
        if entry.endswith(".h5")
    ]
    # The coordinates are identical across years, so any year's file works.
    with WaveX(f"s3://{h5_files[0]}", hsds=False) as s3_waves:
        meta = s3_waves.meta
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    meta.to_parquet(cache_path)
    return meta


def _meta_tree(meta: pd.DataFrame) -> cKDTree:
    """Returns a cKDTree of the (latitude, longitude) coordinates in meta."""
    return cKDTree(meta[["latitude", "longitude"]].to_numpy())


def _meta_for_gids(meta: pd.DataFrame, gids) -> pd.DataFrame:
    """
    Returns cached meta rows for the given grid ids, indexed from 0.

    Parameters
    ----------
    meta : pandas.DataFrame
        Cached location metadata indexed by gid
    gids : int, list, or array of int
        Grid ids to read

    Returns
    -------
    meta : pandas.DataFrame
        Meta rows for the given grid ids, indexed from 0
    """
    gids = [int(g) for g in np.atleast_1d(gids)]
    return meta.loc[gids, :].reset_index(drop=True)


def cache_wave_meta(
    lat_lon: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    data_type: str = "3-hour",
    datasets: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Pre-builds the local coordinate metadata cache so spatial queries never read
    coordinates from S3 at request time.

    This is optional. The request functions already cache each region's
    coordinates lazily on first use, downloading only the region a request
    falls in. Use this to warm the cache ahead of time, scoped to what is
    needed: pass lat_lon to cache only the region(s) those points fall in, pass
    datasets to cache an explicit list, or pass nothing to cache every region.

    Parameters
    ----------
    lat_lon : tuple or list of tuples, optional
        Point(s) whose region(s) to cache. The region is inferred from each
        point, so only the regions to be queried are downloaded.
    data_type : string, optional
        "3-hour" or "1-hour", paired with lat_lon. Default "3-hour".
    datasets : list of (region, data_type) tuples, optional
        Explicit datasets to cache, instead of inferring from lat_lon.
    """
    if datasets is None:
        if lat_lon is not None:
            points = lat_lon if isinstance(lat_lon[0], (list, tuple)) else [lat_lon]
            regions = {region_selection(point) for point in points}
            datasets = [(region, data_type) for region in regions]
        else:
            datasets = [
                ("West_Coast", "3-hour"),
                ("Atlantic", "3-hour"),
                ("Hawaii", "3-hour"),
                ("West_Coast", "1-hour"),
            ]
    for region, dataset_type in datasets:
        _cached_meta(region, dataset_type)


def _block_cache_path(
    region: str, data_type: str, years: List[int], parameter: str, block: int
) -> str:
    """Local parquet path for one cached location chunk (a block of gids)."""
    kind = "virtual_buoy" if data_type == "1-hour" else "wave"
    span = "_".join(str(y) for y in sorted(years))
    name = f"{region}_{kind}_{span}_{parameter}_block{block}.parquet"
    return os.path.join(_get_cache_dir(), "blocks", name)


def _data_block_size(rex_waves: Union[WaveX, MultiYearWaveX], parameter: str) -> int:
    """
    Returns the number of locations stored per chunk (the gid-dimension chunk
    size) for a data variable.

    The data is chunked across locations, so reading a whole chunk costs the
    same fetch as reading one location in it. This is the natural block size for
    the location cache.
    """
    for obj in (getattr(rex_waves, "resource", None), rex_waves):
        chunks = getattr(obj, "chunks", None)
        if isinstance(chunks, dict) and chunks.get(parameter):
            return int(chunks[parameter][-1])
    raise ValueError(f"Could not determine the chunk size for {parameter}")


def _gid_block(
    rex_waves: Union[WaveX, MultiYearWaveX],
    parameter: str,
    gid: int,
    block_size: int,
    n_gids: int,
    region: str,
    data_type: str,
    years: List[int],
) -> pd.DataFrame:
    """
    Returns the time series for the whole chunk of locations containing gid,
    from a local cache, reading and caching it from the resource on a miss.

    One location's chunk also holds its neighbors, so reading and caching the
    whole chunk costs the same fetch as a single location but serves every
    location in the chunk on later queries. Columns are the gids in the block,
    as strings.
    """
    block = gid // block_size
    cache_path = _block_cache_path(region, data_type, years, parameter, block)
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)

    start = block * block_size
    stop = min(start + block_size, n_gids)
    block_df = rex_waves.get_gid_df(parameter, list(range(start, stop)))
    block_df.columns = block_df.columns.astype(str)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    block_df.to_parquet(cache_path)
    return block_df


# A network read (HSDS or S3) can fail transiently, for example when the HSDS
# server is busy. Retry a failed read this many times, doubling the wait each
# time, before giving up.
_READ_ATTEMPTS = 4
_READ_BACKOFF_SECONDS = 2


def _read_with_retry(read):
    """
    Calls read, retrying a transient network failure with exponential backoff.

    Parameters
    ----------
    read : callable
        Zero-argument function that performs the read and returns its result

    Returns
    -------
    result
        The value returned by read
    """
    delay = _READ_BACKOFF_SECONDS
    for attempt in range(_READ_ATTEMPTS):
        try:
            return read()
        except (ResourceRuntimeError, IOError):
            if attempt == _READ_ATTEMPTS - 1:
                raise
            sleep(delay)
            delay *= 2


def _read_point_blocked(
    rex_waves: Union[WaveX, MultiYearWaveX],
    parameter: Union[str, List[str]],
    lat_lon,
    meta_cache: pd.DataFrame,
    region: str,
    data_type: str,
    years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like _read_point_data, but reads and caches the whole location chunk around
    each requested point. Nearby and repeat queries are then served from the
    local cache. See the cache_block option of request_wpto_point_data.

    Parameters
    ----------
    rex_waves : WaveX or MultiYearWaveX
        Open rex resource
    parameter : string or list of strings
        Parameter(s) to read
    lat_lon : tuple or list of tuples
        Latitude/longitude point(s) to read
    meta_cache : pandas.DataFrame
        Cached location metadata indexed by gid
    region : string
        Region name, used in the block cache path
    data_type : string
        "3-hour" or "1-hour", used in the block cache path
    years : list of int
        Years read, used in the block cache path

    Returns
    -------
    data : pandas.DataFrame
        Time-series data with columns renamed to ``{parameter}_{i}``
    meta : pandas.DataFrame
        Meta rows for the read columns, with a ``gid`` column
    """
    parameters = parameter if isinstance(parameter, list) else [parameter]
    gids = [int(g) for g in np.atleast_1d(rex_waves.lat_lon_gid(lat_lon))]
    n_gids = len(meta_cache)
    block_size = _data_block_size(rex_waves, parameters[0])
    time_index = rex_waves.time_index

    columns = {}
    for param in parameters:
        for i, gid in enumerate(gids):
            block_df = _gid_block(
                rex_waves, param, gid, block_size, n_gids, region, data_type, years
            )
            columns[f"{param}_{i}"] = block_df[str(gid)].to_numpy()
    data = pd.DataFrame(columns, index=time_index)

    meta = _meta_for_gids(meta_cache, gids)
    meta["gid"] = rex_waves.lat_lon_gid(lat_lon)
    return data, meta


def _read_point_data(
    rex_waves: Union[WaveX, MultiYearWaveX],
    parameter: Union[str, List[str]],
    lat_lon,
    meta_cache: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read renamed point time-series data and meta (with gid) from an open rex
    resource. Works against either an HSDS-backed or S3-backed resource.

    The spatial lookup uses the resource's pre-built tree and the meta comes
    from the cached coordinate metadata, so no coordinates are read from the
    resource.

    Parameters
    ----------
    rex_waves : WaveX or MultiYearWaveX
        Open rex resource
    parameter : string or list of strings
        Parameter(s) to read
    lat_lon : tuple or list of tuples
        Latitude/longitude point(s) to read
    meta_cache : pandas.DataFrame
        Cached location metadata indexed by gid

    Returns
    -------
    data : pandas.DataFrame
        Time-series data with columns renamed to ``{parameter}_{i}``
    meta : pandas.DataFrame
        Meta rows for the read columns, with a ``gid`` column
    """
    if isinstance(parameter, list):
        data_list = []
        for param in parameter:
            temp_data = rex_waves.get_lat_lon_df(param, lat_lon)
            cols = temp_data.columns[:]
            temp_data = temp_data.rename(
                columns={col: f"{param}_{i}" for i, col in enumerate(cols)}
            )
            data_list.append(temp_data)
        data = pd.concat(data_list, axis=1)
    else:
        data = rex_waves.get_lat_lon_df(parameter, lat_lon)
        cols = data.columns[:]
        data = data.rename(
            columns={col: f"{parameter}_{i}" for i, col in enumerate(cols)}
        )

    # A custom-path read has no cached coordinate metadata, so read the meta
    # from the resource itself in that case.
    if meta_cache is None:
        meta_cache = rex_waves.meta
    meta = _meta_for_gids(meta_cache, cols)
    meta["gid"] = rex_waves.lat_lon_gid(lat_lon)
    return data, meta


def region_selection(lat_lon: Union[List[float], Tuple[float, float]]) -> str:
    """
    Returns the name of the predefined region in which the given
    coordinates reside. Can be used to check if the passed lat/lon
    pair is within the WPTO hindcast dataset.

    Parameters
    ----------
    lat_lon : list or tuple
        Latitude and longitude coordinates as floats or integers

    Returns
    -------
    region : string
        Name of predefined region for given coordinates
    """
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")

    if not all(isinstance(coord, (float, int)) for coord in lat_lon):
        raise TypeError(
            f"lat_lon values must be of type float or int. Got: {type(lat_lon[0])}"
        )

    regions: Dict[str, Dict[str, List[float]]] = {
        "Hawaii": {"lat": [15.0, 27.000002], "lon": [-164.0, -151.0]},
        "West_Coast": {"lat": [30.0906, 48.8641], "lon": [-130.072, -116.899]},
        "Atlantic": {"lat": [24.382, 44.8247], "lon": [-81.552, -65.721]},
    }

    def region_search(
        lat_lon: Union[List[float], Tuple[float, float]],
        region: str,
        regions: Dict[str, Dict[str, List[float]]],
    ) -> bool:
        return all(
            regions[region][dk][0] <= d <= regions[region][dk][1]
            for dk, d in {"lat": lat_lon[0], "lon": lat_lon[1]}.items()
        )

    region = [region for region in regions if region_search(lat_lon, region, regions)]

    if not region:
        raise ValueError("ERROR: coordinates out of bounds.")

    return region[0]


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def request_wpto_point_data(
    data_type: str,
    parameter: Union[str, List[str]],
    lat_lon: Union[Tuple[float, float], List[Tuple[float, float]]],
    years: List[int],
    tree: Optional[str] = None,
    unscale: bool = True,
    str_decode: bool = True,
    hsds: bool = True,
    path: Optional[str] = None,
    to_pandas: bool = True,
    cache_block: bool = False,
) -> Tuple[Union[pd.DataFrame, xr.Dataset], pd.DataFrame]:
    """
    Returns data from the WPTO wave hindcast hosted on AWS at the
    specified latitude and longitude point(s), or the closest
    available point(s).
    Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more
    information about the dataset and available locations and years.

    Note: To access the WPTO hindcast data, you will need to configure
    h5pyd for data access on HSDS. Please see the WPTO_hindcast_example
    notebook for setup instructions.

    Parameters
    ----------
    data_type : string
        Data set type of interest
        Options: '3-hour' '1-hour'
    parameter : string or list of strings
        Dataset parameter to be downloaded
        3-hour dataset options: 'directionality_coefficient',
            'energy_period', 'maximum_energy_direction'
            'mean_absolute_period', 'mean_zero-crossing_period',
            'omni-directional_wave_power', 'peak_period'
            'significant_wave_height', 'spectral_width', 'water_depth'
        1-hour dataset options: 'directionality_coefficient',
            'energy_period', 'maximum_energy_direction'
            'mean_absolute_period', 'mean_zero-crossing_period',
            'omni-directional_wave_power', 'peak_period',
            'significant_wave_height', 'spectral_width',
            'water_depth', 'maximim_energy_direction',
            'mean_wave_direction', 'frequency_bin_edges'
    lat_lon : tuple or list of tuples
        Latitude longitude pairs at which to extract data
    years : list
        Year(s) to be accessed. The years 1979-2010 available.
        Examples: [1996] or [2004,2006,2007]
    tree : str | cKDTree (optional)
        cKDTree or path to .pkl file containing pre-computed tree
        of lat, lon coordinates, default = None
    unscale : bool (optional)
        Boolean flag to automatically unscale variables on extraction
        Default = True
    str_decode : bool (optional)
        Boolean flag to decode the bytestring meta data into normal
        strings. Setting this to False will speed up the meta data read.
        Default = True
    hsds : bool (optional)
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS. Setting to False will indicate to look for files on
        local machine, not AWS. Default = True
    path : string (optional)
        Optionally override with a custom .h5 filepath. Useful when setting
        `hsds=False`.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.
    cache_block : bool (optional)
        Read and cache the whole location chunk around each requested point,
        rather than only the requested location. Default = False.

        The hindcast data is chunked across locations, so fetching one location
        already transfers a chunk of roughly a thousand neighboring locations.
        With ``cache_block=True`` that whole chunk is kept and cached locally
        (under the hindcast cache directory, one parquet file per
        region/years/parameter/chunk), so later requests for any nearby location
        and the same years and parameters are served from disk without another
        download. This trades local disk for far fewer downloads and is useful
        when analyzing many points in the same area. The returned data is
        identical to ``cache_block=False``; only the caching differs. The cached
        chunk files can be inspected or pruned by hand.

    Returns
    ---------
    data: pandas DataFrame or xarray Dataset
        Data indexed by datetime with columns named for parameter
        and cooresponding metadata index
    meta: DataFrame
        Location metadata for the requested data location
    """
    if not isinstance(parameter, (str, list)):
        raise TypeError(
            f"parameter must be of type string or list. Got: {type(parameter)}"
        )
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")
    if not isinstance(data_type, str):
        raise TypeError(f"data_type must be a string. Got: {type(data_type)}")
    if not isinstance(years, list):
        raise TypeError(f"years must be a list. Got: {type(years)}")
    if not isinstance(tree, (str, type(None))):
        raise TypeError(f"If specified, tree must be a string. Got: {type(tree)}")
    if not isinstance(unscale, bool):
        raise TypeError(
            f"If specified, unscale must be bool type. Got: {type(unscale)}"
        )
    if not isinstance(str_decode, bool):
        raise TypeError(
            f"If specified, str_decode must be bool type. Got: {type(str_decode)}"
        )
    if not isinstance(hsds, bool):
        raise TypeError(f"If specified, hsds must be bool type. Got: {type(hsds)}")
    if not isinstance(path, (str, type(None))):
        raise TypeError(f"If specified, path must be a string. Got: {type(path)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(
            f"If specified, to_pandas must be bool type. Got: {type(to_pandas)}"
        )
    if not isinstance(cache_block, bool):
        raise TypeError(
            f"If specified, cache_block must be bool type. Got: {type(cache_block)}"
        )

    # Attempt to load data from cache
    # Construct a string representation of the function parameters
    hash_params = (
        f"{data_type}_{parameter}_{lat_lon}_{years}_{tree}_{unscale}_"
        f"{str_decode}_{hsds}_{path}_{to_pandas}"
    )
    cache_dir = _get_cache_dir()
    data, meta, _ = handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": None, "metadata": None, "write_json": None},
    )

    if data is not None:
        return data, meta

    if "directional_wave_spectrum" in parameter:
        sys.exit("This function does not support directional_wave_spectrum output")

    # Check for multiple region selection
    if isinstance(lat_lon[0], float):
        region = region_selection(lat_lon)
    else:
        region_list = []
        for loc in lat_lon:
            region_list.append(region_selection(loc))
        if region_list.count(region_list[0]) == len(lat_lon):
            region = region_list[0]
        else:
            sys.exit("Coordinates must be within the same region!")

    if path:
        wave_path = path
    elif data_type == "3-hour":
        wave_path = f"/nrel/US_wave/{region}/{region}_wave_*.h5"
    elif data_type == "1-hour":
        wave_path = f"/nrel/US_wave/virtual_buoy/{region}/{region}_virtual_buoy_*.h5"
    else:
        raise ValueError(
            f"Invalid data_type: {data_type}. Must be '3-hour' or '1-hour'"
        )

    # Use the cached coordinate metadata for the spatial lookup so the
    # coordinates are not read from HSDS or S3 on every request. Custom-path
    # reads have no cached metadata and let rex build the tree from the file.
    meta_cache = None if path else _cached_meta(region, data_type)
    if tree is None and meta_cache is not None:
        tree = _meta_tree(meta_cache)

    wave_kwargs = {
        "tree": tree,
        "unscale": unscale,
        "str_decode": str_decode,
        "hsds": hsds,
        "years": years,
    }

    def read_point():
        rex_waves = _open_wave_resource(
            MultiYearWaveX,
            wave_path,
            wave_kwargs,
            s3_region=region,
            s3_data_type=data_type,
            s3_fallback=hsds and path is None,
        )
        with rex_waves:
            if cache_block and meta_cache is not None:
                return _read_point_blocked(
                    rex_waves, parameter, lat_lon, meta_cache, region, data_type, years
                )
            return _read_point_data(rex_waves, parameter, lat_lon, meta_cache)

    # Retry the open and read together so a transient HSDS or S3 error (for
    # example a busy server during the multi-year setup) is recovered.
    data, meta = _read_with_retry(read_point)

    if not to_pandas:
        data = convert_to_dataset(data)
        data["time_index"] = pd.to_datetime(data.time_index)

        if isinstance(parameter, list):
            n_loc = 1 if isinstance(lat_lon[0], float) else len(lat_lon)
            param_coords = [f"{param}_{n_loc - 1}" for param in parameter]
            data.coords["parameter"] = xr.DataArray(param_coords, dims="parameter")

        data.coords["year"] = xr.DataArray(years, dims="year")

        meta_ds = meta.to_xarray()
        data = xr.merge([data, meta_ds])

        # Remove the 'index' coordinate
        data = data.drop_vars("index")

    # save_to_cache(hash_params, data, meta)
    handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": data, "metadata": meta, "write_json": None},
    )

    return data, meta


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def request_wpto_directional_spectrum(
    lat_lon: Union[Tuple[float, float], List[Tuple[float, float]]],
    year: str,
    tree: Optional[str] = None,
    unscale: bool = True,
    str_decode: bool = True,
    hsds: bool = True,
    path: Optional[str] = None,
) -> Tuple[xr.Dataset, pd.DataFrame]:
    """
    Returns directional spectra data from the WPTO wave hindcast hosted
    on AWS at the specified latitude and longitude point(s),
    or the closest available point(s). The data is returned as an
    xarray Dataset with keys indexed by a graphical identifier (gid).
    `gid`s are integers which represent a lat, long on which data is
    stored. Requesting an array of `lat_lons` will return a dataset
    with multiple `gids` representing the data closest to each requested
    `lat`, `lon`.

    Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more
    information about the dataset and available
    locations and years.

    Note: To access the WPTO hindcast data, you will need to configure
    h5pyd for data access on HSDS.
    Please see the WPTO_hindcast_example notebook for more information.

    Parameters
    ----------
    lat_lon: tuple or list of tuples
        Latitude longitude pairs at which to extract data
    year : string
        Year to be accessed. The years 1979-2010 available.
        Only one year can be requested at a time.
    tree : str | cKDTree (optional)
        cKDTree or path to .pkl file containing pre-computed tree
        of lat, lon coordinates, default = None
    unscale : bool (optional)
        Boolean flag to automatically unscale variables on extraction
        Default = True
    str_decode : bool (optional)
        Boolean flag to decode the bytestring meta data into normal
        strings. Setting this to False will speed up the meta data read.
        Default = True
    hsds : bool (optional)
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS. Setting to False will indicate to look for files on
        local machine, not AWS. Default = True
    path : string (optional)
        Optionally override with a custom .h5 filepath. Useful when setting
        `hsds=False`

    Returns
    ---------
    data: xarray Dataset
        Coordinates as datetime, frequency, and direction for data at
        specified location(s)
    meta: DataFrame
        Location metadata for the requested data location
    """
    if not isinstance(lat_lon, (list, tuple)):
        raise TypeError(f"lat_lon must be of type list or tuple. Got: {type(lat_lon)}")
    if not isinstance(year, str):
        raise TypeError(f"year must be a string. Got: {type(year)}")
    if not isinstance(tree, (str, type(None))):
        raise TypeError(f"If specified, tree must be a string. Got: {type(tree)}")
    if not isinstance(unscale, bool):
        raise TypeError(
            f"If specified, unscale must be bool type. Got: {type(unscale)}"
        )
    if not isinstance(str_decode, bool):
        raise TypeError(
            f"If specified, str_decode must be bool type. Got: {type(str_decode)}"
        )
    if not isinstance(hsds, bool):
        raise TypeError(f"If specified, hsds must be bool type. Got: {type(hsds)}")
    if not isinstance(path, (str, type(None))):
        raise TypeError(f"If specified, path must be a string. Got: {type(path)}")

    # check for multiple region selection
    if isinstance(lat_lon[0], float):
        region = region_selection(lat_lon)
    else:
        reglist = [region_selection(loc) for loc in lat_lon]
        if reglist.count(reglist[0]) == len(lat_lon):
            region = reglist[0]
        else:
            sys.exit("Coordinates must be within the same region!")

    # Attempt to load data from cache
    hash_params = f"{lat_lon}_{year}_{tree}_{unscale}_{str_decode}_{hsds}_{path}"
    cache_dir = _get_cache_dir()
    data, meta, _ = handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": None, "metadata": None, "write_json": None},
    )

    if data is not None:
        return data, meta

    wave_path = path or (
        f"/nrel/US_wave/virtual_buoy/{region}/{region}_virtual_buoy_{year}.h5"
    )
    parameter = "directional_wave_spectrum"
    # Use the cached coordinate metadata for the spatial lookup so coordinates
    # are not read from HSDS or S3 on every request. Custom-path reads have no
    # cached metadata and let rex build the tree from the file.
    meta_cache = None if path else _cached_meta(region, "1-hour")
    if tree is None and meta_cache is not None:
        tree = _meta_tree(meta_cache)
    wave_kwargs = {
        "tree": tree,
        "unscale": unscale,
        "str_decode": str_decode,
        "hsds": hsds,
    }

    rex_waves = _open_wave_resource(
        WaveX,
        wave_path,
        wave_kwargs,
        s3_region=region,
        s3_data_type="1-hour",
        s3_year=year,
        s3_fallback=hsds and path is None,
    )
    with rex_waves:
        # Get graphical identifier
        gid = rex_waves.lat_lon_gid(lat_lon)

        # Setup index and columns
        columns = [gid] if isinstance(gid, (int, np.integer)) else gid
        time_index = rex_waves.time_index
        frequency = rex_waves["frequency"]
        direction = rex_waves["direction"]
        index = pd.MultiIndex.from_product(
            [time_index, frequency, direction],
            names=["time_index", "frequency", "direction"],
        )

        # Create bins for multiple smaller API dataset requests
        num_bins = 6
        length = len(rex_waves)
        quotient, remainder = divmod(length, num_bins)
        bins = [i * quotient for i in range(num_bins + 1)]
        bins[-1] += remainder
        index_bins = (np.array(bins) * len(frequency) * len(direction)).tolist()

        # Request multiple datasets and add to dictionary
        datas = {}
        for i in range(len(bins) - 1):
            idx = index[index_bins[i] : index_bins[i + 1]]

            # Read each bin with a retry so a transient network error is
            # recovered without failing the whole request.
            data_array = _read_with_retry(
                lambda i=i: rex_waves[parameter, bins[i] : bins[i + 1], :, :, gid]
            )

            ax1 = np.prod(data_array.shape[:3])
            ax2 = data_array.shape[-1] if len(data_array.shape) == 4 else 1
            datas[i] = pd.DataFrame(
                data_array.reshape(ax1, ax2), columns=columns, index=idx
            )

        data_raw = pd.concat(datas.values())
        data = data_raw.to_xarray()
        data["time_index"] = pd.to_datetime(data.time_index)

        # Get metadata from the cached coordinates, or the resource itself for
        # a custom path.
        if meta_cache is None:
            meta_cache = rex_waves.meta
        meta = _meta_for_gids(meta_cache, columns)
        meta["gid"] = gid

        # Convert gid to integer or list of integers
        gid_list = (
            [int(g) for g in gid] if isinstance(gid, (list, np.ndarray)) else [int(gid)]
        )

        data_var_concat = xr.concat([data[g] for g in gid_list], dim="gid")

        # Create a new DataArray with the correct dimensions and coordinates
        spectral_density = xr.DataArray(
            data_var_concat.data.reshape(
                -1, len(frequency), len(direction), len(gid_list)
            ),
            dims=["time_index", "frequency", "direction", "gid"],
            coords={
                "time_index": data["time_index"],
                "frequency": data["frequency"],
                "direction": data["direction"],
                "gid": gid_list,
            },
        )

        # Create the new dataset
        data = xr.Dataset(
            {"spectral_density": spectral_density},
            coords={
                "time_index": data["time_index"],
                "frequency": data["frequency"],
                "direction": data["direction"],
                "gid": gid_list,
            },
        )

    handle_caching(
        hash_params,
        cache_dir,
        cache_content={"data": data, "metadata": meta, "write_json": None},
    )

    return data, meta


def _get_cache_dir() -> str:
    """
    Returns the path to the cache directory.
    """
    return os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "hindcast")
