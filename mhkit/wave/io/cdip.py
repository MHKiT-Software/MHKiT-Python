import os
import pandas as pd
import numpy as np
import datetime
import netCDF4
import pytz
from mhkit.utils.cache import handle_caching
from mhkit.utils import convert_nested_dict_and_pandas


def _validate_date(date_text):
    """
    Checks date format to ensure YYYY-MM-DD format and return date in
    datetime format.

    Parameters
    ----------
    date_text: string
        Date string format to check

    Returns
    -------
    dt: datetime
    """

    if not isinstance(date_text, str):
        raise ValueError("date_text must be of type string. Got: {date_text}")

    try:
        dt = datetime.datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    else:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt


def _start_and_end_of_year(year):
    """
    Returns a datetime start and end for a given year

    Parameters
    ----------
    year: int
        Year to get start and end dates

    Returns
    -------
    start_year: datetime object
        start of the year
    end_year: datetime object
        end of the year
    """

    if not isinstance(year, (type(None), int, list)):
        raise ValueError("year must be of type int, list, or None. Got: {type(year)}")

    try:
        year = str(year)
        start_year = datetime.datetime.strptime(year, "%Y")
    except ValueError as exc:
        raise ValueError("Incorrect years format, should be YYYY") from exc
    else:
        next_year = datetime.datetime.strptime(f"{int(year)+1}", "%Y")
        end_year = next_year - datetime.timedelta(days=1)
    return start_year, end_year


def _dates_to_timestamp(nc, start_date=None, end_date=None):
    """
    Returns timestamps from dates.

    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type
    start_date: string
        Start date in YYYY-MM-DD, e.g. '2012-04-01'
    end_date: string
        End date in YYYY-MM-DD, e.g. '2012-04-30'

    Returns
    -------
    start_stamp: float
         seconds since the Epoch to start_date
    end_stamp: float
         seconds since the Epoch to end_date
    """

    if start_date and not isinstance(start_date, datetime.datetime):
        raise ValueError(
            f"start_date must be of type datetime.datetime or None. Got: {type(start_date)}"
        )

    if end_date and not isinstance(end_date, datetime.datetime):
        raise ValueError(
            f"end_date must be of type datetime.datetime or None. Got: {type(end_date)}"
        )

    time_all = nc.variables["waveTime"][:].compressed()
    t_i = datetime.datetime.fromtimestamp(time_all[0]).astimezone(pytz.timezone("UTC"))
    t_f = datetime.datetime.fromtimestamp(time_all[-1]).astimezone(pytz.timezone("UTC"))
    time_range_all = [t_i, t_f]

    if start_date:
        start_date = start_date.astimezone(pytz.UTC)
        if start_date > time_range_all[0] and start_date < time_range_all[1]:
            start_stamp = start_date.timestamp()
        else:
            print(
                f"WARNING: Provided start_date ({start_date}) is "
                f"not in the returned data range {time_range_all} \n"
                f"Setting start_date to the earliest date in range "
                f"{time_range_all[0]}"
            )
            start_stamp = time_range_all[0].timestamp()

    if end_date:
        end_date = end_date.astimezone(pytz.UTC)
        if end_date > time_range_all[0] and end_date < time_range_all[1]:
            end_stamp = end_date.timestamp()
        else:
            print(
                f"WARNING: Provided end_date ({end_date}) is "
                f"not in the returned data range {time_range_all} \n"
                f"Setting end_date to the latest date in range "
                f"{time_range_all[1]}"
            )
            end_stamp = time_range_all[1].timestamp()

    if start_date and not end_date:
        end_stamp = time_range_all[1].timestamp()

    elif end_date and not start_date:
        start_stamp = time_range_all[0].timestamp()

    if not start_date:
        start_stamp = time_range_all[0].timestamp()
    if not end_date:
        end_stamp = time_range_all[1].timestamp()

    return start_stamp, end_stamp


def request_netCDF(station_number, data_type):
    """
    Returns historic or realtime data from CDIP THREDDS server

    Parameters
    ----------
    station_number: string
        CDIP station number of interest
    data_type: string
        'historic' or 'realtime'

    Returns
    -------
    nc: xarray Dataset
        netCDF data for the given station number and data type
    """

    if not isinstance(station_number, (str, type(None))):
        raise ValueError(
            f"station_number must be of type string. Got: {type(station_number)}"
        )

    if not isinstance(data_type, str):
        raise ValueError(f"data_type must be of type string. Got: {type(data_type)}")

    if data_type not in ["historic", "realtime"]:
        raise ValueError('data_type must be "historic" or "realtime". Got: {data_type}')

    BASE_URL = "http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/"

    if data_type == "historic":
        data_url = (
            f"{BASE_URL}archive/{station_number}p1/{station_number}p1_historic.nc"
        )
    else:  # data_type == 'realtime'
        data_url = f"{BASE_URL}realtime/{station_number}p1_rt.nc"

    nc = netCDF4.Dataset(data_url)

    return nc


def request_parse_workflow(
    nc=None,
    station_number=None,
    parameters=None,
    years=None,
    start_date=None,
    end_date=None,
    data_type="historic",
    all_2D_variables=False,
    silent=False,
    to_pandas=True,
):
    """
    Parses a passed CDIP netCDF file or requests a station number
    from http://cdip.ucsd.edu/) and parses. This function can return specific
    parameters is passed. Years may be non-consecutive e.g. [2001, 2010].
    Time may be sliced by dates (start_date or end date in YYYY-MM-DD).
    data_type defaults to historic but may also be set to 'realtime'.
    By default 2D variables are not parsed if all 2D varaibles are needed. See
    the MHKiT CDiP example Jupyter notbook for information on available parameters.


    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type. Can be the output of
        request_netCDF
    station_number: string
        Station number of CDIP wave buoy
    parameters: string or list of strings
        Parameters to return. If None will return all varaibles except
        2D-variables.
    years: int or list of int
        Year date, e.g. 2001 or [2001, 2010]
    start_date: string
        Start date in YYYY-MM-DD, e.g. '2012-04-01'
    end_date: string
        End date in YYYY-MM-DD, e.g. '2012-04-30'
    data_type: string
        Either 'historic' or 'realtime'
    all_2D_variables: boolean
        Will return all 2D data. Enabling this will add significant
        processing time. If all 2D variables are not needed it is
        recomended to pass 2D parameters of interest using the
        'parameters' keyword and leave this set to False. Default False.
    silent: boolean
        Set to True to prevent the print statement that announces when 2D
        variable processing begins. Default False.
    to_pandas: bool (optional)
        Flag to output a dictionary of pandas objects instead of a dictionary
        of xarray objects. Default = True.


    Returns
    -------
    data: dictionary
        'data': dictionary of variables
            'vars': pandas DataFrame or xarray Dataset
                1D variables indexed by time
            'vars2D': dictionary of DataFrames or Datasets, optional
                If 2D-vars are passed in the 'parameters key' or if run
                with all_2D_variables=True, then this key will appear
                with a dictonary of DataFrames of 2D variables.
        'metadata': dictionary
            Anything not of length time
    """
    if not isinstance(station_number, (str, type(None))):
        raise TypeError(
            f"station_number must be of type string. Got: {type(station_number)}"
        )

    if not isinstance(parameters, (str, type(None), list)):
        raise TypeError(
            f"parameters must be of type str or list of strings. Got: {type(parameters)}"
        )

    if start_date is not None:
        if isinstance(start_date, str):
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                start_date = start_date.replace(tzinfo=pytz.UTC)
            except ValueError as exc:
                raise ValueError("Incorrect data format, should be YYYY-MM-DD") from exc
        else:
            raise TypeError(f"start_date must be of type str. Got: {type(start_date)}")

    if end_date is not None:
        if isinstance(end_date, str):
            try:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                end_date = end_date.replace(tzinfo=pytz.UTC)
            except ValueError as exc:
                raise ValueError("Incorrect data format, should be YYYY-MM-DD") from exc
        else:
            raise TypeError(f"end_date must be of type str. Got: {type(end_date)}")

    if not isinstance(years, (type(None), int, list)):
        raise TypeError(
            f"years must be of type int or list of ints. Got: {type(years)}"
        )

    if not isinstance(data_type, str):
        raise TypeError(f"data_type must be of type string. Got: {type(data_type)}")

    if data_type not in ["historic", "realtime"]:
        raise ValueError(
            f'data_type must be "historic" or "realtime". Got: {data_type}'
        )

    if not any([nc, station_number]):
        raise ValueError("Must provide either a CDIP netCDF file or a station number.")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if not nc:
        nc = request_netCDF(station_number, data_type)

    # Define the path to the cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mhkit", "cdip")

    buoy_name = (
        nc.variables["metaStationName"][:].compressed().tobytes().decode("utf-8")
    )

    multiyear = False
    if years:
        if isinstance(years, int):
            start_date = datetime.datetime(years, 1, 1, tzinfo=pytz.UTC)
            end_date = datetime.datetime(years + 1, 1, 1, tzinfo=pytz.UTC)
        elif isinstance(years, list):
            if len(years) == 1:
                start_date = datetime.datetime(years[0], 1, 1, tzinfo=pytz.UTC)
                end_date = datetime.datetime(years[0] + 1, 1, 1, tzinfo=pytz.UTC)
            else:
                multiyear = True
    if not multiyear:
        # Check the cache first
        hash_params = f"{station_number}-{parameters}-{start_date}-{end_date}"
        data = handle_caching(hash_params, cache_dir)

        if data[:2] == (None, None):
            data = get_netcdf_variables(
                nc,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                all_2D_variables=all_2D_variables,
                silent=silent,
            )
            handle_caching(hash_params, cache_dir, data=data)
        else:
            data = data[0]

    else:
        data = {"data": {}, "metadata": {}}
        multiyear_data = {}
        for year in years:
            start_date = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)
            end_date = datetime.datetime(year + 1, 1, 1, tzinfo=pytz.UTC)

            # Check the cache for each individual year
            hash_params = f"{station_number}-{parameters}-{start_date}-{end_date}"
            year_data = handle_caching(hash_params, cache_dir)
            if year_data[:2] == (None, None):
                year_data = get_netcdf_variables(
                    nc,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters,
                    all_2D_variables=all_2D_variables,
                    silent=silent,
                )
                # Cache the individual year's data
                handle_caching(hash_params, cache_dir, data=year_data)
            else:
                year_data = year_data[0]
            multiyear_data[year] = year_data["data"]

        for data_key in year_data["data"].keys():
            if data_key.endswith("2D"):
                data["data"][data_key] = {}
                for data_key2D in year_data["data"][data_key].keys():
                    data_list = []
                    for year in years:
                        data2D = multiyear_data[year][data_key][data_key2D]
                        data_list.append(data2D)
                    data["data"][data_key][data_key2D] = pd.concat(data_list)
            else:
                data_list = [multiyear_data[year][data_key] for year in years]
                data["data"][data_key] = pd.concat(data_list)

    if buoy_name:
        try:
            data.setdefault("metadata", {})["name"] = buoy_name
        except:
            pass

    if not to_pandas:
        data = convert_nested_dict_and_pandas(data)

    return data


def get_netcdf_variables(
    nc,
    start_date=None,
    end_date=None,
    parameters=None,
    all_2D_variables=False,
    silent=False,
    to_pandas=True,
):
    """
    Iterates over and extracts variables from CDIP bouy data. See
    the MHKiT CDiP example Jupyter notbook for information on available
    parameters.

    Parameters
    ----------
    nc: netCDF Object
        netCDF data for the given station number and data type
    start_stamp: float
        Data of interest start in seconds since epoch
    end_stamp: float
        Data of interest end in seconds since epoch
    parameters: string or list of strings
        Parameters to return. If None will return all varaibles except
        2D-variables. Default None.
    all_2D_variables: boolean
        Will return all 2D data. Enabling this will add significant
        processing time. If all 2D variables are not needed it is
        recomended to pass 2D parameters of interest using the
        'parameters' keyword and leave this set to False. Default False.
    silent: boolean
        Set to True to prevent the print statement that announces when 2D
        variable processing begins. Default False.
    to_pandas: bool (optional)
        Flag to output a dictionary of pandas objects instead of a dictionary
        of xarray objects. Default = True.


    Returns
    -------
    results: dictionary
        'data': dictionary of variables
            'vars': pandas DataFrame or xarray Dataset
                1D variables indexed by time
            'vars2D': dictionary of DataFrames or Datasets, optional
                If 2D-vars are passed in the 'parameters key' or if run
                with all_2D_variables=True, then this key will appear
                with a dictonary of DataFrames/Datasets of 2D variables.
        'metadata': dictionary
            Anything not of length time
    """

    if not isinstance(nc, netCDF4.Dataset):
        raise TypeError("nc must be netCDF4 dataset. Got: {type(nc)}")

    if start_date and isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    if end_date and isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    if not isinstance(parameters, (str, type(None), list)):
        raise TypeError(
            "parameters must be of type str or list of strings. Got: {type(parameters)}"
        )

    if not isinstance(all_2D_variables, bool):
        raise TypeError(
            "all_2D_variables must be a boolean. Got: {type(all_2D_variables)}"
        )

    if parameters:
        if isinstance(parameters, str):
            parameters = [parameters]
        for param in parameters:
            if not isinstance(param, str):
                raise TypeError("All elements of parameters must be strings.")

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    buoy_name = (
        nc.variables["metaStationName"][:].compressed().tobytes().decode("utf-8")
    )

    allVariables = [var for var in nc.variables]
    allVariableSet = set(allVariables)

    twoDimensionalVars = [
        "waveEnergyDensity",
        "waveMeanDirection",
        "waveA1Value",
        "waveB1Value",
        "waveA2Value",
        "waveB2Value",
        "waveCheckFactor",
        "waveSpread",
        "waveM2Value",
        "waveN2Value",
    ]
    twoDimensionalVarsSet = set(twoDimensionalVars)

    # If parameters are provided, convert them into a set
    if parameters:
        params = set(parameters)
    else:
        params = set()

    # If all_2D_variables is True, add all 2D variables to params
    if all_2D_variables:
        params.update(twoDimensionalVarsSet)

    include_params = params & allVariableSet
    if params != include_params:
        not_found = params - include_params
        print(
            f"WARNING: {not_found} was not found in data.\n"
            f"Possible parameters are:\n {allVariables}"
        )

    include_params_2D = include_params & twoDimensionalVarsSet
    include_params -= include_params_2D

    include_2D_variables = bool(include_params_2D)
    if include_2D_variables:
        include_params.add("waveFrequency")

    include_vars = include_params

    # when parameters is None and all_2D_variables is False
    if not parameters and not all_2D_variables:
        include_vars = allVariableSet - twoDimensionalVarsSet

    start_stamp, end_stamp = _dates_to_timestamp(
        nc, start_date=start_date, end_date=end_date
    )

    prefixs = ["wave", "sst", "gps", "dwr", "meta"]
    variables_by_type = {
        prefix: [var for var in include_vars if var.startswith(prefix)]
        for prefix in prefixs
    }
    variables_by_type = {
        prefix: vars for prefix, vars in variables_by_type.items() if vars
    }

    results = {"data": {}, "metadata": {}}
    for prefix in variables_by_type:
        time_variables = {}
        metadata = {}

        if prefix != "meta":
            prefixTime = nc.variables[f"{prefix}Time"][:]

            masked_time = np.ma.masked_outside(prefixTime, start_stamp, end_stamp)
            mask = masked_time.mask
            var_time = masked_time.compressed()
            N_time = masked_time.size

            for var in variables_by_type[prefix]:
                variable = np.ma.filled(nc.variables[var])
                if variable.size == N_time:
                    variable = np.ma.masked_array(variable, mask).astype(float)
                    time_variables[var] = variable.compressed()
                else:
                    metadata[var] = nc.variables[var][:].compressed()

            time_slice = pd.to_datetime(var_time, unit="s")
            data = pd.DataFrame(time_variables, index=time_slice)
            results["data"][prefix] = data
            results["data"][prefix].name = buoy_name

        results["metadata"][prefix] = metadata

        if (prefix == "wave") and (include_2D_variables):
            if not silent:
                print("Processing 2D Variables:")

            vars2D = {}
            columns = metadata["waveFrequency"]
            N_time = len(time_slice)
            N_frequency = len(columns)
            try:
                l = len(mask)
            except:
                mask = np.array([False] * N_time)

            mask2D = np.tile(mask, (len(columns), 1)).T
            for var in include_params_2D:
                variable2D = nc.variables[var][:].data
                variable2D = np.ma.masked_array(variable2D, mask2D)
                variable2D = variable2D.compressed().reshape(N_time, N_frequency)
                variable = pd.DataFrame(variable2D, index=time_slice, columns=columns)
                vars2D[var] = variable
            results["data"]["wave2D"] = vars2D
    results["metadata"]["name"] = buoy_name

    if not to_pandas:
        results = convert_nested_dict_and_pandas(results)

    return results


def _process_multiyear_data(nc, years, parameters, all_2D_variables):
    """
    A helper function to process multiyear data.

    Parameters
    ----------
    nc : netCDF4.Dataset
        netCDF file containing the data
    years : list of int
        A list of years to process
    parameters : list of str
        A list of parameters to return
    all_2D_variables : bool
        Whether to return all 2D variables

    Returns
    -------
    data : dict
        A dictionary containing the processed data
    """

    data = {}
    for year in years:
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year + 1, 1, 1)

        year_data = get_netcdf_variables(
            nc,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            all_2D_variables=all_2D_variables,
        )
        data[year] = year_data

    return data
