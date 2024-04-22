from scipy.io import loadmat
from os.path import isfile
import pandas as pd
import xarray as xr
import numpy as np
import re
from mhkit.utils import convert_to_dataset, convert_nested_dict_and_pandas


def read_table(swan_file, to_pandas=True):
    """
    Reads in SWAN table format output

    Parameters
    ----------
    swan_file: str
        filename to import
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    swan_data: pandas DataFrame or xarray Dataset
        Dataframe of swan output
    metaDict: Dictionary
        Dictionary of metaData
    """
    if not isinstance(swan_file, str):
        raise TypeError(f"swan_file must be of type str. Got: {type(swan_file)}")
    if not isfile(swan_file):
        raise ValueError(f"File not found: {swan_file}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    f = open(swan_file, "r")
    header_line_number = 4
    for i in range(header_line_number + 2):
        line = f.readline()
        if line.startswith("% Run"):
            metaDict = _parse_line_metadata(line)
            if metaDict["Table"].endswith("SWAN"):
                metaDict["Table"] = metaDict["Table"].split(" SWAN")[:-1]
        if i == header_line_number:
            header = re.split("\s+", line.rstrip().strip("%").lstrip())
            metaDict["header"] = header
        if i == header_line_number + 1:
            units = re.split(
                "\s+", line.strip(" %\n").replace("[", "").replace("]", "")
            )
            metaDict["units"] = units
    f.close()

    swan_data = pd.read_csv(swan_file, sep="\s+", comment="%", names=metaDict["header"])

    if not to_pandas:
        swan_data = convert_to_dataset(swan_data)

    return swan_data, metaDict


def read_block(swan_file, to_pandas=True):
    """
    Reads in SWAN block output with headers and creates a dictionary
    of DataFrames or Datasets for each SWAN output variable in the output file.

    Parameters
    ----------
    swan_file: str
        swan block file to import
    to_pandas: bool (optional)
        Flag to output a dictionary of pandas objects instead of a dictionary
        of xarray objects. Default = True.

    Returns
    -------
    data: Dictionary
        Dictionary of DataFrames or Datasets of swan output variables
    metaDict: Dictionary
        Dictionary of metaData dependent on file type
    """
    if not isinstance(swan_file, str):
        raise TypeError(f"swan_file must be of type str. Got: {type(swan_file)}")
    if not isfile(swan_file):
        raise ValueError(f"File not found: {swan_file}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    extension = swan_file.split(".")[1].lower()
    if extension == "mat":
        dataDict = _read_block_mat(swan_file)
        metaData = {"filetype": "mat", "variables": [var for var in dataDict.keys()]}
    else:
        dataDict, metaData = _read_block_txt(swan_file)

    if not to_pandas:
        dataDict = convert_nested_dict_and_pandas(dataDict)

    return dataDict, metaData


def _read_block_txt(swan_file):
    """
    Reads in SWAN block output with headers and creates a dictionary
    of DataFrames for each SWAN output variable in the output file.

    Parameters
    ----------
    swan_file: str
        swan block file to import (must be written with headers)

    Returns
    -------
    dataDict: Dictionary
        Dictionary of DataFrame of swan output variables
    metaDict: Dictionary
        Dictionary of metaData dependent on file type
    """
    if not isinstance(swan_file, str):
        raise TypeError(f"swan_file must be of type str. Got: {type(swan_file)}")
    if not isfile(swan_file):
        raise ValueError(f"File not found: {swan_file}")

    f = open(swan_file)
    runLines = []
    metaDict = {}
    column_position = None
    dataDict = {}
    for position, line in enumerate(f):
        if line.startswith("% Run"):
            varPosition = position
            runLines.extend([position])
            column_position = position + 5
            varDict = _parse_line_metadata(line)
            varDict["unitMultiplier"] = float(varDict["Unit"].split(" ")[0])

            metaDict[varPosition] = varDict
            variable = varDict["vars"]
            dataDict[variable] = {}

        if position == column_position and column_position != None:
            columns = line.strip("% \n").split()
            metaDict[varPosition]["cols"] = columns
            N_columns = len(columns)
            columns_position = None

        if not line.startswith("%"):
            raw_data = " ".join(re.split(" |\.", line.strip(" \n"))).split()
            index_number = int(raw_data[0])
            columns_data = raw_data[1:]
            data = []
            possibleNaNs = ["****"]
            NNaNsTotal = sum([line.count(nanVal) for nanVal in possibleNaNs])

            if NNaNsTotal > 0:
                for vals in columns_data:
                    NNaNs = 0
                    for nanVal in possibleNaNs:
                        NNaNs += vals.count(nanVal)
                    if NNaNs > 0:
                        for i in range(NNaNs):
                            data.extend([np.nan])
                    else:
                        data.extend([float(vals)])
            else:
                data.extend([float(val) for val in columns_data])

            dataDict[variable][index_number] = data

    metaData = pd.DataFrame(metaDict).T
    f.close()

    for var in metaData.vars.values:
        df = pd.DataFrame(dataDict[var]).T
        varCols = metaData[metaData.vars == var].cols.values.tolist()[0]
        colsDict = dict(zip(df.columns.values.tolist(), varCols))
        df.rename(columns=colsDict)
        unitMultiplier = metaData[metaData.vars == var].unitMultiplier.values[0]
        dataDict[var] = df * unitMultiplier

    metaData.pop("cols")
    metaData = metaData.set_index("vars").T.to_dict()
    return dataDict, metaData


def _read_block_mat(swan_file):
    """
    Reads in SWAN matlab output and creates a dictionary of DataFrames
    for each swan output variable.

    Parameters
    ----------
    swan_file: str
        filename to import

    Returns
    -------
    dataDict: Dictionary
        Dictionary of DataFrame of swan output variables
    """
    if not isinstance(swan_file, str):
        raise TypeError(f"swan_file must be of type str. Got: {type(swan_file)}")
    if not isfile(swan_file):
        raise ValueError(f"File not found: {swan_file}")

    dataDict = loadmat(swan_file, struct_as_record=False, squeeze_me=True)
    removeKeys = ["__header__", "__version__", "__globals__"]
    for key in removeKeys:
        dataDict.pop(key, None)
    for key in dataDict.keys():
        dataDict[key] = pd.DataFrame(dataDict[key])
    return dataDict


def _parse_line_metadata(line):
    """
    Parses the variable metadata into a dictionary

    Parameters
    ----------
    line: str
        line from block swan data to parse

    Returns
    -------
    metaDict: Dictionary
        Dictionary of variable metadata
    """
    if not isinstance(line, str):
        raise TypeError(f"line must be of type str. Got: {type(line)}")

    metaDict = {}
    meta = re.sub(
        "\s+", " ", line.replace(",", " ").strip("% \n").replace("**", "vars:")
    )
    mList = meta.split(":")
    elms = [elm.split(" ") for elm in mList]
    for elm in elms:
        try:
            elm.remove("")
        except:
            pass
    for i in range(len(elms) - 1):
        elm = elms[i]
        key = elm[-1]
        val = " ".join(elms[i + 1][:-1])
        metaDict[key] = val
    metaDict[key] = " ".join(elms[-1])

    return metaDict


def dictionary_of_block_to_table(dictionary_of_DataFrames, names=None, to_pandas=True):
    """
    Converts a dictionary of structured 2D grid SWAN block format
    x (columns),y (index) to SWAN table format x (column),y (column),
    values (column) DataFrame or Dataset.

    Parameters
    ----------
    dictionary_of_DataFrames: Dictionary
        Dictionary of DataFrames in with columns as X indicie and Y as index.
    names: List (Optional)
        Name of data column in returned table. Default=Dictionary.keys()
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    swanTables: pandas DataFrame or xarray Dataset
        DataFrame/Dataset with columns x,y,values where values = Dictionary.keys()
        or names
    """
    if not isinstance(dictionary_of_DataFrames, dict):
        raise TypeError(
            f"dictionary_of_DataFrames must be of type dict. Got: {type(dictionary_of_DataFrames)}"
        )
    if not bool(dictionary_of_DataFrames):
        raise ValueError(
            f"dictionary_of_DataFrames is empty. Got: {dictionary_of_DataFrames}"
        )
    for key in dictionary_of_DataFrames:
        if not isinstance(dictionary_of_DataFrames[key], pd.DataFrame):
            raise TypeError(
                f"Dictionary key:{key} must be of type pd.DataFrame. Got: {type(dictionary_of_DataFrames[key])}"
            )
    if not isinstance(names, type(None)):
        if not isinstance(names, list):
            raise TypeError(
                f"If specified, names must be of type list. Got: {type(names)}"
            )
        if not all([isinstance(elm, str) for elm in names]):
            raise ValueError(
                f"If specified, all elements in names must be of type string. Got: {names}"
            )
        if not len(names) == len(dictionary_of_DataFrames):
            raise ValueError(
                "If specified, names must the same length as dictionary_of_DataFrames"
            )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    if names == None:
        variables = [var for var in dictionary_of_DataFrames.keys()]
    else:
        variables = names

    var0 = variables[0]
    swanTables = block_to_table(dictionary_of_DataFrames[var0], name=var0)
    for var in variables[1:]:
        tmp_dat = block_to_table(dictionary_of_DataFrames[var], name=var)
        swanTables[var] = tmp_dat[var]

    if not to_pandas:
        swanTables = convert_to_dataset(swanTables)

    return swanTables


def block_to_table(data, name="values", to_pandas=True):
    """
    Converts structured 2D grid SWAN block format x (columns), y (index)
    to SWAN table format x (column),y (column), values (column)
    DataFrame.

    Parameters
    ----------
    data: pandas DataFrame or xarray Dataset
        DataFrame in with columns as X indicie and Y as index.
    name: string (Optional)
        Name of data column in returned table. Default='values'
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    table: pandas DataFrame or xarray Dataset
        DataFrame with columns x,y,values
    """
    if isinstance(data, xr.Dataset):
        data = data.to_pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be of type pd.DataFrame. Got: {type(data)}")
    if not isinstance(name, str):
        raise TypeError(f"If specified, name must be of type str. Got: {type(name)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    table = data.unstack().reset_index(name=name)
    table = table.rename(columns={"level_0": "x", "level_1": "y"})
    table.sort_values(["x", "y"], ascending=[True, True], inplace=True)

    if not to_pandas:
        table = convert_to_dataset(table)

    return table
