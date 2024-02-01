"""
io.py

This module provides functions to read and parse MoorDyn output files.

The main function read_moordyn takes as input the path to a MoorDyn output file and optionally
the path to a MoorDyn input file. It reads the data from the output file, stores it in an 
xarray dataset, and then if provided, parses the input file for additional metadata to store 
as attributes in the dataset.

The helper function _moordyn_input is used to parse the MoorDyn output file. It loops through 
each line in the output file, parses various sets of properties and parameters, and stores 
them as attributes in the provided dataset.

Typical usage example:

    dataset = read_moordyn(filepath="FAST.MD.out", input_file="FAST.MD.input")
"""

import os
import pandas as pd


def read_moordyn(filepath, input_file=None):
    """
    Reads in MoorDyn OUT files such as "FAST.MD.out" and
    "FAST.MD.Line1.out" and stores inside xarray. Also allows for
    parsing and storage of MoorDyn input file as attributes inside
    the xarray.

    Parameters
    ----------
    filepath : str
        Path to MoorDyn OUT file
    inputfile : str (optional)
        Path to MoorDyn input file

    Returns
    -------
    xr.Dataset
        Dataset containing parsed MoorDyn OUT file

    Raises
    ------
    TypeError
        Checks for correct input types for filepath and input_file
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be of type str")
    if input_file:
        if not isinstance(input_file, str):
            raise TypeError("input_file must be of type str")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No file found at provided path: {filepath}")

    data = pd.read_csv(
        filepath, header=0, skiprows=[1], sep=" ", skipinitialspace=True, index_col=0
    )
    data = data.dropna(axis=1)
    dataset = data.to_xarray()

    if input_file:
        dataset = _moordyn_input(input_file, dataset)

    return dataset


def _moordyn_input(input_file, dataset):
    """
    Internal function used to parse MoorDyn input file and write attributes.

    Parameters
    ----------
    input_file : str
        Path to moordyn input file
    dataset : xr.Dataset
        xarray Dataset to be written to

    Returns
    -------
    xr.Dataset
        return Dataset that includes input file parameters as attributes
    """

    with open(input_file, "r", encoding="utf-8") as moordyn_file:
        for line in moordyn_file:  # loop through each line in the file
            # get line type property sets
            if line.count("---") > 0 and (
                line.upper().count("LINE DICTIONARY") > 0
                or line.upper().count("LINE TYPES") > 0
            ):
                linetypes = dict()
                # skip this header line, plus channel names and units lines
                line = next(moordyn_file)
                variables = line.split()
                line = next(moordyn_file)
                units = line.split()
                line = next(moordyn_file)
                while line.count("---") == 0:
                    entries = line.split()
                    linetypes[entries[0]] = dict()
                    for x in range(1, len(entries)):
                        linetypes[entries[0]][variables[x]] = entries[x]
                    line = next(moordyn_file)
                linetypes["units"] = units[1:]
                dataset.attrs["LINE_TYPES"] = linetypes

            # get properties of each Point
            if line.count("---") > 0 and (
                line.upper().count("POINTS") > 0
                or line.upper().count("POINT LIST") > 0
                or line.upper().count("POINT PROPERTIES") > 0
            ):
                # skip this header line, plus channel names and units lines
                line = next(moordyn_file)
                variables = line.split()
                line = next(moordyn_file)
                units = line.split()
                line = next(moordyn_file)
                points = dict()
                while line.count("---") == 0:
                    entries = line.split()
                    points[entries[0]] = dict()
                    for x in range(1, len(entries)):
                        points[entries[0]][variables[x]] = entries[x]
                    line = next(moordyn_file)
                points["units"] = units[1:]
                dataset.attrs["POINTS"] = points

            # get properties of each line
            if line.count("---") > 0 and (
                line.upper().count("LINES") > 0
                or line.upper().count("LINE LIST") > 0
                or line.upper().count("LINE PROPERTIES") > 0
            ):
                # skip this header line, plus channel names and units lines
                line = next(moordyn_file)
                variables = line.split()
                line = next(moordyn_file)
                units = line.split()
                line = next(moordyn_file)
                lines = {}
                while line.count("---") == 0:
                    entries = line.split()
                    lines[entries[0]] = dict()
                    for x in range(1, len(entries)):
                        lines[entries[0]][variables[x]] = entries[x]
                    line = next(moordyn_file)
                lines["units"] = units[1:]
                dataset.attrs["LINES"] = lines

            # get options entries
            if line.count("---") > 0 and "options" in line.lower():
                line = next(moordyn_file)  # skip this header line
                options = {}
                while line.count("---") == 0:
                    entries = line.split()
                    options[entries[1]] = entries[0]
                    line = next(moordyn_file)
                dataset.attrs["OPTIONS"] = options

    moordyn_file.close()

    return dataset
