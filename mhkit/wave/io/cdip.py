import pandas as pd
import numpy as np

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import calendar


def read_file(file_name, missing_values=['MM',9999,999,99]):
    """
    Reads a CDIP wave buoy data file (from http://cdip.ucsd.edu/).
    

    Parameters
    ------------
    file_name: string
        Name of NDBC wave buoy data file
    
    missing_value: list of values
        List of values that denote missing data    
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to header row 
        
    metadata: dict or None
        Dictionary with {column name: units} key value pairs when the CDIP file  
        contains unit information, otherwise None is returned
    """

    
    return data, metadata



