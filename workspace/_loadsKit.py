# import statements
import pandas as pd 
import numpy as np
import fatpack #### this is a 3rd party module used for rainflow counting
from scipy.stats import binned_statistic

################ general functions

def bin_stats(cv,stat,maxcv,bwidth=0.1):
    """
    use to bin calculated statistics into current speed bins according to IEC
    
    Parameters:
    -----------------
    cv : array
        array containing mean current speed stat values in m/s
    
    stat : array 
        array containing statistics of load variable in question
    
    maxcv : float/int
        maximum current speed used for creating rightmost bin edge

    bwidth : float/int
        width of bin (m/s)

    Returns:
    ----------------
    baverages : array
        array containing the load means of each current speed bin

    bstat : statistical object
        object containing additional information related to the binning process
    """

    # create bin edges for current speed with step size = 0.1m/s per IEC standard
    bedges = list(range(0,maxcv,bwidth))

    # bin data
    bstat = binned_statistic(cv.reset_index(drop=True),stat,statistic='mean',bins=bedges)

    # get std of bins
    std = []
    stat = stat.to_frame() # need to test if this is necessary
    stdev = stat.set_index(bstat.binnumber)
    for i in range(1,len(bstat.bin_edges)):
        x = stdev.loc[i].std(ddof=0)
        std.append(x[0])

    # extract load means of each bin
    baverages = bstat.statistic

    # convert return variables to dataframes
    baverages = pd.DataFrame(baverages)
    bstd = pd.DataFrame(std)

    return baverages, bstd


################ ultimate loads functions





################ fatigue functions

def get_DEL(var, m, binNum=100, t=600):
    """ Calculates the 1Hz damage equivalent lifetime load of a variable
    
    Parameters: 
    -----------
    var : array
        contains data of variable/channel being analyzed (should be 1 Hz)
    
    m : float/int
        fatigue slope factor of material
    
    binNum : int
        number of bins for rainflow counting method (minimum=100)
    
    t : float/int
        length of measured data (seconds)
    
    Returns:
    -----------
    DEL : float
        1Hz damage equivalent load of chosen variable  
    """
    # find rainflow ranges
    ranges = fatpack.find_rainflow_ranges(var)

    # find range count and bin
    Nrf, Srf = fatpack.find_range_count(ranges,binNum)

    # get DEL
    DELs = Srf**m * Nrf / t
    DEL = DELs.sum() ** (1/m)

    # TODO: figure out if this is necessary
    #DEL = pd.DataFrame(DEL)

    return DEL

    

