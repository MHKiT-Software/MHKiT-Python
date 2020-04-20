# import statements
import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt 

def get_stats(data,freq,period=600):
    """
    function used to obtain statistics from a dataset

    Parameters:
    ----------------------
    data : pandas dataframe or series
        dataframe containg multiple or a single variable to be analyzed with statistical window
    period : float/int
        statistical window of interest (ex. 600 seconds) [sec]
    freq : float/int
        sample rate of data [1/sec]
    
    Returns:
    ----------------------
    means,maxs,mins,stds,absv : pandas dataframes or series
        dataframes containing calculated statistical values of data
    """
    # check to see if data contains enough data points for statistical window
    if len(data)%(period*freq) > 0:
        raise Exception('WARNING: there were not enought data points in the last statistical period. No stats calculated')
    
    time = data.time[0]   

    # calculate stats
    means = data.mean()
    maxs = data.max()
    mins = data.min()
    stds = data.std()
    #absv = data.abs().max()

    # TODO: handle if time variable exists
    # TODO: handle vector averaging

    # TODO: add statement to check output type
    return time, means.to_frame(),maxs.to_frame(),mins.to_frame(),stds.to_frame()

def unwrapvec(data):
    """
    function used to unwrap vectors into 0-360 deg range

    Parameters:
    ---------------
    data : dataframe or array
        list of data points to be unwrapped [deg]
    
    Returns:
    --------------
    data : dataframe or array
        returns list of data points unwrapped between 0-360 deg
    """
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = data[i]+360
        elif data[i] > 360:
            data[i] = data[i]-360
    if max(data) > 360 or min(data) < 0:
        data = unwrapvec(data)
    return data

# TODO: allow function to perform action on entire array
def matlab2datetime(matlab_datenum):
    """
    conversion of matlab datenum format to python datetime

    Parameters:
    ----------------
    matlab_datenum : float/int
        value of matlab datenum to be converted

    Returns:
    -----------------
    pdate : float/int
        equivalent python datetime value
    """
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    pdate = day + dayfrac
    return pdate


# TODO: add function description
def statplotter(x,vmean,vmax,vmin,xlabel=None,ylabel=None,title=None,savepath=None):
    fig, ax = plt.subplots(figsize=(6,4))
    s = 10
    ax.scatter(x,vmean,label='mean',s=s)
    ax.scatter(x,vmax,label='max',s=s)
    ax.scatter(x,vmin,label='min',s=s)
    ax.grid(alpha=0.4)
    ax.legend(loc='best')
    if xlabel!=None: ax.set_xlabel(xlabel)
    if ylabel!=None: ax.set_ylabel(ylabel)
    if title!=None: ax.set_title(title)
    fig.tight_layout()
    if savepath==None: plt.show()
    else: 
        fig.savefig(savepath)
        plt.close()
    

# def statsplotter_bin(x,varmean,std=None,varmax=None,varmin=None,xlabel=None,ylabel=None,title=None,savepath=None):
#     fig, ax = plt.subplots(figsize=(8,6))

