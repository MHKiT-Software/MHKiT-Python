# import statements
import pandas as pd 
import numpy as np 
from datetime import timedelta

import matplotlib.pyplot as plt 

def get_stats(data,tname,stat,period,freq):
    """
    function used to obtain statistics from a dataset

    Parameters:
    ----------------------
    data : pandas dataframe or series
        dataframe containg data to be analyzed
    tname : string
        name of column containing time in datetime format
    stat : string input of mean, max, min, or std
        type of statistics to be calculated
    period : float/int
        statistical window of interest (ex. 600 seconds) [sec]
    freq : float/int
        sample rate of data [1/sec]
    
    Returns:
    ----------------------
    stats : pandas dataframe or series
        dataframe containing calculated statistical values of data
    """
    # check to see if data contains enough data points for statistical window
    # if len(data)%(period*freq) > 0:
    #     remain = len(data) % (period*freq)
    #     data = data.iloc[0:-int(remain)]
    #     print('WARNING: there were not enought data points in the last statistical period. Last '+str(remain)+' points were removed.')
    
    # convert period to timedelta
    td = timedelta(seconds=period)

    # calculate statistical parameter
    if stat == 'mean' : stats = data.resample(td,on=tname).mean()
    elif stat == 'max' : stats = data.resample(td,on=tname).max()
    elif stat == 'min' : stats = data.resample(td,on=tname).min()
    elif stat == 'std' : stats = data.resample(td,on=tname).std()
    elif stat == 'abs' : stats = data.resample(td,on=tname).abs().max()
    else:
        print('Invalid stat input')

    # drop first and last entry to make sure each stat window has enough points
    stats = stats.iloc[1:-1]

    return stats

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
    

def statsplotter_bin(x,varmean,std=None,varmax=None,varmin=None,xlabel=None,ylabel=None,title=None,savepath=None):
    fig, ax = plt.subplots(figsize=(8,6))






# def get_stats(data,period,freq):
#     """
#     function used to obtain statistics from a dataset

#     Parameters:
#     ----------------------
#     data : pandas dataframe

#     period : float/int
#         statistical window of interest (ex. 600 seconds) [sec]
#     freq : float/int
#         sample rate of data [1/sec]
    
#     Returns:
#     ----------------------
#     dfstats : pandas multi-index dataframe
#         dataframe containing calculated statistical values of data
#     """
#     # check to see if data contains enough data points for statistical window
#     if len(data)%(period*freq) > 0:
#         remain = len(data) % (period*freq)
#         data = data.iloc[0:-int(remain)]
#         print('WARNING: there were not enought data points in the last statistical period. Last '+str(remain)+' points were removed.')
    
#     # convert period to timedelta
#     td = timedelta(seconds=period)

#     # create multi-index
#     time = data.resample(td).mean().index
#     iterables = [['mean','max','min','std'], time]
#     index = pd.MultiIndex.from_product(iterables,names=['Stat','Time'])  

#     # calculate statistical parameter and create dataframe
#     dfstats = pd.DataFrame(columns=data.columns.values,index=index)
#     dfstats.loc['mean'] = data.resample(td).mean().values
#     dfstats.loc['max'] = data.resample(td).max().values
#     dfstats.loc['min'] = data.resample(td).min().values
#     dfstats.loc[('std')] = data.resample(td).std().values

#     return dfstats