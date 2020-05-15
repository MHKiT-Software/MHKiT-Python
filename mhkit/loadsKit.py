# import statements
import pandas as pd 
import numpy as np
import fatpack #### this is a 3rd party module used for rainflow counting
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt 

################ general functions

def bin_stats(df,x,maxX,statlist=[],bwidth=0.1):
    """
    use to bin calculated statistics against "x" according to IEC
    
    Parameters:
    -----------------
    df : pd.Dataframe
        matrix containing time-series statistics of variables
    
    x : array
        contains array of variable to bin data against (ie. wind speed)
    
    statlist : list, optional 
        contains names of variables to be binned. Bins all variables if left empty
    
    maxX : float/int
        maximum value used for creating rightmost bin edge of x

    bwidth : float/int, optional 
        width of bins

    Returns:
    ----------------
    baverages : array
        array containing the load means of each bin

    bstd : array
        object containing additional information related to the binning process
    """
    # check data types
    try:
        x = np.asarray(x)
    except:
        pass
    assert isinstance(df, pd.DataFrame), 'df must be of type pd.DataFrame'
    assert isinstance(x, np.ndarray), 'x must be of np.ndarray'
    assert isinstance(maxX, (float,int)), 'maxX must be of type float or int'
    assert isinstance(bwidth, (float,int)), 'bwidth must be of type float or int'

    # determine variables to analyze
    if len(statlist)==0: # if not specified, bin all variables
        statlist=df.columns.values
    else:
        assert isinstance(statlist, list), 'stat must be of type list'

    # create bin edges with step size = bwidth
    bedges = list(range(0,maxX+1,bwidth))

    # pre-allocate list variables
    bstatlist = []
    bstdlist = []

    # loop through statlist and get binned means
    for chan in statlist:
        # bin data
        bstat = binned_statistic(x,df[chan],statistic='mean',bins=bedges)
        # get std of bins
        std = []
        stdev = pd.DataFrame(df[chan])
        stdev.set_index(bstat.binnumber,inplace=True)
        for i in range(1,len(bstat.bin_edges)):
            try:
                temp = stdev.loc[i].std(ddof=0)
                std.append(temp[0])
            except:
                std.append(np.nan)
        bstatlist.append(bstat.statistic)
        bstdlist.append(std)
 
    # convert return variables to dataframes
    baverages = pd.DataFrame(np.transpose(bstatlist),columns=statlist)
    bstd = pd.DataFrame(np.transpose(bstdlist),columns=statlist)

    # check if any nans exist
    if baverages.isna().any().any():
        print('Warning: some bins may be empty!')

    return baverages, bstd


################ ultimate loads functions

# TODO: ranked loads and/or ultimate loads per wind speed



################ fatigue functions

def get_DELs(df, chan_dict, binNum=100, t=600):
    """ Calculates the damage equivalent load of multiple variables
    
    Parameters: 
    -----------
    df : pd.DataFrame
        contains dataframe of variables/channels being analyzed
    
    chan_dict : list, tuple
        tuple/list containing channel names to be analyzed and corresponding fatigue slope factor "m"
        ie. ('TwrBsFxt',4)
        
    binNum : int
        number of bins for rainflow counting method (minimum=100)
    
    t : float/int
        Used to control DEL frequency. Default for 1Hz is 600 seconds for 10min data
        
    Returns:
    -----------
    dfDEL : pd.DataFrame
        Damage equivalent load of each specified variable  
    
    """
    # check data types
    assert isinstance(df, pd.DataFrame), 'df must be of type pd.DataFrame'
    assert isinstance(chan_dict, (list,tuple)), 'chan_dict must be of type list or tuple'
    assert isinstance(binNum, (float,int)), 'binNum must be of type float or int'
    assert isinstance(t, (float,int)), 't must be of type float or int'

    # create dictionary from chan_dict
    dic = dict(chan_dict)

    # pre-allocate list
    dflist = []

    # loop through channels and apply corresponding fatigue slope
    for var in dic.keys():
        # find rainflow ranges
        ranges = fatpack.find_rainflow_ranges(df[var])

        # find range count and bin
        Nrf, Srf = fatpack.find_range_count(ranges,binNum)

        # get DEL
        DELs = Srf**dic[var] * Nrf / t
        DEL = DELs.sum() ** (1/dic[var])
        dflist.append(DEL)
    
    # create dataframe to return
    dfDEL = pd.DataFrame(np.transpose(dflist))
    dfDEL = dfDEL.T
    dfDEL.columns = dic.keys()

    return dfDEL

    
################ plotting functions

def statplotter(x,vmean,vmax,vmin,vstdev=[],xlabel=None,ylabel=None,title=None,savepath=None):
    """
    plot showing standard raw statistics of variable

    Parameters:
    ------------------
    x : numpy array
        array of x-axis values
    vmean : numpy array
        array of mean statistical values of variable
    vmax : numpy array
        array of max statistical values of variable
    vmin : numpy array
        array of min statistical values of variable
    vstdev : numpy array, optional
        array of standard deviation statistical values of variable
    xlabel : string, optional
        add xlabel to plot
    ylabel : string, optional
        add ylabel to plot
    title : string, optional
        add title to plot
    savepath : string, optional
        path and filename to save figure. Plt.show() is called otherwise

    Returns:
    -------------------
    figure
    """
    # check data type
    try:
        x = np.array(x)
        vmean = np.array(vmean)
        vmax = np.array(vmax)
        vmin = np.array(vmin)
    except:
        pass
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'
    assert isinstance(vmean, np.ndarray), 'vmean must be of type np.ndarray'
    assert isinstance(vmax, np.ndarray), 'vmax must be of type np.ndarray'
    assert isinstance(vmin, np.ndarray), 'vmin must be of type np.ndarray'

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,vmax,'^',label='max',mfc='none')
    ax.plot(x,vmean,'o',label='mean',mfc='none')
    ax.plot(x,vmin,'v',label='min',mfc='none')
    if len(vstdev)>0: ax.plot(x,vstdev,'+',label='stdev',c='m')
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
