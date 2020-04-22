# import statements
import pandas as pd 
import numpy as np
import fatpack #### this is a 3rd party module used for rainflow counting
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt 

################ general functions

def bin_stats(x,stat,maxX,bwidth=0.1):
    """
    use to bin calculated statistics into current speed bins according to IEC
    
    Parameters:
    -----------------
    x : array
        array containing values to bin against
    
    stat : array 
        array containing statistics of load variable in question
    
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
        x = np.array(x)
        stat = np.array(stat)
    except:
        pass
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'
    assert isinstance(stat, np.ndarray), 'stat must be of type np.ndarray'
    assert isinstance(maxX, (float,int)), 'maxX must be of type float or int'
    assert isinstance(bwidth, (float,int)), 'bwidth must be of type float or int'


    # create bin edges with step size = bwidth
    bedges = list(range(0,maxX,bwidth))

    # bin data
    bstat = binned_statistic(x,stat,statistic='mean',bins=bedges)

    # get std of bins
    std = []
    stdev = pd.DataFrame(stat,index=bstat.binnumber)
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
    """ Calculates the damage equivalent load of a single variable
    
    Parameters: 
    -----------
    var : array
        contains data of variable/channel being analyzed
    
    m : float/int
        fatigue slope factor of material
    
    binNum : int
        number of bins for rainflow counting method (minimum=100)
    
    t : float/int
        length of measured data (seconds)
    
    Returns:
    -----------
    DEL : float
        Damage equivalent load of single variable  
    """
    # check data types
    try:
        var = np.array(var)
    except:
        pass
    assert isinstance(var, np.ndarray), 'var must be of type np.ndarray'
    assert isinstance(m, (float,int)), 'm must be of type float or int'
    assert isinstance(binNum, (float,int)), 'binNum must be of type float or int'
    assert isinstance(t, (float,int)), 't must be of type float or int'

    # find rainflow ranges
    ranges = fatpack.find_rainflow_ranges(var)

    # find range count and bin
    Nrf, Srf = fatpack.find_range_count(ranges,binNum)

    # get DEL
    DELs = Srf**m * Nrf / t
    DEL = DELs.sum() ** (1/m)

    return DEL

    
################ plotting functions

def statplotter(x,vmean,vmax,vmin,xlabel=None,ylabel=None,title=None,savepath=None):
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
    assert isinstance(xlabel, str), 'xlabel must be of type str'
    assert isinstance(ylabel, str), 'ylabel must be of type str'
    assert isinstance(title, str), 'title must be of type str'
    assert isinstance(savepath, str), 'savepath must be of type str'

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
