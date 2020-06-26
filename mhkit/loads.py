from scipy.stats import binned_statistic
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import fatpack


def bin_stats(df,x,bin_edges,statlist=[]):
    """
    Use to bin calculated statistics against "x" according to IEC
    
    Parameters:
    -----------------
    df : pd.Dataframe
        Matrix containing time-series statistics of variables
    
    x : array
        Array of variable to bin data against (ie. wind speed)
    
    bin_edges : array
        Array of desired bin edges w/ consistent step size. 

    statlist : list, optional 
        Names of variables to be binned. Bins all variables if left empty
    
    Returns:
    ----------------
    baverages : pd.DataFrame
        Load means of each bin

    bstd : pd.DataFrame
        Additional information related to the binning process
    """
    # check data types
    try:
        x = np.asarray(x)
        bin_edges = np.asarray(bin_edges)
    except:
        pass
    assert isinstance(df, pd.DataFrame), 'df must be of type pd.DataFrame'
    assert isinstance(x, np.ndarray), 'x must be of type np.ndarray'
    assert isinstance(bin_edges, np.ndarray), 'bin_edges must be of type np.ndarray'

    # determine variables to analyze
    if len(statlist)==0: # if not specified, bin all variables
        statlist=df.columns.values
    else:
        assert isinstance(statlist, list), 'stat must be of type list'

    # pre-allocate list variables
    bstatlist = []
    bstdlist = []

    # loop through statlist and get binned means
    for chan in statlist:
        # bin data
        bstat = binned_statistic(x,df[chan],statistic='mean',bins=bin_edges)
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


################ fatigue functions

def damage_equivalent_load(var, m, bin_num=100, t=600):
    """ Calculates the damage equivalent load of a single variable
    
    Parameters: 
    -----------
    var : array
        Data of variable/channel being analyzed
    
    m : float/int
        Fatigue slope factor of material
    
    bin_num : int
        Number of bins for rainflow counting method (minimum=100)
    
    t : float/int
        Length of measured data (seconds)
    
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
    assert isinstance(bin_num, (float,int)), 'bin_num must be of type float or int'
    assert isinstance(t, (float,int)), 't must be of type float or int'

    # find rainflow ranges
    ranges = fatpack.find_rainflow_ranges(var)

    # find range count and bin
    Nrf, Srf = fatpack.find_range_count(ranges, bin_num)

    # get DEL
    DELs = Srf**m * Nrf / t
    DEL = DELs.sum() ** (1/m)

    return DEL

    
################ plotting functions

def plot_statistics(x,vmean,vmax,vmin,vstdev=[],xlabel=None,ylabel=None,title=None,savepath=None):
    """
    plot showing standard raw statistics of variable

    Parameters:
    ------------------
    x : numpy array
        Array of x-axis values
    vmean : numpy array
        Array of mean statistical values of variable
    vmax : numpy array
        Array of max statistical values of variable
    vmin : numpy array
        Array of min statistical values of variable
    vstdev : numpy array, optional
        Array of standard deviation statistical values of variable
    xlabel : string, optional
        xlabel to plot
    ylabel : string, optional
        ylabel to plot
    title : string, optional
        title to plot
    savepath : string, optional
        Path and filename to save figure. Plt.show() is called otherwise

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


def plot_bin_statistics(bcenters,bmean,bmax,bmin,bstdmean,bstdmax,bstdmin,xlabel=None,ylabel=None,title=None,savepath=None):
    """
    Plot showing standard binned statistics of single variable

    Parameters:
    ------------------
    bcenters : numpy array
        x-axis bin center values
    bmean : numpy array
        Binned mean statistical values of variable
    bmax : numpy array
        Binned max statistical values of variable
    bmin : numpy array
        Binned min statistical values of variable
    bstdmean : numpy array
        Standard deviations of mean binned statistics
    bstdmax : numpy array
        Standard deviations of max binned statistics
    bstdmin : numpy array
        Standard deviations of min binned statistics
    xlabel : string, optional
        xlabel for plot
    ylabel : string, optional
        ylabel for plot
    title : string, optional
        Title for plot
    savepath : string, optional
        Path and filename to save figure. Plt.show() is used by default.

    Returns:
    -------------------
    figure
    """
    fig, ax = plt.subplots(figsize=(7,5))
    ax.errorbar(bcenters,bmax,marker='^',mfc='none',yerr=bstdmax,capsize=4,label='max')
    ax.errorbar(bcenters,bmean,marker='o',mfc='none',yerr=bstdmean,capsize=4,label='mean')
    ax.errorbar(bcenters,bmin,marker='v',mfc='none',yerr=bstdmin,capsize=4,label='min')
    ax.grid(alpha=0.5)
    ax.legend(loc='best')
    if xlabel!=None: ax.set_xlabel(xlabel)
    if ylabel!=None: ax.set_ylabel(ylabel)
    if title!=None: ax.set_title(title)
    fig.tight_layout()
    if savepath==None: plt.show()
    else: 
        fig.savefig(savepath)
        plt.close()
