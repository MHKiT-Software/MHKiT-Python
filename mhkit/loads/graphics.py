import matplotlib.pyplot as plt
import numpy as np

def plot_statistics(x,y_mean,y_max,y_min,y_stdev=[],**kwargs):
    '''
    Plot showing standard raw statistics of variable

    Parameters
    -----------
    x : numpy array
        Array of x-axis values
    y_mean : numpy array
        Array of mean statistical values of variable
    y_max : numpy array
        Array of max statistical values of variable
    y_min : numpy array
        Array of min statistical values of variable
    y_stdev : numpy array, optional
        Array of standard deviation statistical values of variable
    **kwargs : optional             
        x_label : string
            x axis label for plot
        y_label : string
            y axis label for plot
        title : string, optional
            Title for plot
        save_path : string
            Path and filename to save figure.

    Returns
    --------
    ax : matplotlib pyplot axes
    '''
    
    try: x = np.array(x)
    except: 'x must be of type np.ndarray'       
    try: y_mean = np.array(y_mean)
    except: 'y_mean must be of type np.ndarray'           
    try:y_max = np.array(y_max)
    except: 'y_max must be of type np.ndarray'
    try: y_min = np.array(y_min)
    except: 'y_min must be of type np.ndarray'
    
    x_label   = kwargs.get("x_label", None)
    y_label   = kwargs.get("y_label", None)
    title     = kwargs.get("title", None)
    save_path = kwargs.get("save_path", None)
    
    assert isinstance(x_label, (str, type(None))), 'x_label must be of type str'
    assert isinstance(y_label, (str, type(None))), 'y_label must be of type str'
    assert isinstance(title, (str, type(None))), 'title must be of type str'
    assert isinstance(save_path, (str, type(None))), 'save_path must be of type str'

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,y_max,'^',label='max',mfc='none')
    ax.plot(x,y_mean,'o',label='mean',mfc='none')
    ax.plot(x,y_min,'v',label='min',mfc='none')
    
    if len(y_stdev)>0: ax.plot(x,y_stdev,'+',label='stdev',c='m')
    ax.grid(alpha=0.4)
    ax.legend(loc='best')
    
    if x_label!=None: ax.set_xlabel(x_label)
    if y_label!=None: ax.set_ylabel(y_label)
    if title!=None: ax.set_title(title)
    
    fig.tight_layout()
    
    if save_path==None: plt.show()
    else: 
        fig.savefig(save_path)
        plt.close()
    return ax


def plot_bin_statistics(bin_centers, bin_mean,bin_max, bin_min,
                        bin_mean_std, bin_max_std, bin_min_std,
                        **kwargs):
    '''
    Plot showing standard binned statistics of single variable

    Parameters
    -----------
    bin_centers : numpy array
        x-axis bin center values
    bin_mean : numpy array
        Binned mean statistical values of variable
    bin_max : numpy array
        Binned max statistical values of variable
    bin_min : numpy array
        Binned min statistical values of variable
    bin_mean_std : numpy array
        Standard deviations of mean binned statistics
    bin_max_std : numpy array
        Standard deviations of max binned statistics
    bin_min_std : numpy array
        Standard deviations of min binned statistics
    **kwargs : optional             
        x_label : string
            x axis label for plot
        y_label : string
            y axis label for plot
        title : string, optional
            Title for plot
        save_path : string
            Path and filename to save figure.

    Returns
    --------
    ax : matplotlib pyplot axes
    '''
        
    try: bin_centers = np.asarray(bin_centers)
    except: 'bin_centers must be of type np.ndarray'    
    
    try: bin_mean = np.asarray(bin_mean)
    except: 'bin_mean must be of type np.ndarray'    
    try: bin_max = np.asarray(bin_max)
    except:'bin_max must be of type np.ndarray'    
    try: bin_min = np.asarray(bin_min) 
    except: 'bin_min must be of type type np.ndarray'
    
    try: bin_mean_std = np.asarray(bin_mean_std)
    except: 'bin_mean_std must be of type np.ndarray'
    try: bin_max_std = np.asarray(bin_max_std)
    except: 'bin_max_std must be of type np.ndarray'
    try: bin_min_std = np.asarray(bin_min_std)
    except: 'bin_min_std must be of type np.ndarray'
    
    x_label   = kwargs.get("x_label", None)
    y_label   = kwargs.get("y_label", None)
    title     = kwargs.get("title", None)
    save_path = kwargs.get("save_path", None)
    
    assert isinstance(x_label, (str, type(None))), 'x_label must be of type str'
    assert isinstance(y_label, (str, type(None))), 'y_label must be of type str'
    assert isinstance(title, (str, type(None))), 'title must be of type str'
    assert isinstance(save_path, (str, type(None))), 'save_path must be of type str'
    
    fig, ax = plt.subplots(figsize=(7,5))
    ax.errorbar(bin_centers,bin_max,marker='^',mfc='none',
                yerr=bin_max_std,capsize=4,label='max')
    ax.errorbar(bin_centers,bin_mean,marker='o',mfc='none',
                yerr=bin_mean_std,capsize=4,label='mean')
    ax.errorbar(bin_centers,bin_min,marker='v',mfc='none',
               yerr=bin_min_std,capsize=4,label='min')
    
    ax.grid(alpha=0.5)
    ax.legend(loc='best')
    
    if x_label!=None: ax.set_xlabel(x_label)
    if y_label!=None: ax.set_ylabel(y_label)
    if title!=None: ax.set_title(title)
    
    fig.tight_layout()
    
    if save_path==None: plt.show()
    else: 
        fig.savefig(save_path)
        plt.close()
    return ax
    
