import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def _xy_plot(x, y, fmt='.', label=None, xlabel=None, ylabel=None, title=None,
             ax=None):
    """
    Base function to plot any x vs y data

    Parameters
    ----------
    x: array-like
        Data for the x axis of plot
    y: array-like
        Data for y axis of plot
        
    Returns
    -------
    ax : matplotlib.pyplot axes
    
    """
    if ax is None:
        plt.figure(figsize=(16,8))
        params = {'legend.fontsize': 'x-large',
                 'axes.labelsize': 'x-large',
                 'axes.titlesize':'x-large',
                 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'}
        plt.rcParams.update(params)
        ax = plt.gca()
        
    ax.plot(x, y, fmt, label=label, markersize=7)
    
    ax.grid(b=True, which='both')
    
    if label:
        ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    return ax

def plot_flow_duration_curve(D, F, label=None, ax=None):
    """
    Plots discharge vs exceedance probability as a Flow Duration Curve (FDC) 
    
    Parameters
    ------------
    D: array-like
        Discharge [m/s] indexed by time
        
    F: array-like 
         Exceedance probability [unitless] indexed by time
         
    label: string
       Label to use in the legend
        
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes
            
    """
    # Sort by F
    temp = pd.DataFrame({'D': D, 'F': F})
    temp.sort_values('F', ascending=False, kind='mergesort', inplace=True)   
    
    ax = _xy_plot(temp['D'], temp['F'], fmt='-', label=label, xlabel='Discharge [$m^3/s$]',
             ylabel='Exceedance Probability', ax=ax)
    plt.xscale('log')

    return ax

def plot_velocity_duration_curve(V, F, label=None, ax=None):
    """
    Plots velocity vs exceedance probability as a Velocity Duration Curve (VDC) 
    
    Parameters
    ------------
    V: array-like 
        Velocity [m/s] indexed by time
        
    F: array-like 
        Exceedance probability [unitless] indexed by time
        
    label: string
       Label to use in the legend
       
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes
            
    """
    # Sort by F
    temp = pd.DataFrame({'V': V, 'F': F})
    temp.sort_values('F', ascending=False, kind='mergesort', inplace=True)  
    
    ax = _xy_plot(temp['V'], temp['F'], fmt='-', label=label, xlabel='Velocity [$m/s$]', 
             ylabel='Exceedance Probability', ax=ax)

    return ax

def plot_power_duration_curve(P, F, label=None, ax=None):
    """
    Plots power vs exceedance probability as a Power Duration Curve (PDC) 

    Parameters
    ------------
    P: array-like 
        Power [W] indexed by time
        
    F: array-like 
        Exceedance probability [unitless] indexed by time
        
    label: string
       Label to use in the legend
       
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes
            
    """
    # Sort by F
    temp = pd.DataFrame({'P': P, 'F': F})
    temp.sort_values('F', ascending=False, kind='mergesort', inplace=True)
    
    ax = _xy_plot(temp['P'], temp['F'], fmt='-', label=label, xlabel='Power [W]', 
             ylabel='Exceedance Probability', ax=ax)

    return ax
    
def plot_discharge_timeseries(Q, label=None, ax=None):
    """
    Plots discharge time-series
    
    Parameters
    ------------
    Q: array-like
        Discharge [m3/s] indexed by time
    
    label: string
       Label to use in the legend
       
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes     
    
    """
    ax = _xy_plot(Q.index, Q, fmt='-', label=label, xlabel='Time', 
             ylabel='Discharge [$m^3/s$]', ax=ax)
    
    return ax

def plot_discharge_vs_velocity(D, V, polynomial_coeff=None, label=None, ax=None):
    """
    Plots discharge vs velocity data along with the polynomial fit
    
    Parameters
    ------------
    D : pandas Series
        Discharge [m/s] indexed by time
        
    V : pandas Series
        Velocity [m/s] indexed by time
        
    polynomial_coeff: numpy polynomial
        Polynomial coefficients, which can be computed using 
        `river.resource.polynomial_fit`.  If None, then the polynomial fit is 
        not included int the plot. 
        
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes
            
    """
    ax = _xy_plot(D, V, fmt='.', label=label, xlabel='Discharge [$m^3/s$]', 
                  ylabel='Velocity [$m/s$]', ax=ax)
    if polynomial_coeff:
        x = np.linspace(D.min(), D.max())
        ax = _xy_plot(x, polynomial_coeff(x), fmt='--', label='Polynomial fit', 
                      xlabel='Discharge [$m^3/s$]', ylabel='Velocity [$m/s$]',
                      ax=ax)

    return ax


def plot_velocity_vs_power(V, P, polynomial_coeff=None, label=None, ax=None):
    """
    Plots velocity vs power data along with the polynomial fit 
    
    Parameters
    ------------
    V : pandas Series
        Velocity [m/s] indexed by time
        
    P: pandas Series
        Power [W] indexed by time
        
    polynomial_coeff: numpy polynomial
        Polynomial coefficients, which can be computed using 
        `river.resource.polynomial_fit`.  If None, then the polynomial fit is 
        not included int the plot. 
        
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single 
        axes is used.
        
    Returns
    ---------
    ax : matplotlib pyplot axes
            
    """
    ax = _xy_plot(V, P, fmt='.', label=label, xlabel='Velocity [$m/s$]', 
                  ylabel='Power [$W$]', ax=ax)
    if polynomial_coeff:
        x = np.linspace(V.min(), V.max())
        ax = _xy_plot(x, polynomial_coeff(x), fmt='--', label='Polynomial fit', 
             xlabel='Velocity [$m/s$]', ylabel='Power [$W$]', ax=ax)
    
    return ax
