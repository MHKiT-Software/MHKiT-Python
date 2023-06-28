import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation


def animate_3d(dsani,xlim=None,ylim=None,zlim=None,interval=10,repeat=False,
              xlabel=None,ylabel=None,zlabel=None,title=None):
    """
    Graphics function that animates the x,y,z node positions of a mooring line over time in 3D.

    Parameters
    ----------
    dsani : xr.Dataset
        Xarray dataset object containing MoorDyn node variables (ie 'Node0px')
    xlim : list, optional
        Two element list for plot: [min x-axis limit, max x-axis limit], by default None
    ylim : list, optional
        Two element list for plot: [min y-axis limit, max y-axis limit], by default None
    zlim : list, optional
        Two element list for plot: [min z-axis limit, max z-axis limit], by default None
    interval : int, optional
        Delay between frames in milliseconds, by default 10
    repeat : bool, optional
        Whether the animation repeats when the sequence of frames is completed, by default False
    xlabel : str, optional
        X-label for plot, by default None
    ylabel : str, optional
        Y-label for plot, by default None
    zlabel : str, optional
        Z-axis label for plot, by default None
    title : str, optional
        Set title of plot, by default None

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object
    
    Raises
    ------
    TypeError
        Checks for correct input types for dsani, xlim, ylim, zlim, interval, repeat, xlabel, 
        ylabel, zlabel, and title
    """
    if not isinstance(dsani, xr.Dataset): raise TypeError('dsani must be of type xr.Dataset')
    if not isinstance(xlim, (list,type(None))): raise TypeError('xlim must of be of type list')
    if not isinstance(ylim, (list,type(None))): raise TypeError('ylim must of be of type list')
    if not isinstance(zlim, (list,type(None))): raise TypeError('zlim must of be of type list')
    if not isinstance(interval, int): raise TypeError('interval must of be of type int')
    if not isinstance(repeat, bool): raise TypeError('repeat must of be of type bool')
    if not isinstance(xlabel, (str,type(None))): raise TypeError('xlabel must of be of type str')
    if not isinstance(ylabel, (str,type(None))): raise TypeError('ylabel must of be of type str')
    if not isinstance(zlabel, (str,type(None))): raise TypeError('zlabel must of be of type str')
    if not isinstance(title, (str,type(None))): raise TypeError('title must of be of type list')
    
    current_idx = list(dsani.dims.mapping.keys())[0]
    dsani = dsani.rename({current_idx: 'time'})

    chans = list(dsani.keys())
    nodesX = [x for x in chans if 'x' in x]
    nodesY = [x for x in chans if 'y' in x]
    nodesZ = [x for x in chans if 'z' in x]
    
    if not xlim: xlim=_find_limits(dsani[nodesX])
    if not ylim: ylim=_find_limits(dsani[nodesY])
    if not zlim: zlim=_find_limits(dsani[nodesZ])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ln, = ax.plot([], [], [], '-o')

    def init():
        ax.set(xlim3d=xlim,ylim3d=ylim,zlim3d=zlim)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if zlabel: ax.set_zlabel(zlabel)
        if title: ax.set_title(title)
        return ln

    def update(frame):
        x = dsani[nodesX].isel(time=frame).to_array().values
        y = dsani[nodesY].isel(time=frame).to_array().values
        z = dsani[nodesZ].isel(time=frame).to_array().values
        ln.set_data(x,y)
        ln.set_3d_properties(z)

    ani = FuncAnimation(fig, update, frames=len(dsani.time),
                        init_func=init, interval=interval,repeat=repeat)
    
    return ani


def animate_2d(dsani,xaxis='x',yaxis='z',xlim=None,ylim=None,interval=10,repeat=False,
              xlabel=None,ylabel=None,title=None):
    """
    Graphics function that creates a 2D animation of the node positions of a mooring line over time.

    Parameters
    ----------
    dsani : xr.Dataset
        Xarray dataset object containing MoorDyn node variables (ie 'Node0px')
    xaxis : str, optional
        lowercase letter of node axis to plot along x-axis, by default 'x'
    yaxis : str, optional
        lowercase latter of node axis to plot along y-axis, by default 'z'
    xlim : list, optional
        Two element list for plot: [min x-axis limit, max x-axis limit], by default None
    ylim : list, optional
        Two element list for plot: [min y-axis limit, max y-axis limit], by default None
    interval : int, optional
        Delay between frames in milliseconds, by default 10
    repeat : bool, optional
        Whether the animation repeats when the sequence of frames is completed, by default False
    xlabel : str, optional
        X-label for plot, by default None
    ylabel : str, optional
        Y-label for plot, by default None
    title : str, optional
        Set title of plot, by default None

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object
    
    Raises
    ------
    TypeError
        Checks for correct input types for dsani, xaxis, yaxis, xlim, ylim, interval, repeat, 
        xlabel, ylabel, and title
    """
    if not isinstance(dsani, xr.Dataset): raise TypeError('dsani must be of type xr.Dataset')
    if not isinstance(xaxis, str): raise TypeError('xaxis must of be of type str')
    if not isinstance(yaxis, str): raise TypeError('yaxis must of be of type str')
    if not isinstance(xlim, (list,type(None))): raise TypeError('xlim must of be of type list')
    if not isinstance(ylim, (list,type(None))): raise TypeError('ylim must of be of type list')
    if not isinstance(interval, int): raise TypeError('interval must of be of type int')
    if not isinstance(repeat, bool): raise TypeError('repeat must of be of type bool')
    if not isinstance(xlabel, (str,type(None))): raise TypeError('xlabel must of be of type str')
    if not isinstance(ylabel, (str,type(None))): raise TypeError('ylabel must of be of type str')
    if not isinstance(title, (str,type(None))): raise TypeError('title must of be of type list')

    current_idx = list(dsani.dims.mapping.keys())[0]
    dsani = dsani.rename({current_idx: 'time'})

    chans = list(dsani.keys())
    nodesX = [x for x in chans if xaxis in x]
    nodesY = [x for x in chans if yaxis in x]
    
    if not xlim: xlim=_find_limits(dsani[nodesX])
    if not ylim: ylim=_find_limits(dsani[nodesY])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()

    ln, = ax.plot([], [], '-o')

    def init():
        ax.set(xlim=xlim,ylim=ylim)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if title: ax.set_title(title)
        return ln

    def update(frame):
        x = dsani[nodesX].isel(time=frame).to_array().values
        y = dsani[nodesY].isel(time=frame).to_array().values
        ln.set_data(x,y)

    ani = FuncAnimation(fig, update, frames=len(dsani.time),
                        init_func=init, interval=interval,repeat=repeat)
    
    return ani


def _find_limits(ds):
    """Auto calculate the min and max plot limits based on provided dataset

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing data pertaining to specific axis

    Returns
    -------
    list
        Min and max plot limits for axis
    """
    x1 = ds.min().to_array().min().values
    x1 = x1 - abs(x1*0.1)
    x2 = ds.max().to_array().max().values
    x2 = x2 + abs(x2*0.1)
    return [x1, x2]