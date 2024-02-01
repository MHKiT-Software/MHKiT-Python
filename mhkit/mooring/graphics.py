"""
graphics.py

This module provides a function for creating animated visualizations of a 
MoorDyn node position dataset using the matplotlib animation API. 

It includes the main function `animate`, which creates either 2D or 3D 
animations depending on the input parameters. 

In the animations, the position of nodes in the MoorDyn dataset are plotted 
over time, allowing the user to visualize how these positions change. 

This module also includes several helper functions that are used by 
`animate` to validate inputs, generate lists of nodes along each axis, 
calculate plot limits, and set labels and titles for plots. 

The user can specify various parameters for the animation such as the 
dimension (2D or 3D), the axes to plot along, the plot limits for each 
axis, the interval between frames, whether the animation repeats, and the 
labels and title for the plot.

Requires:
- matplotlib
- xarray
"""

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation


def animate(
    dsani,
    dimension="2d",
    xaxis="x",
    yaxis="z",
    zaxis="y",
    xlim=None,
    ylim=None,
    zlim=None,
    interval=10,
    repeat=False,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
):
    """
    Graphics function that creates a 2D or 3D animation of the node positions of a mooring line over time.

    Parameters
    ----------
    dsani : xr.Dataset
        Xarray dataset object containing MoorDyn node variables (ie 'Node0px')
    dimension : str, optional
        Dimension of animation ('2d' or '3d'), by default '2d'
    xaxis : str, optional
        lowercase letter of node axis to plot along x-axis, by default 'x'
    yaxis : str, optional
        lowercase latter of node axis to plot along y-axis, by default 'z'
    zaxis : str, optional
        lowercase latter of node axis to plot along z-axis, by default 'y' (only used in 3d)
    xlim : list, optional
        Two element list for plot: [min x-axis limit, max x-axis limit], by default None
    ylim : list, optional
        Two element list for plot: [min y-axis limit, max y-axis limit], by default None
    zlim : list, optional
        Two element list for plot: [min z-axis limit, max z-axis limit], by default None (only used in 3d)
    interval : int, optional
        Delay between frames in milliseconds, by default 10
    repeat : bool, optional
        Whether the animation repeats when the sequence of frames is completed, by default False
    xlabel : str, optional
        X-label for plot, by default None
    ylabel : str, optional
        Y-label for plot, by default None
    zlabel : str, optional
        Z-label for plot, by default None (only used in 3d)
    title : str, optional
        Set title of plot, by default None

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object

    Raises
    ------
    TypeError
        Checks for correct input types for dsani, dimension, xaxis, yaxis, zaxis, xlim, ylim,
        zlim, interval, repeat, xlabel, ylabel, zlabel, and title
    """
    _validate_input(
        dsani, xlim, ylim, interval, repeat, xlabel, ylabel, title, dimension
    )
    if dimension == "3d":
        if not isinstance(zlim, (list, type(None))):
            raise TypeError("zlim must be of type list")
        if not isinstance(zlabel, (str, type(None))):
            raise TypeError("zlabel must be of type str")
    if not isinstance(xaxis, str):
        raise TypeError("xaxis must be of type str")
    if not isinstance(yaxis, str):
        raise TypeError("yaxis must be of type str")
    if not isinstance(zaxis, str):
        raise TypeError("zaxis must be of type str")

    current_idx = list(dsani.dims.mapping.keys())[0]
    dsani = dsani.rename({current_idx: "time"})

    nodes_x, nodes_y, nodes_z = _get_axis_nodes(dsani, xaxis, yaxis, zaxis)

    if not xlim:
        xlim = _find_limits(dsani[nodes_x])
    if not ylim:
        ylim = _find_limits(dsani[nodes_y])
    if dimension == "3d" and not zlim:
        zlim = _find_limits(dsani[nodes_z])

    fig = plt.figure()
    if dimension == "3d":
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot()
    ax.grid()

    if dimension == "2d":
        (ln,) = ax.plot([], [], "-o")

        def init():
            ax.set(xlim=xlim, ylim=ylim)
            _set_labels(ax, xlabel, ylabel, title)
            return ln

        def update(frame):
            x = dsani[nodes_x].isel(time=frame).to_array().values
            y = dsani[nodes_y].isel(time=frame).to_array().values
            ln.set_data(x, y)

    elif dimension == "3d":
        (ln,) = ax.plot([], [], [], "-o")

        def init():
            ax.set(xlim3d=xlim, ylim3d=ylim, zlim3d=zlim)
            _set_labels(ax, xlabel, ylabel, title, zlabel)
            return ln

        def update(frame):
            x = dsani[nodes_x].isel(time=frame).to_array().values
            y = dsani[nodes_y].isel(time=frame).to_array().values
            z = dsani[nodes_z].isel(time=frame).to_array().values
            ln.set_data(x, y)
            ln.set_3d_properties(z)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(dsani.time),
        init_func=init,
        interval=interval,
        repeat=repeat,
    )

    return ani


def _validate_input(
    dsani, xlim, ylim, interval, repeat, xlabel, ylabel, title, dimension
):
    """
    Validate common input parameters for animate function.
    """
    if not isinstance(dsani, xr.Dataset):
        raise TypeError("dsani must be of type xr.Dataset")
    if not isinstance(xlim, (list, type(None))):
        raise TypeError("xlim must be of type list")
    if not isinstance(ylim, (list, type(None))):
        raise TypeError("ylim must be of type list")
    if not isinstance(interval, int):
        raise TypeError("interval must be of type int")
    if not isinstance(repeat, bool):
        raise TypeError("repeat must be of type bool")
    if not isinstance(xlabel, (str, type(None))):
        raise TypeError("xlabel must be of type str")
    if not isinstance(ylabel, (str, type(None))):
        raise TypeError("ylabel must be of type str")
    if not isinstance(title, (str, type(None))):
        raise TypeError("title must be of type str")
    if dimension not in ["2d", "3d"]:
        raise ValueError('dimension must be either "2d" or "3d"')


def _get_axis_nodes(dsani, xaxis, yaxis, zaxis):
    """
    Helper function to generate the list of nodes along each axis.

    Parameters
    ----------
    dsani : xr.Dataset
        Xarray dataset object containing MoorDyn node variables (ie 'Node0px')
    xaxis : str
        lowercase letter of node axis to plot along x-axis
    yaxis : str
        lowercase latter of node axis to plot along y-axis
    zaxis : str
        lowercase latter of node axis to plot along z-axis

    Returns
    -------
    nodesX : list
        List of nodes along the x-axis
    nodesY : list
        List of nodes along the y-axis
    nodesZ : list
        List of nodes along the z-axis
    """
    nodes = [s for s in list(dsani.data_vars) if "Node" in s]
    nodes_x = [s for s in nodes if f"p{xaxis}" in s]
    nodes_y = [s for s in nodes if f"p{yaxis}" in s]
    nodes_z = [s for s in nodes if f"p{zaxis}" in s]

    return nodes_x, nodes_y, nodes_z


def _find_limits(dataset):
    """Auto calculate the min and max plot limits based on provided dataset

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing data pertaining to specific axis

    Returns
    -------
    list
        Min and max plot limits for axis
    """
    x_1 = dataset.min().to_array().min().values
    x_1 = x_1 - abs(x_1 * 0.1)
    x_2 = dataset.max().to_array().max().values
    x_2 = x_2 + abs(x_2 * 0.1)
    return [x_1, x_2]


def _set_labels(ax, xlabel=None, ylabel=None, title=None, zlabel=None):
    """
    Helper function to set the labels and title for a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to set labels and title on.
    xlabel : str, optional
        X-axis label, by default None
    ylabel : str, optional
        Y-axis label, by default None
    title : str, optional
        Title of the plot, by default None
    zlabel : str, optional
        Z-axis label, by default None for 2D plots
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if zlabel:
        ax.set_zlabel(zlabel)
