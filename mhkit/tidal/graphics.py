"""
MHKiT Tidal Graphics Module

This module provides functions for visualizing tidal data.
It includes tools for creating plots and graphs to analyze tidal
resource and performance data.
"""

import bisect
import numpy as np
from scipy.interpolate import interpn as _interpn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from mhkit.river.resource import exceedance_probability
from mhkit.tidal.resource import _histogram, _flood_or_ebb
from mhkit.river.graphics import plot_velocity_duration_curve, _xy_plot
from mhkit.utils import convert_to_dataarray

# Explicitly declare the river functions to be exported
__all__ = [
    "plot_velocity_duration_curve",
]

viridis = mpl.colormaps["viridis"]


def _initialize_polar(ax=None, metadata=None, flood=None, ebb=None):
    """
    Initializes a polar plots with cardinal directions and ebb/flow

    Parameters
    ----------
    ax :axes
    metadata: dictionary
        Contains site meta data
    Returns
    -------
    ax: axes
    """

    if ax is None:
        # Initialize polar plot
        plt.figure(figsize=(12, 8))
        ax = plt.axes(polar=True)
    # Angles are measured clockwise from true north
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    xticks = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Polar plots do not have minor ticks, insert flood/ebb into major ticks
    xtick_degrees = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Set title and metadata box
    if metadata is not None:
        # Set the Title
        plt.title(metadata["name"])
        # List of strings for metadata box
        bouy_str = [
            f'Lat = {float(metadata["lat"]):0.2f}$\\degree$',
            f'Lon = {float(metadata["lon"]):0.2f}$\\degree$',
        ]
        # Create string for text box
        bouy_data = "\n".join(bouy_str)
        # Set the text box
        ax.text(
            -0.3,
            0.80,
            bouy_data,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox={"facecolor": "none", "edgecolor": "k", "pad": 5},
        )
    # If defined plot flood and ebb directions as major ticks
    if flood is not None:
        # Get flood direction in degrees
        flood_direction = flood
        # Polar plots do not have minor ticks,
        #    insert flood/ebb into major ticks
        bisect.insort(xtick_degrees, flood_direction)
        # Get location in list
        idx_flood = xtick_degrees.index(flood_direction)
        # Insert label at appropriate location
        xticks[idx_flood:idx_flood] = ["\nFlood"]
    if ebb is not None:
        # Get ebb direction in degrees
        ebb_direction = ebb
        # Polar plots do not have minor ticks,
        #    insert flood/ebb into major ticks
        bisect.insort(xtick_degrees, ebb_direction)
        # Get location in list
        idx_ebb = xtick_degrees.index(ebb_direction)
        # Insert label at appropriate location
        xticks[idx_ebb:idx_ebb] = ["\nEbb"]
    ax.set_xticks(np.array(xtick_degrees) * np.pi / 180.0)
    ax.set_xticklabels(xticks)
    return ax


def _check_inputs(directions, velocities, flood, ebb):
    """
    Runs checks on inputs for the graphics functions.

    Parameters
    ----------
    directions: array-like
        Directions in degrees with 0 degrees specified as true north
    velocities: array-like
        Velocities in m/s
    flood: float
        Direction in degrees added to theta ticks
    ebb: float
        Direction in degrees added to theta ticks
    """

    velocities = convert_to_dataarray(velocities)
    directions = convert_to_dataarray(directions)

    if len(velocities) != len(directions):
        raise ValueError("velocities and directions must have the same length")
    if all(np.nan_to_num(velocities.values) < 0):
        raise ValueError("All velocities must be positive")
    if all(np.nan_to_num(directions.values) < 0) and all(
        np.nan_to_num(directions.values) > 360
    ):
        raise ValueError("directions must be between 0 and 360 degrees")
    if not isinstance(flood, (int, float, type(None))):
        raise TypeError("flood must be of type int or float")
    if not isinstance(ebb, (int, float, type(None))):
        raise TypeError("ebb must be of type int or float")
    if flood is not None and not 0 <= flood <= 360:
        raise ValueError("flood must be between 0 and 360 degrees")
    if ebb is not None and not 0 <= ebb <= 360:
        raise ValueError("ebb must be between 0 and 360 degrees")


def plot_rose(
    directions,
    velocities,
    width_dir,
    width_vel,
    ax=None,
    metadata=None,
    flood=None,
    ebb=None,
):
    """
    Creates a polar histogram. Direction angles from binned histogram must
    be specified such that 0 degrees is north.

    Parameters
    ----------
    directions: array-like
        Directions in degrees with 0 degrees specified as true north
    velocities: array-like
        Velocities in m/s
    width_dir: float
        Width of directional bins for histogram in degrees
    width_vel: float
        Width of velocity bins for histogram in m/s
    ax: float
        Polar plot axes to add polar histogram
    metadata: dictionary
        If provided needs keys ['name', 'lat', 'lon'] for plot title
        and information box on plot
    flood: float
        Direction in degrees added to theta ticks
    ebb: float
        Direction in degrees added to theta ticks
    Returns
    -------
    ax: figure
        Water current rose plot
    """
    # pylint: disable=too-many-positional-arguments, disable=too-many-arguments, disable=too-many-locals
    # Validate inputs inline to reduce function calls
    _check_inputs(directions, velocities, flood, ebb)
    if not isinstance(width_dir, (int, float)) or width_dir < 0:
        raise ValueError("width_dir must be a positive number")
    if not isinstance(width_vel, (int, float)) or width_vel < 0:
        raise ValueError("width_vel must be a positive number")

    # Compute histogram and bin edges
    histogram, _, vel_edges = _histogram(directions, velocities, width_dir, width_vel)

    # Initialize polar plot
    ax = _initialize_polar(ax=ax, metadata=metadata, flood=flood, ebb=ebb)

    # Define bin properties
    dir_bins, vel_bins = histogram.shape
    thetas = np.linspace(0, 2 * np.pi, dir_bins, endpoint=False)
    colors = viridis(np.linspace(0, 1, vel_bins))
    labels = [f"{i:.1f}-{j:.1f}" for i, j in zip(vel_edges[:-1], vel_edges[1:])]

    # Plot histogram
    r_offset = np.zeros(dir_bins)
    for vel_bin in range(vel_bins):
        ax.bar(
            thetas,
            histogram[:, vel_bin],
            width=(2 * np.pi / dir_bins),
            bottom=r_offset,
            color=colors[vel_bin],
            label=labels[vel_bin],
        )
        r_offset += histogram[:, vel_bin]  # Update in place

    # Configure legend and ticks
    plt.legend(
        loc="best", title="Velocity bins [m/s]", bbox_to_anchor=(1.29, 1.00), ncol=1
    )
    yticks = plt.yticks()
    plt.yticks(yticks[0], [f"{y:.1f}%" for y in yticks[0]])

    return ax


def plot_joint_probability_distribution(
    directions,
    velocities,
    width_dir,
    width_vel,
    ax=None,
    metadata=None,
    flood=None,
    ebb=None,
):
    """
    Creates a polar histogram. Direction angles from binned histogram must
    be specified such that 0 is north.

    Parameters
    ----------
    directions: array-like
        Directions in degrees with 0 degrees specified as true north
    velocities: array-like
        Velocities in m/s
    width_dir: float
        Width of directional bins for histogram in degrees
    width_vel: float
        Width of velocity bins for histogram in m/s
    ax: float
        Polar plot axes to add polar histogram
    metadata: dictionary
        If provided needs keys ['name', 'Lat', 'Lon'] for plot title
        and information box on plot
    flood: float
        Direction in degrees added to theta ticks
    ebb: float
        Direction in degrees added to theta ticks
    Returns
    -------
    ax: figure
        Joint probability distribution
    """
    # pylint: disable=too-many-positional-arguments, disable=too-many-arguments, disable=too-many-locals
    _check_inputs(directions, velocities, flood, ebb)

    if not isinstance(width_dir, (int, float)):
        raise TypeError("width_dir must be of type int or float")
    if not isinstance(width_vel, (int, float)):
        raise TypeError("width_vel must be of type int or float")
    if width_dir < 0 or width_vel < 0:
        raise ValueError("width_dir and width_vel must be greater than 0")

    histogram, dir_edges, vel_edges = _histogram(
        directions, velocities, width_dir, width_vel
    )
    ax = _initialize_polar(ax=ax, metadata=metadata, flood=flood, ebb=ebb)

    dir_bins = 0.5 * (dir_edges[1:] + dir_edges[:-1])
    vel_bins = 0.5 * (vel_edges[1:] + vel_edges[:-1])
    dir_bins[[0, -1]] = dir_edges[[0, -1]]
    vel_bins[[0, -1]] = vel_edges[[0, -1]]

    z = _interpn(
        (dir_bins, vel_bins),
        histogram,
        np.vstack([directions, velocities]).T,
        method="splinef2d",
        bounds_error=False,
    )

    idx = z.argsort()
    theta = directions.values[idx] * np.pi / 180
    r = velocities.values[idx]

    sx = ax.scatter(theta, r, c=z[idx], s=5, edgecolor=None)
    plt.colorbar(sx, ax=ax, label="Joint Probability [%]")

    ax.set_yticklabels([f"{y:.1f} $m/s$" for y in ax.get_yticks()])

    return ax


def plot_current_timeseries(
    directions, velocities, principal_direction, label=None, ax=None
):
    """
    Returns a plot of velocity from an array of direction and speed
    data in the direction of the supplied principal_direction.

    Parameters
    ----------
    directions: array-like
        Time-series of directions [degrees]
    velocities: array-like
        Time-series of speeds [m/s]
    principal_direction: float
        Direction to compute the velocity in [degrees]
    label: string
        Label to use in the legend
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single
        axes is used.

    Returns
    -------
    ax: figure
        Time-series plot of current-speed velocity
    """

    _check_inputs(directions, velocities, flood=None, ebb=None)

    if not isinstance(principal_direction, (int, float)):
        raise TypeError("principal_direction must be of type int or float")
    if (principal_direction < 0) or (principal_direction > 360):
        raise ValueError("principal_direction must be between 0 and 360 degrees")

    # Rotate coordinate system by supplied principal_direction
    principal_directions = directions - principal_direction
    # Calculate the velocity
    velocity = velocities * np.cos(np.pi / 180 * principal_directions)
    # Call on standard xy plotting
    ax = _xy_plot(
        velocities.index,
        velocity,
        fmt="-",
        label=label,
        xlabel="Time",
        ylabel="Velocity [$m/s$]",
        ax=ax,
    )
    return ax


def tidal_phase_probability(directions, velocities, flood, ebb, bin_size=0.1, ax=None):
    """
    Discretizes the tidal series speed by bin size and returns a plot
    of the probability for each bin in the flood or ebb tidal phase.

    Parameters
    ----------
    directions: array-like
        Time-series of directions [degrees]
    velocities: array-like
        Time-series of speeds [m/s]
    flood: float or int
        Principal component of flow in the flood direction [degrees]
    ebb: float or int
        Principal component of flow in the ebb direction [degrees]
    bin_size: float
        Speed bin size. Optional. Default = 0.1 m/s
    ax : matplotlib axes object
        Axes for plotting. If None, then a new figure with a single
        axes is used.

    Returns
    -------
    ax: figure
    """
    # pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
    _check_inputs(directions, velocities, flood, ebb)
    if bin_size < 0:
        raise ValueError("bin_size must be greater than 0")

    if ax is None:
        ax = plt.subplots(figsize=(12, 8))[1]

    is_ebb = _flood_or_ebb(directions, flood, ebb)

    n_bins = int(round(velocities.max(), round(bin_size / 0.1)) / bin_size)

    bins = np.histogram_bin_edges(velocities, bins=n_bins)
    h_ebb, _ = np.histogram(velocities[is_ebb], bins=bins)
    h_flood, _ = np.histogram(velocities[~is_ebb], bins=bins)

    p_ebb = h_ebb / (h_ebb + h_flood)
    p_flood = h_flood / (h_ebb + h_flood)

    center = (bins[:-1] + bins[1:]) / 2
    width = 0.9 * (bins[1] - bins[0])

    mask1 = p_ebb >= p_flood

    ax.bar(
        center[mask1],
        height=p_ebb[mask1],
        edgecolor="black",
        width=width,
        label="Ebb",
        color="blue",
    )
    ax.bar(
        center,
        height=p_flood,
        edgecolor="black",
        width=width,
        alpha=1,
        label="Flood",
        color="orange",
    )
    ax.bar(
        center[~mask1],
        height=p_ebb[~mask1],
        alpha=1,
        edgecolor="black",
        width=width,
        color="blue",
    )

    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(linestyle=":")

    return ax


def tidal_phase_exceedance(directions, velocities, flood, ebb, bin_size=0.1, ax=None):
    """
    Returns a stacked area plot of the exceedance probability for the
    flood and ebb tidal phases.

    Parameters
    ----------
    directions: array-like
        Time-series of directions [degrees]
    velocities: array-like
        Time-series of speeds [m/s]
    flood: float or int
        Principal component of flow in the flood direction [degrees]
    ebb: float or int
        Principal component of flow in the ebb direction [degrees]
    bin_size: float
        Speed bin size. Optional. Deaful = 0.1 m/s
    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single
        axes is used.

    Returns
    -------
    ax: figure
    """
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    _check_inputs(directions, velocities, flood, ebb)
    if bin_size < 0:
        raise ValueError("bin_size must be greater than 0")

    if ax is None:
        ax = plt.subplots(figsize=(12, 8))[1]

    is_ebb = _flood_or_ebb(directions, flood, ebb)

    s_ebb = velocities[is_ebb]
    s_flood = velocities[~is_ebb]

    f_ebb = exceedance_probability(s_ebb)["F"]
    f_flood = exceedance_probability(s_flood)["F"]

    decimals = round(bin_size / 0.1)
    s_new = np.arange(
        np.around(velocities.min(), decimals),
        np.around(velocities.max(), decimals) + bin_size,
        bin_size,
    )

    f_ebb = interp1d(s_ebb, f_ebb, bounds_error=False)
    f_flood = interp1d(s_flood, f_flood, bounds_error=False)

    f_max_total = np.nanmax(f_ebb(s_new)) + np.nanmax(f_flood(s_new))

    ax.stackplot(
        s_new,
        f_ebb(s_new) / f_max_total * 100,
        f_flood(s_new) / f_max_total * 100,
        labels=["Ebb", "Flood"],
    )

    plt.xlabel("velocity [m/s]")
    plt.ylabel("Probability of Exceedance")
    plt.legend()
    plt.grid(linestyle=":", linewidth=1)

    return ax
