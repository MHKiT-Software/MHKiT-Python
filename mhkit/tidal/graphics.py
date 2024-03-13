import numpy as np
import bisect
from scipy.interpolate import interpn as _interpn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mhkit.river.resource import exceedance_probability
from mhkit.tidal.resource import _histogram, _flood_or_ebb
from mhkit.river.graphics import plot_velocity_duration_curve, _xy_plot
from mhkit.utils import convert_to_dataarray


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

    if ax == None:
        # Initialize polar plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(polar=True)
    # Angles are measured clockwise from true north
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    xticks = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Polar plots do not have minor ticks, insert flood/ebb into major ticks
    xtickDegrees = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Set title and metadata box
    if metadata != None:
        # Set the Title
        plt.title(metadata["name"])
        # List of strings for metadata box
        bouy_str = [
            f'Lat = {float(metadata["lat"]):0.2f}$\degree$',
            f'Lon = {float(metadata["lon"]):0.2f}$\degree$',
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
            bbox=dict(facecolor="none", edgecolor="k", pad=5),
        )
    # If defined plot flood and ebb directions as major ticks
    if flood != None:
        # Get flood direction in degrees
        floodDirection = flood
        # Polar plots do not have minor ticks,
        #    insert flood/ebb into major ticks
        bisect.insort(xtickDegrees, floodDirection)
        # Get location in list
        idxFlood = xtickDegrees.index(floodDirection)
        # Insert label at appropriate location
        xticks[idxFlood:idxFlood] = ["\nFlood"]
    if ebb != None:
        # Get flood direction in degrees
        ebbDirection = ebb
        # Polar plots do not have minor ticks,
        #    insert flood/ebb into major ticks
        bisect.insort(xtickDegrees, ebbDirection)
        # Get location in list
        idxEbb = xtickDegrees.index(ebbDirection)
        # Insert label at appropriate location
        xticks[idxEbb:idxEbb] = ["\nEbb"]
    ax.set_xticks(np.array(xtickDegrees) * np.pi / 180.0)
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
    if flood is not None:
        if (flood < 0) and (flood > 360):
            raise ValueError("flood must be between 0 and 360 degrees")
    if ebb is not None:
        if (ebb < 0) and (ebb > 360):
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
    be specified such that 0  degrees is north.

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
    metadata: dictonary
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

    _check_inputs(directions, velocities, flood, ebb)

    if not isinstance(width_dir, (int, float)):
        raise TypeError("width_dir must be of type int or float")
    if not isinstance(width_vel, (int, float)):
        raise TypeError("width_vel must be of type int or float")
    if width_dir < 0:
        raise ValueError("width_dir must be greater than 0")
    if width_vel < 0:
        raise ValueError("width_vel must be greater than 0")

    # Calculate the 2D histogram
    H, dir_edges, vel_edges = _histogram(directions, velocities, width_dir, width_vel)
    # Determine number of bins
    dir_bins = H.shape[0]
    vel_bins = H.shape[1]
    # Create the angles
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / dir_bins)
    # Initialize the polar polt
    ax = _initialize_polar(ax=ax, metadata=metadata, flood=flood, ebb=ebb)
    # Set bar color based on wind speed
    colors = plt.cm.viridis(np.linspace(0, 1.0, vel_bins))
    # Set the current speed bin label names
    # Calculate the 2D histogram
    labels = [f"{i:.1f}-{j:.1f}" for i, j in zip(vel_edges[:-1], vel_edges[1:])]
    # Initialize the vertical-offset (polar radius) for the stacked bar chart.
    r_offset = np.zeros(dir_bins)
    for vel_bin in range(vel_bins):
        # Plot fist set of bars in all directions
        ax.bar(
            thetas,
            H[:, vel_bin],
            width=(2 * np.pi / dir_bins),
            bottom=r_offset,
            color=colors[vel_bin],
            label=labels[vel_bin],
        )
        # Increase the radius offset in all directions
        r_offset = r_offset + H[:, vel_bin]
    # Add the a legend for current speed bins
    plt.legend(
        loc="best", title="Velocity bins [m/s]", bbox_to_anchor=(1.29, 1.00), ncol=1
    )
    # Get the r-ticks (polar y-ticks)
    yticks = plt.yticks()
    # Format y-ticks with  units for clarity
    rticks = [f"{y:.1f}%" for y in yticks[0]]
    # Set the y-ticks
    plt.yticks(yticks[0], rticks)
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
    metadata: dictonary
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

    _check_inputs(directions, velocities, flood, ebb)

    if not isinstance(width_dir, (int, float)):
        raise TypeError("width_dir must be of type int or float")
    if not isinstance(width_vel, (int, float)):
        raise TypeError("width_vel must be of type int or float")
    if width_dir < 0:
        raise ValueError("width_dir must be greater than 0")
    if width_vel < 0:
        raise ValueError("width_vel must be greater than 0")

    # Calculate the 2D histogram
    H, dir_edges, vel_edges = _histogram(directions, velocities, width_dir, width_vel)
    # Initialize the polar polt
    ax = _initialize_polar(ax=ax, metadata=metadata, flood=flood, ebb=ebb)
    # Set the current speed bin label names
    labels = [f"{i:.1f}-{j:.1f}" for i, j in zip(vel_edges[:-1], vel_edges[1:])]
    # Set vel & dir bins to middle of bin except at ends
    dir_bins = 0.5 * (dir_edges[1:] + dir_edges[:-1])  # set all bins to middle
    vel_bins = 0.5 * (vel_edges[1:] + vel_edges[:-1])
    # Reset end of bin range to edge of bin
    dir_bins[0] = dir_edges[0]
    vel_bins[0] = vel_edges[0]
    dir_bins[-1] = dir_edges[-1]
    vel_bins[-1] = vel_edges[-1]
    # Interpolate the bins back to specific data points
    z = _interpn(
        (dir_bins, vel_bins),
        H,
        np.vstack([directions, velocities]).T,
        method="splinef2d",
        bounds_error=False,
    )
    # Plot the most probable data last
    idx = z.argsort()
    # Convert to radians and order points by probability
    theta, r, z = directions.values[idx] * np.pi / 180, velocities.values[idx], z[idx]
    # Create scatter plot colored by probability density
    sx = ax.scatter(theta, r, c=z, s=5, edgecolor=None)
    # Create colorbar
    plt.colorbar(sx, ax=ax, label="Joint Probability [%]")

    # Get the r-ticks (polar y-ticks)
    yticks = ax.get_yticks()
    # Set y-ticks labels
    ax.set_yticks(yticks)  # to avoid matplotlib warning
    ax.set_yticklabels([f"{y:.1f} $m/s$" for y in yticks])

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
    if (principal_direction < 0) and (principal_direction > 360):
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
    speed: array-like
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

    _check_inputs(directions, velocities, flood, ebb)
    if bin_size < 0:
        raise ValueError("bin_size must be greater than 0")

    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 8))

    isEbb = _flood_or_ebb(directions, flood, ebb)

    decimals = round(bin_size / 0.1)
    N_bins = int(round(velocities.max(), decimals) / bin_size)

    H, bins = np.histogram(velocities, bins=N_bins)
    H_ebb, bins1 = np.histogram(velocities[isEbb], bins=bins)
    H_flood, bins2 = np.histogram(velocities[~isEbb], bins=bins)

    p_ebb = H_ebb / H
    p_flood = H_flood / H

    center = (bins[:-1] + bins[1:]) / 2
    width = 0.9 * (bins[1] - bins[0])

    mask1 = np.ma.where(p_ebb >= p_flood)
    mask2 = np.ma.where(p_flood >= p_ebb)

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
        center[mask2],
        height=p_ebb[mask2],
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

    _check_inputs(directions, velocities, flood, ebb)
    if bin_size < 0:
        raise ValueError("bin_size must be greater than 0")

    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 8))

    isEbb = _flood_or_ebb(directions, flood, ebb)

    s_ebb = velocities[isEbb]
    s_flood = velocities[~isEbb]

    F = exceedance_probability(velocities)["F"]
    F_ebb = exceedance_probability(s_ebb)["F"]
    F_flood = exceedance_probability(s_flood)["F"]

    decimals = round(bin_size / 0.1)
    s_new = np.arange(
        np.around(velocities.min(), decimals),
        np.around(velocities.max(), decimals) + bin_size,
        bin_size,
    )

    f_total = interp1d(velocities, F, bounds_error=False)
    f_ebb = interp1d(s_ebb, F_ebb, bounds_error=False)
    f_flood = interp1d(s_flood, F_flood, bounds_error=False)

    F_total = f_total(s_new)
    F_ebb = f_ebb(s_new)
    F_flood = f_flood(s_new)

    F_max_total = np.nanmax(F_ebb) + np.nanmax(F_flood)

    ax.stackplot(
        s_new,
        F_ebb / F_max_total * 100,
        F_flood / F_max_total * 100,
        labels=["Ebb", "Flood"],
    )

    plt.xlabel("velocity [m/s]")
    plt.ylabel("Probability of Exceedance")
    plt.legend()
    plt.grid(linestyle=":", linewidth=1)

    return ax
