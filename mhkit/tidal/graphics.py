import numpy as np
import pandas as pd
import math
import bisect
from scipy.interpolate import interpn as _interpn
import matplotlib.pyplot as plt 
from mhkit.tidal.resource import _histogram
from mhkit.river.graphics import plot_velocity_duration_curve, _xy_plot


def _initialize_polar(ax = None, metadata=None, flood=None, ebb=None):
    '''
    Initializes a polar plots with cardinal directions and ebb/flow
    
    Parameters
    ----------
    ax :axes
    metadata: dictionary
        Contains site meta data
    Returns
    -------
    ax: axes
    '''
    if ax ==None:
        # Initialize polar plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(polar=True)
    # Angles are measured clockwise from true north
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    xticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    # Polar plots do not have minor ticks, insert flood/ebb into major ticks
    xtickDegrees = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Set title and metadata box
    if metadata != None:
        # Set the Title
        plt.title(metadata['name'])
        # List of strings for metadata box
        bouy_str = [f'Lat = {float(metadata["lat"]):0.2f}$\degree$', 
                    f'Lon = {float(metadata["lon"]):0.2f}$\degree$']
        # Create string for text box
        bouy_data = '\n'.join(bouy_str)
        # Set the text box
        ax.text(-0.3, 0.80, bouy_data, transform=ax.transAxes, fontsize=14,
                verticalalignment='top',bbox=dict(facecolor='none', 
                edgecolor='k', pad=5) )
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
        xticks[idxFlood:idxFlood]=['\nFlood']
    if ebb != None:
        # Get flood direction in degrees
        ebbDirection =ebb
        # Polar plots do not have minor ticks, 
        #    insert flood/ebb into major ticks
        bisect.insort(xtickDegrees, ebbDirection)
        # Get location in list
        idxEbb = xtickDegrees.index(ebbDirection) 
        # Insert label at appropriate location
        xticks[idxEbb:idxEbb]=['\nEbb']
    ax.set_xticks(np.array(xtickDegrees)*np.pi/180.)  
    ax.set_xticklabels(xticks)
    return ax


def plot_rose(directions, velocities, width_dir, width_vel, metadata=None,
              flood=None, ebb=None):
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
    # Calculate the 2D histogram
    H, dir_edges, vel_edges = _histogram(directions, velocities, width_dir, width_vel)
    # Determine number of bins
    dir_bins = H.shape[0]
    vel_bins = H.shape[1]
    # Create the angles 
    thetas = np.arange(0,2*np.pi, 2*np.pi/dir_bins)
    # Initialize the polar polt
    ax = _initialize_polar(metadata=metadata, flood=flood, ebb=ebb)
    # Set bar color based on wind speed
    colors = plt.cm.viridis(np.linspace(0, 1.0, vel_bins))
    # Set the current speed bin label names
    # Calculate the 2D histogram
    labels = [ f'{i:.1f}-{j:.1f}' for i,j in zip(vel_edges[:-1],vel_edges[1:])]
    # Initialize the vertical-offset (polar radius) for the stacked bar chart.
    r_offset = np.zeros(dir_bins)
    for vel_bin in range(vel_bins):
        # Plot fist set of bars in all directions
        ax = plt.bar(thetas, H[:,vel_bin], width=(2*np.pi/dir_bins), 
                     bottom=r_offset, color=colors[vel_bin], label=labels[vel_bin])
        # Increase the radius offset in all directions
        r_offset = r_offset + H[:,vel_bin]
    # Add the a legend for current speed bins 
    plt.legend(loc='best',title='Velocity bins [m/s]', bbox_to_anchor=(1.29, 1.00), ncol=1)
    # Get the r-ticks (polar y-ticks)
    yticks = plt.yticks()
    # Format y-ticks with  units for clarity 
    rticks =  [f'{y:.1f}%' for y in yticks[0]]
    # Set the y-ticks
    plt.yticks(yticks[0],rticks)
    return ax


def plot_joint_probability_distribution(directions, velocities, width_dir, 
                                        width_vel, metadata=None,
                                        flood=None, ebb=None):
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
    # Calculate the 2D histogram
    H, dir_edges, vel_edges = _histogram(directions, velocities, width_dir, width_vel)
    # Initialize the polar polt
    ax = _initialize_polar(metadata=metadata, flood=flood, ebb=ebb)
    # Set the current speed bin label names
    labels = [ f'{i:.1f}-{j:.1f}' for i,j in zip(vel_edges[:-1],vel_edges[1:])]
    # Set vel & dir bins to middle of bin except at ends
    dir_bins = 0.5*(dir_edges[1:] + dir_edges[:-1]) # set all bins to middle
    vel_bins = 0.5*(vel_edges[1:] + vel_edges[:-1])
    # Reset end of bin range to edge of bin
    dir_bins[0] = dir_edges[0]
    vel_bins[0] = vel_edges[0]
    dir_bins[-1] = dir_edges[-1]
    vel_bins[-1] = vel_edges[-1]
    # Interpolate the bins back to specific data points
    z = _interpn( (dir_bins, vel_bins  ) ,
                  H , np.vstack([directions,velocities]).T , method = "splinef2d",
                  bounds_error = False )
    # Plot the most probable data last 
    idx=z.argsort()
    # Convert to radians and order points by probability
    theta,r,z = directions.values[idx]*np.pi/180. , velocities.values[idx], z[idx]
    # Create scatter plot colored by probability density    
    sx=ax.scatter(theta, r, c=z, s=5, edgecolor=None)
    # Create colorbar
    plt.colorbar(sx, label='Joint Probability [%]')
    # Get the r-ticks (polar y-ticks)
    yticks = plt.yticks()
    # Format y-ticks with  units for clarity 
    yticks =  [f'{y:.1f} $m/s$' for y in yticks[0]]
    # Set the y-ticks
    ax.set_yticklabels(yticks)
    return ax


def plot_current_timeseries(directions, speeds, principal_direction,
                            label=None, ax=None):
    '''
    Returns a plot of velocity from an array of direction and speed
    data in the direction of the supplied principal_direction.

    Parameters
    ----------
    direction: array-like
        Time-series of directions [degrees]
    speed: array-like
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
    '''
    # Rotate coordinate system by supplied principal_direction
    principal_directions = directions - principal_direction
    # Calculate the velocity
    velocities = speeds * np.cos(np.pi/180*principal_directions)
    # Call on standard xy plotting
    ax = _xy_plot(velocities.index, velocities, fmt='-', label=label, xlabel='Time',
                     ylabel='Velocity [$m/s$]', ax=ax)
    return ax
