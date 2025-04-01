"""
The graphics module provides plotting utilities for river and tidal energy resource data.

This module contains functions for creating standardized visualizations of:
- Flow Duration Curves (FDC)
- Velocity Duration Curves (VDC)
- Power Duration Curves (PDC)
- Discharge time series
- Discharge vs velocity relationships
- Velocity vs power relationships

Each plotting function accepts data in array-like format and returns matplotlib axes
objects for further customization if needed. All plots follow consistent styling and
include proper units and labels by default.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mhkit.utils import convert_to_dataarray


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def _xy_plot(x, y, fmt=".", label=None, xlabel=None, ylabel=None, title=None, ax=None):
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
        plt.figure(figsize=(16, 8))
        params = {
            "legend.fontsize": "x-large",
            "axes.labelsize": "x-large",
            "axes.titlesize": "x-large",
            "xtick.labelsize": "x-large",
            "ytick.labelsize": "x-large",
        }
        plt.rcParams.update(params)
        ax = plt.gca()

    ax.plot(x, y, fmt, label=label, markersize=7)

    ax.grid()

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


def plot_flow_duration_curve(discharge, exceedance_prob, label=None, ax=None):
    """
    Plots discharge vs exceedance probability as a Flow Duration Curve (FDC)

    Parameters
    ------------
    discharge: array-like
        Discharge [m/s] indexed by time

    exceedance_prob: array-like
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
    # Sort by exceedance_prob
    temp = xr.Dataset(
        data_vars={"discharge": discharge, "exceedance_prob": exceedance_prob}
    )
    temp = temp.sortby("exceedance_prob", ascending=False)

    ax = _xy_plot(
        temp["discharge"],
        temp["exceedance_prob"],
        fmt="-",
        label=label,
        xlabel="Discharge [$m^3/s$]",
        ylabel="Exceedance Probability",
        ax=ax,
    )
    plt.xscale("log")

    return ax


def plot_velocity_duration_curve(velocity, exceedance_prob, label=None, ax=None):
    """
    Plots velocity vs exceedance probability as a Velocity Duration Curve (VDC)

    Parameters
    ------------
    velocity: array-like
        Velocity [m/s] indexed by time

    exceedance_prob: array-like
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
    # Sort by exceedance_prob
    temp = xr.Dataset(
        data_vars={"velocity": velocity, "exceedance_prob": exceedance_prob}
    )
    temp = temp.sortby("exceedance_prob", ascending=False)

    ax = _xy_plot(
        temp["velocity"],
        temp["exceedance_prob"],
        fmt="-",
        label=label,
        xlabel="Velocity [$m/s$]",
        ylabel="Exceedance Probability",
        ax=ax,
    )

    return ax


def plot_power_duration_curve(power, exceedance_prob, label=None, ax=None):
    """
    Plots power vs exceedance probability as a Power Duration Curve (PDC)

    Parameters
    ------------
    power: array-like
        Power [W] indexed by time

    exceedance_prob: array-like
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
    # Sort by exceedance_prob
    temp = xr.Dataset(data_vars={"power": power, "exceedance_prob": exceedance_prob})
    temp.sortby("exceedance_prob", ascending=False)

    ax = _xy_plot(
        temp["power"],
        temp["exceedance_prob"],
        fmt="-",
        label=label,
        xlabel="Power [W]",
        ylabel="Exceedance Probability",
        ax=ax,
    )

    return ax


def plot_discharge_timeseries(discharge, time_dimension="", label=None, ax=None):
    """
    Plots discharge time-series

    Parameters
    ------------
    discharge: array-like
        Discharge [m3/s] indexed by time

    time_dimension: string (optional)
        Name of the xarray dimension corresponding to time. If not supplied,
        defaults to the first dimension.

    label: string
       Label to use in the legend

    ax : matplotlib axes object
        Axes for plotting.  If None, then a new figure with a single
        axes is used.

    Returns
    ---------
    ax : matplotlib pyplot axes

    """
    discharge = convert_to_dataarray(discharge)

    if time_dimension == "":
        time_dimension = list(discharge.coords)[0]

    ax = _xy_plot(
        discharge.coords[time_dimension].values,
        discharge,
        fmt="-",
        label=label,
        xlabel="Time",
        ylabel="Discharge [$m^3/s$]",
        ax=ax,
    )

    return ax


def plot_discharge_vs_velocity(
    discharge, velocity, polynomial_coeff=None, label=None, ax=None
):
    """
    Plots discharge vs velocity data along with the polynomial fit

    Parameters
    ------------
    discharge : array-like
        Discharge [m/s] indexed by time

    velocity : array-like
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
    ax = _xy_plot(
        discharge,
        velocity,
        fmt=".",
        label=label,
        xlabel="Discharge [$m^3/s$]",
        ylabel="Velocity [$m/s$]",
        ax=ax,
    )
    if polynomial_coeff:
        x = np.linspace(discharge.min(), discharge.max())
        ax = _xy_plot(
            x,
            polynomial_coeff(x),
            fmt="--",
            label="Polynomial fit",
            xlabel="Discharge [$m^3/s$]",
            ylabel="Velocity [$m/s$]",
            ax=ax,
        )

    return ax


def plot_velocity_vs_power(velocity, power, polynomial_coeff=None, label=None, ax=None):
    """
    Plots velocity vs power data along with the polynomial fit

    Parameters
    ------------
    velocity : array-like
        Velocity [m/s] indexed by time

    power: array-like
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
    ax = _xy_plot(
        velocity,
        power,
        fmt=".",
        label=label,
        xlabel="Velocity [$m/s$]",
        ylabel="Power [$W$]",
        ax=ax,
    )
    if polynomial_coeff:
        x = np.linspace(velocity.min(), velocity.max())
        ax = _xy_plot(
            x,
            polynomial_coeff(x),
            fmt="--",
            label="Polynomial fit",
            xlabel="Velocity [$m/s$]",
            ylabel="Power [$W$]",
            ax=ax,
        )

    return ax
