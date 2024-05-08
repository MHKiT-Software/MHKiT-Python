import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mhkit.utils import convert_to_dataarray


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
    temp = xr.Dataset(data_vars={"D": D, "F": F})
    temp.sortby("F", ascending=False)

    ax = _xy_plot(
        temp["D"],
        temp["F"],
        fmt="-",
        label=label,
        xlabel="Discharge [$m^3/s$]",
        ylabel="Exceedance Probability",
        ax=ax,
    )
    plt.xscale("log")

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
    temp = xr.Dataset(data_vars={"V": V, "F": F})
    temp.sortby("F", ascending=False)

    ax = _xy_plot(
        temp["V"],
        temp["F"],
        fmt="-",
        label=label,
        xlabel="Velocity [$m/s$]",
        ylabel="Exceedance Probability",
        ax=ax,
    )

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
    temp = xr.Dataset(data_vars={"P": P, "F": F})
    temp.sortby("F", ascending=False)

    ax = _xy_plot(
        temp["P"],
        temp["F"],
        fmt="-",
        label=label,
        xlabel="Power [W]",
        ylabel="Exceedance Probability",
        ax=ax,
    )

    return ax


def plot_discharge_timeseries(Q, time_dimension="", label=None, ax=None):
    """
    Plots discharge time-series

    Parameters
    ------------
    Q: array-like
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
    Q = convert_to_dataarray(Q)

    if time_dimension == "":
        time_dimension = list(Q.coords)[0]

    ax = _xy_plot(
        Q.coords[time_dimension].values,
        Q,
        fmt="-",
        label=label,
        xlabel="Time",
        ylabel="Discharge [$m^3/s$]",
        ax=ax,
    )

    return ax


def plot_discharge_vs_velocity(D, V, polynomial_coeff=None, label=None, ax=None):
    """
    Plots discharge vs velocity data along with the polynomial fit

    Parameters
    ------------
    D : array-like
        Discharge [m/s] indexed by time

    V : array-like
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
        D,
        V,
        fmt=".",
        label=label,
        xlabel="Discharge [$m^3/s$]",
        ylabel="Velocity [$m/s$]",
        ax=ax,
    )
    if polynomial_coeff:
        x = np.linspace(D.min(), D.max())
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


def plot_velocity_vs_power(V, P, polynomial_coeff=None, label=None, ax=None):
    """
    Plots velocity vs power data along with the polynomial fit

    Parameters
    ------------
    V : array-like
        Velocity [m/s] indexed by time

    P: array-like
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
        V,
        P,
        fmt=".",
        label=label,
        xlabel="Velocity [$m/s$]",
        ylabel="Power [$W$]",
        ax=ax,
    )
    if polynomial_coeff:
        x = np.linspace(V.min(), V.max())
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
