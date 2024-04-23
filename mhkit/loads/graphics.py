"""
This module provides functionalities for plotting statistical data
related to a given variable or dataset. 

    - `plot_statistics` is designed to plot raw statistical measures
      (mean, maximum, minimum, and optional standard deviation) of a
      variable across a series of x-axis values. It allows for
      customization of plot labels, title, and saving the plot to a file.

    - `plot_bin_statistics` extends these capabilities to binned data,
      offering a way to visualize binned statistics (mean, maximum, minimum)
      along with their respective standard deviations. This function also 
      supports label and title customization, as well as saving the plot to 
      a specified path.
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from mhkit.utils.type_handling import to_numeric_array


# pylint: disable=R0914
def plot_statistics(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_max: np.ndarray,
    y_min: np.ndarray,
    y_stdev: Optional[np.ndarray] = None,
    **kwargs: Dict[str, Any],
) -> plt.Axes:
    """
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
    """
    if y_stdev is None:
        y_stdev = []

    input_variables = [x, y_mean, y_max, y_min, y_stdev]

    variable_names = ["x", "y_mean", "y_max", "y_min", "y_stdev"]
    # Convert each input variable to a numeric array, ensuring all are numeric
    for i, variable in enumerate(input_variables):
        input_variables[i] = to_numeric_array(variable, variable_names[i])

    x, y_mean, y_max, y_min, y_stdev = input_variables

    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", None)
    title = kwargs.get("title", None)
    save_path = kwargs.get("save_path", None)

    if not isinstance(x_label, (str, type(None))):
        raise TypeError(f"x_label must be of type str. Got: {type(x_label)}")
    if not isinstance(y_label, (str, type(None))):
        raise TypeError(f"y_label must be of type str. Got: {type(y_label)}")
    if not isinstance(title, (str, type(None))):
        raise TypeError(f"title must be of type str. Got: {type(title)}")
    if not isinstance(save_path, (str, type(None))):
        raise TypeError(f"save_path must be of type str. Got: {type(save_path)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_max, "^", label="max", mfc="none")
    ax.plot(x, y_mean, "o", label="mean", mfc="none")
    ax.plot(x, y_min, "v", label="min", mfc="none")

    if len(y_stdev) > 0:
        ax.plot(x, y_stdev, "+", label="stdev", c="m")
    ax.grid(alpha=0.4)
    ax.legend(loc="best")

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path)
        plt.close()
    return ax


# pylint: disable=R0913
def plot_bin_statistics(
    bin_centers: np.ndarray,
    bin_mean: np.ndarray,
    bin_max: np.ndarray,
    bin_min: np.ndarray,
    bin_mean_std: np.ndarray,
    bin_max_std: np.ndarray,
    bin_min_std: np.ndarray,
    **kwargs: Dict[str, Any],
) -> plt.Axes:
    """
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
    """

    input_variables = [
        bin_centers,
        bin_mean,
        bin_max,
        bin_min,
        bin_mean_std,
        bin_max_std,
        bin_min_std,
    ]
    variable_names = [
        "bin_centers",
        "bin_mean",
        "bin_max",
        "bin_min",
        "bin_mean_std",
        "bin_max_std",
        "bin_min_std",
    ]

    # Convert each input variable to a numeric array, ensuring all are numeric
    for i, variable in enumerate(input_variables):
        input_variables[i] = to_numeric_array(variable, variable_names[i])

    (
        bin_centers,
        bin_mean,
        bin_max,
        bin_min,
        bin_mean_std,
        bin_max_std,
        bin_min_std,
    ) = input_variables

    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", None)
    title = kwargs.get("title", None)
    save_path = kwargs.get("save_path", None)

    if not isinstance(x_label, (str, type(None))):
        raise TypeError(f"x_label must be of type str. Got: {type(x_label)}")
    if not isinstance(y_label, (str, type(None))):
        raise TypeError(f"y_label must be of type str. Got: {type(y_label)}")
    if not isinstance(title, (str, type(None))):
        raise TypeError(f"title must be of type str. Got: {type(title)}")
    if not isinstance(save_path, (str, type(None))):
        raise TypeError(f"save_path must be of type str. Got: {type(save_path)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        bin_centers,
        bin_max,
        marker="^",
        mfc="none",
        yerr=bin_max_std,
        capsize=4,
        label="max",
    )
    ax.errorbar(
        bin_centers,
        bin_mean,
        marker="o",
        mfc="none",
        yerr=bin_mean_std,
        capsize=4,
        label="mean",
    )
    ax.errorbar(
        bin_centers,
        bin_min,
        marker="v",
        mfc="none",
        yerr=bin_min_std,
        capsize=4,
        label="min",
    )

    ax.grid(alpha=0.5)
    ax.legend(loc="best")

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path)
        plt.close()
    return ax
