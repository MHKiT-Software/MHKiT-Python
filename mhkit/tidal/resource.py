"""
This module provides utility functions for analyzing river and tidal flow
directions and velocities, including principal flow direction calculation,
histogram-based probability distributions, and ebb/flood classification.

Functions
---------
- `_histogram`: Computes a joint probability histogram of flow directions
  and velocities.
- `_normalize_angle`: Normalizes an angle to the range [0, 360] degrees.
- `principal_flow_directions`: Determines principal flow directions for
  ebb and flood cycles.
- `_flood_or_ebb`: Identifies whether flow directions correspond to ebb
  or flood conditions.
"""

import math
import numpy as np
from mhkit.river.resource import exceedance_probability, Froude_number
from mhkit.utils import convert_to_dataarray

__all__ = [
    "exceedance_probability",
    "Froude_number",
    "principal_flow_directions",
    "_histogram",
    "_flood_or_ebb",
]


def _histogram(
    directions: np.ndarray, velocities: np.ndarray, width_dir: float, width_vel: float
) -> tuple[np.ndarray, list, list]:
    """
    Wrapper around numpy histogram 2D. Used to find joint probability
    between directions and velocities. Returns joint probability H as [%].

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
    Returns
    -------
    H: matrix
        Joint probability as [%]
    dir_edges: list
        List of directional bin edges
    vel_edges: list
        List of velocity bin edges
    """

    # Number of directional bins
    n_dir = math.ceil(360 / width_dir)
    # Max bin (round up to nearest integer)
    velocity_max = math.ceil(velocities.max())
    # Number of velocity bins
    n_vel = math.ceil(velocity_max / width_vel)
    # 2D Histogram of current speed and direction
    joint_probability, dir_edges, vel_edges = np.histogram2d(
        directions,
        velocities,
        bins=(n_dir, n_vel),
        range=[[0, 360], [0, velocity_max]],
        density=True,
    )
    # density = true therefore bin value * bin area summed =1
    bin_area = width_dir * width_vel
    # Convert joint_probability values to percent [%]
    joint_probability = joint_probability * bin_area * 100
    return joint_probability, dir_edges, vel_edges


def _normalize_angle(degree: float) -> float:
    """
    Normalizes degrees to be between 0 and 360

    Parameters
    ----------
    degree: int or float

    Returns
    -------
    new_degree: float
        Normalized between 0 and 360 degrees
    """
    # Set new degree as remainder
    new_degree = degree % 360
    # Ensure positive
    new_degree = (new_degree + 360) % 360
    return new_degree


def principal_flow_directions(
    directions: np.ndarray, width_dir: float
) -> tuple[float, float]:
    """
    Calculates principal flow directions for ebb and flood cycles

    The weighted average (over the working velocity range of the TEC)
    should be considered to be the principal direction of the current,
    and should be used for both the ebb and flood cycles to determine
    the TEC optimum orientation.

    Parameters
    ----------
    directions: numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Flow direction in degrees CW from North, from 0 to 360
    width_dir: float
        Width of directional bins for histogram in degrees

    Returns
    -------
    principal directions: tuple(float,float)
        Principal directions 1 and 2 in degrees

    Notes
    -----
    One must determine which principal direction is flood and which is
    ebb based on knowledge of the measurement site.
    """
    # pylint: disable=too-many-locals

    directions = convert_to_dataarray(directions)
    if any(directions < 0) or any(directions > 360):
        violating_values = [d for d in directions if d < 0 or d > 360]
        raise ValueError(
            f"directions must be between 0 and 360 degrees. Values out of range: {violating_values}"
        )

    # Number of directional bins
    n_dir = int(360 / width_dir)
    # Compute directional histogram
    histogram1, _ = np.histogram(directions, bins=n_dir, range=[0, 360], density=True)
    # Convert to percent
    histogram1 = histogram1 * 100  # [%]
    # Determine if there are an even or odd number of bins
    odd = bool(n_dir % 2)
    # Shift by 180 degrees and sum
    if odd:
        # Then split middle bin counts to left and right
        histogram_0_to_180 = histogram1[0 : n_dir // 2]
        histogram_180_to_360 = histogram1[n_dir // 2 + 1 :]
        histogram_0_to_180[-1] += histogram1[n_dir // 2] / 2
        histogram_180_to_360[0] += histogram1[n_dir // 2] / 2
        # Add the two
        histogram_180 = histogram_0_to_180 + histogram_180_to_360
    else:
        histogram_180 = histogram1[0 : n_dir // 2] + histogram1[n_dir // 2 : n_dir + 1]

    # Find the maximum value
    max_degree_stacked = histogram_180.argmax()
    # Shift by 90 to find angles normal to principal direction
    flood_ebb_normal_degree1 = _normalize_angle(max_degree_stacked + 90.0)
    # Find the complimentary angle
    flood_ebb_normal_degree2 = _normalize_angle(flood_ebb_normal_degree1 + 180.0)
    # Reset values so that the Degree1 is the smaller angle, and Degree2 the large
    flood_ebb_normal_degree1 = min(flood_ebb_normal_degree1, flood_ebb_normal_degree2)
    flood_ebb_normal_degree2 = flood_ebb_normal_degree1 + 180.0
    # Slice directions on the 2 semi circles
    mask = (directions >= flood_ebb_normal_degree1) & (
        directions <= flood_ebb_normal_degree2
    )
    d1 = directions[mask]
    d2 = directions[~mask]
    # Shift second set of of directions to not break between 360 and 0
    d2 -= 180
    # Renormalize the points (gets rid of negatives)
    d2 = _normalize_angle(d2)
    # Number of bins for semi-circle
    n_dir = int(180 / width_dir)
    # Compute 1D histograms on both semi circles
    histogram_d1, dir1_edges = np.histogram(d1, bins=n_dir, density=True)
    histogram_d2, dir2_edges = np.histogram(d2, bins=n_dir, density=True)
    # Convert to percent
    histogram_d1 = histogram_d1 * 100  # [%]
    histogram_d2 = histogram_d2 * 100  # [%]
    # Principal Directions average of the 2 bins
    principal_direction1 = 0.5 * (
        dir1_edges[histogram_d1.argmax()] + dir1_edges[histogram_d1.argmax() + 1]
    )
    principal_direction2 = (
        0.5
        * (dir2_edges[histogram_d2.argmax()] + dir2_edges[histogram_d2.argmax() + 1])
        + 180.0
    )

    return principal_direction1, principal_direction2


def _flood_or_ebb(d: np.ndarray, flood: float, ebb: float) -> np.ndarray:
    """
    Returns a mask which is True for directions on the ebb side of the
    midpoints between the flood and ebb directions on the unit circle
    and False for directions on the Flood side.

    Parameters
    ----------
    d: array-like
        Directions to considered of length N
    flood: float or int
        Principal component of flow in the flood direction in degrees
    ebb: float or int
        Principal component of flow in the ebb direction in degrees

    Returns
    -------
    is_ebb: boolean array
        array of length N which is True for directions on the ebb side
        of the midpoints between flood and ebb on the unit circle and
        false otherwise.
    """
    max_angle = max(ebb, flood)
    min_angle = min(ebb, flood)

    lower_split = (min_angle + (360 - max_angle + min_angle) / 2) % 360
    upper_split = lower_split + 180

    if lower_split <= ebb < upper_split:
        is_ebb = ((d < upper_split) & (d >= lower_split)).values
    else:
        is_ebb = ~((d < upper_split) & (d >= lower_split)).values

    return is_ebb
