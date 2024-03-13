import numpy as np
import math
from mhkit.river.resource import exceedance_probability, Froude_number
from mhkit.utils import convert_to_dataarray


def _histogram(directions, velocities, width_dir, width_vel):
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
    N_dir = math.ceil(360 / width_dir)
    # Max bin (round up to nearest integer)
    vel_max = math.ceil(velocities.max())
    # Number of velocity bins
    N_vel = math.ceil(vel_max / width_vel)
    # 2D Histogram of current speed and direction
    H, dir_edges, vel_edges = np.histogram2d(
        directions,
        velocities,
        bins=(N_dir, N_vel),
        range=[[0, 360], [0, vel_max]],
        density=True,
    )
    # density = true therefore bin value * bin area summed =1
    bin_area = width_dir * width_vel
    # Convert H values to percent [%]
    H = H * bin_area * 100
    return H, dir_edges, vel_edges


def _normalize_angle(degree):
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


def principal_flow_directions(directions, width_dir):
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

    directions = convert_to_dataarray(directions)
    if any(directions < 0) or any(directions > 360):
        violating_values = [d for d in directions if d < 0 or d > 360]
        raise ValueError(
            f"directions must be between 0 and 360 degrees. Values out of range: {violating_values}"
        )

    # Number of directional bins
    N_dir = int(360 / width_dir)
    # Compute directional histogram
    H1, dir_edges = np.histogram(directions, bins=N_dir, range=[0, 360], density=True)
    # Convert to percent
    H1 = H1 * 100  # [%]
    # Determine if there are an even or odd number of bins
    odd = bool(N_dir % 2)
    # Shift by 180 degrees and sum
    if odd:
        # Then split middle bin counts to left and right
        H0to180 = H1[0 : N_dir // 2]
        H180to360 = H1[N_dir // 2 + 1 :]
        H0to180[-1] += H1[N_dir // 2] / 2
        H180to360[0] += H1[N_dir // 2] / 2
        # Add the two
        H180 = H0to180 + H180to360
    else:
        H180 = H1[0 : N_dir // 2] + H1[N_dir // 2 : N_dir + 1]

    # Find the maximum value
    maxDegreeStacked = H180.argmax()
    # Shift by 90 to find angles normal to principal direction
    floodEbbNormalDegree1 = _normalize_angle(maxDegreeStacked + 90.0)
    # Find the complimentary angle
    floodEbbNormalDegree2 = _normalize_angle(floodEbbNormalDegree1 + 180.0)
    # Reset values so that the Degree1 is the smaller angle, and Degree2 the large
    floodEbbNormalDegree1 = min(floodEbbNormalDegree1, floodEbbNormalDegree2)
    floodEbbNormalDegree2 = floodEbbNormalDegree1 + 180.0
    # Slice directions on the 2 semi circles
    mask = (directions >= floodEbbNormalDegree1) & (directions <= floodEbbNormalDegree2)
    d1 = directions[mask]
    d2 = directions[~mask]
    # Shift second set of of directions to not break between 360 and 0
    d2 -= 180
    # Renormalize the points (gets rid of negatives)
    d2 = _normalize_angle(d2)
    # Number of bins for semi-circle
    n_dir = int(180 / width_dir)
    # Compute 1D histograms on both semi circles
    Hd1, dir1_edges = np.histogram(d1, bins=n_dir, density=True)
    Hd2, dir2_edges = np.histogram(d2, bins=n_dir, density=True)
    # Convert to percent
    Hd1 = Hd1 * 100  # [%]
    Hd2 = Hd2 * 100  # [%]
    # Principal Directions average of the 2 bins
    PrincipalDirection1 = 0.5 * (
        dir1_edges[Hd1.argmax()] + dir1_edges[Hd1.argmax() + 1]
    )
    PrincipalDirection2 = (
        0.5 * (dir2_edges[Hd2.argmax()] + dir2_edges[Hd2.argmax() + 1]) + 180.0
    )

    return PrincipalDirection1, PrincipalDirection2


def _flood_or_ebb(d, flood, ebb):
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
