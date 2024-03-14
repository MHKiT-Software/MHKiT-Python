import numpy as np
import xarray as xr
from mhkit.utils import convert_to_dataarray

from mhkit import dolfyn
from mhkit.river.performance import (
    circular,
    ducted,
    rectangular,
    multiple_circular,
    tip_speed_ratio,
    power_coefficient,
)


def _slice_circular_capture_area(diameter, hub_height, doppler_cell_size):
    """
    Slices a circle (capture area) based on ADCP depth bins mapped
    across the face of the capture area.

    Parameters
    -------------
    diameter: numeric
        Diameter of the capture area.

    hub_height: numeric
        Turbine hub height altitude above the seabed. Assumes ADCP
        depth bins are referenced to the seafloor.

    doppler_cell_size: numeric
        ADCP depth bin size.

    Returns
    ---------
    capture_area_slice: xarray.DataArray
        Capture area sliced into horizontal slices of height
        `doppler_cell_size`, centered on `hub height`.
    """

    def area_of_circle_segment(radius, angle):
        # Calculating area of sector
        area_of_sector = np.pi * radius**2 * (angle / 360)
        # Calculating area of triangle
        area_of_triangle = 0.5 * radius**2 * np.sin((np.pi * angle) / 180)
        return area_of_sector - area_of_triangle

    def point_on_circle(y, r):
        return np.sqrt(r**2 - y**2)

    # Capture area - from mhkit.river.performance
    d = diameter
    cs = doppler_cell_size

    A_cap = np.pi * (d / 2) ** 2  # m^2
    # Need to chop up capture area into slices based on bin size
    # For a cirle:
    r_min = hub_height - d / 2
    r_max = hub_height + d / 2
    A_edge = np.arange(r_min, r_max + cs, cs)
    A_rng = A_edge[:-1] + cs / 2  # Center of each slice

    # y runs from the bottom edge of the lower centerline slice to
    # the top edge of the lowest slice
    # Will need to figure out y if the hub height isn't centered
    y = abs(A_edge - np.mean(A_edge))
    y[np.where(abs(y) > (d / 2))] = d / 2

    # Even vs odd number of slices
    if y.size % 2:
        odd = 1
    else:
        odd = 0
        y = y[: len(y) // 2]
        y = np.append(y, 0)

    x = point_on_circle(y, d / 2)
    radii = np.rad2deg(np.arctan(x / y) * 2)
    # Segments go from outside of circle towards middle
    As = area_of_circle_segment(d / 2, radii)
    # Subtract segments to get area of slices
    As_slc = As[1:] - As[:-1]

    if not odd:
        # Make middle slice half whole
        As_slc[-1] = As_slc[-1] * 2
        # Copy-flip the other slices to get the whole circle
        As_slc = np.append(As_slc, np.flip(As_slc[:-1]))
    else:
        As_slc = abs(As_slc)

    return xr.DataArray(As_slc, coords={"range": A_rng})


def _slice_rectangular_capture_area(height, width, hub_height, doppler_cell_size):
    """
    Slices a rectangular (capture area) based on ADCP depth bins mapped
    across the face of the capture area.

    Parameters
    -------------
    height: numeric
        Height of the capture area.

    width: numeric
        Width of the capture area.

    hub_height: numeric
        Turbine hub height altitude above the seabed. Assumes ADCP depth
        bins are referenced to the seafloor.

    doppler_cell_size: numeric
        ADCP depth bin size.

    Returns
    ---------
    capture_area_slice: xarray.DataArray
        Capture area sliced into horizontal slices of height
        `doppler_cell_size`, centered on `hub height`.
    """

    # Need to chop up capture area into slices based on bin size
    # For a rectangle it's pretty simple
    cs = doppler_cell_size
    r_min = hub_height - height / 2
    r_max = hub_height + height / 2
    A_edge = np.arange(r_min, r_max + cs, cs)
    A_rng = A_edge[:-1] + cs / 2  # Center of each slice

    As_slc = np.ones(len(A_rng)) * width * cs

    return xr.DataArray(As_slc, coords={"range": A_rng})


def power_curve(
    power,
    velocity,
    hub_height,
    doppler_cell_size,
    sampling_frequency,
    window_avg_time=600,
    turbine_profile="circular",
    diameter=None,
    height=None,
    width=None,
    to_pandas=True,
):
    """
    Calculates power curve and power statistics for a marine energy
    device based on IEC/TS 62600-200 section 9.3.

    Parameters
    -------------
    power: numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Device power output timeseries.
    velocity: numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        1D or 2D streamwise sea water velocity or sea water speed.
    hub_height: numeric
        Turbine hub height altitude above the seabed. Assumes ADCP
        depth bins are referenced to the seafloor.
    doppler_cell_size: numeric
        ADCP depth bin size.
    sampling_frequency: numeric
        ADCP sampling frequency in Hz.
    window_avg_time: int, optional
        Time averaging window in seconds. Defaults to 600.
    turbine_profile: 'circular' or 'rectangular', optional
        Shape of swept area of the turbine. Defaults to 'circular'.
    diameter: numeric, optional
        Required for turbine_profile='circular'. Defaults to None.
    height: numeric, optional
        Required for turbine_profile='rectangular'. Defaults to None.
    width: numeric, optional
        Required for turbine_profile='rectangular'. Defaults to None.
    to_pandas: bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    device_power_curve: pandas DataFrame or xarray Dataset
        Power-weighted velocity, mean power, power std dev, max and
        min power vs hub-height velocity.
    """

    # Velocity should be a 2D xarray or pandas array and have dims (range, time)
    # Power should have a timestamp coordinate/index
    power = convert_to_dataarray(power)
    velocity = convert_to_dataarray(velocity)
    if len(velocity.shape) != 2:
        raise ValueError(
            "Velocity should be 2 dimensional and have \
                         dimensions of 'time' (temporal) and 'range' (spatial)."
        )

    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Numeric positive checks
    numeric_params = [
        hub_height,
        doppler_cell_size,
        sampling_frequency,
        window_avg_time,
    ]
    numeric_param_names = [
        "hub_height",
        "doppler_cell_size",
        "sampling_frequency",
        "window_avg_time",
    ]
    for param, name in zip(numeric_params, numeric_param_names):
        if not isinstance(param, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if param <= 0:
            raise ValueError(f"{name} must be positive.")

    # Turbine profile related checks
    if turbine_profile not in ["circular", "rectangular"]:
        raise ValueError(
            "`turbine_profile` must be one of 'circular' or 'rectangular'."
        )
    if turbine_profile == "circular":
        if diameter is None:
            raise TypeError(
                "`diameter` cannot be None for input `turbine_profile` = 'circular'."
            )
        elif not isinstance(diameter, (int, float)) or diameter <= 0:
            raise ValueError("`diameter` must be a positive number.")
        else:  # If the checks pass, calculate A_slc
            A_slc = _slice_circular_capture_area(
                diameter, hub_height, doppler_cell_size
            )
    else:  # Rectangular profile
        if height is None or width is None:
            raise TypeError(
                "`height` and `width` cannot be None for input `turbine_profile` = 'rectangular'."
            )
        elif not all(
            isinstance(val, (int, float)) and val > 0 for val in [height, width]
        ):
            raise ValueError("`height` and `width` must be positive numbers.")
        else:  # If the checks pass, calculate A_slc
            A_slc = _slice_rectangular_capture_area(
                height, width, hub_height, doppler_cell_size
            )

    # Streamwise data
    U = abs(velocity)
    time = U["time"].values
    # Interpolate power to velocity timestamps
    P = power.interp(time=U["time"], method="linear")

    # Power weighted velocity in capture area
    # Interpolate U range to capture area slices, then cube and multiply by area
    U_hat = U.interp(range=A_slc["range"], method="linear") ** 3 * A_slc
    # Average the velocity across the capture area and divide out area
    U_hat = (U_hat.sum("range") / A_slc.sum()) ** (-1 / 3)

    # Time-average velocity at hub-height
    bnr = dolfyn.VelBinner(
        n_bin=window_avg_time * sampling_frequency, fs=sampling_frequency
    )
    # Hub-height velocity mean
    mean_hub_vel = xr.DataArray(
        bnr.mean(U.sel(range=hub_height, method="nearest").values),
        coords={"time": bnr.mean(time)},
    )

    # Power-weighted hub-height velocity mean
    U_hat_bar = xr.DataArray(
        (bnr.mean(U_hat.values**3)) ** (-1 / 3), coords={"time": bnr.mean(time)}
    )

    # Average power
    P_bar = xr.DataArray(bnr.mean(P.values), coords={"time": bnr.mean(time)})

    # Then reorganize into 0.1 m velocity bins and average
    U_bins = np.arange(0, np.nanmax(mean_hub_vel) + 0.1, 0.1)
    U_hub_vel = mean_hub_vel.assign_coords({"time": mean_hub_vel}).rename(
        {"time": "speed"}
    )
    U_hub_mean = U_hub_vel.groupby_bins("speed", U_bins).mean()
    U_hat_vel = U_hat_bar.assign_coords({"time": mean_hub_vel}).rename(
        {"time": "speed"}
    )
    U_hat_mean = U_hat_vel.groupby_bins("speed", U_bins).mean()

    P_bar_vel = P_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})
    P_bar_mean = P_bar_vel.groupby_bins("speed", U_bins).mean()
    P_bar_std = P_bar_vel.groupby_bins("speed", U_bins).std()
    P_bar_max = P_bar_vel.groupby_bins("speed", U_bins).max()
    P_bar_min = P_bar_vel.groupby_bins("speed", U_bins).min()

    device_power_curve = xr.Dataset(
        {
            "U_avg": U_hub_mean,
            "U_avg_power_weighted": U_hat_mean,
            "P_avg": P_bar_mean,
            "P_std": P_bar_std,
            "P_max": P_bar_max,
            "P_min": P_bar_min,
        }
    )
    device_power_curve = device_power_curve.rename({"speed_bins": "U_bins"})

    if to_pandas:
        device_power_curve = device_power_curve.to_pandas()

    return device_power_curve


def _average_velocity_bins(U, U_hub, bin_size):
    """
    Groups time-ensembles into velocity bins based on hub-height
    velocity and averages them.

    Parameters
    -------------
    U: xarray.DataArray
        Input variable to group by velocity.
    U_hub: xarray.DataArray
        Sea water velocity at hub height.
    bin_size: numeric
        Velocity averaging window size in m/s.

    Returns
    ---------
    U_binned: xarray.DataArray
        Data grouped into velocity bins.
    """

    # Reorganize into velocity bins and average
    U_bins = np.arange(0, np.nanmax(U_hub) + bin_size, bin_size)

    # Group time-ensembles into velocity bins based on hub-height velocity and average
    U_binned = U.assign_coords({"time": U_hub}).rename({"time": "speed"})
    U_binned = U_binned.groupby_bins("speed", U_bins).mean()

    return U_binned


def _apply_function(function, bnr, U):
    """
    Applies a specified function ('mean', 'rms', or 'std') to the input
    data array U, grouped into bins as specified by the binning rules in bnr.

    Parameters
    -------------
    function: str
        The name of the function to apply. Must be one of 'mean',
        'rms', or 'std'.
    bnr: dolfyn.VelBinner or similar
        The binning rule object that determines how data in U is
        grouped into bins.
    U: xarray.DataArray
        The input data array to which the function is applied.

    Returns
    ---------
    xarray.DataArray
        The input data array U after the specified function has been
        applied, grouped into bins according to bnr.
    """

    if function == "mean":
        # Average data into 5-10 minute ensembles
        return xr.DataArray(
            bnr.mean(abs(U).values),
            coords={"range": U.range, "time": bnr.mean(U["time"].values)},
        )
    elif function == "rms":
        # Reshape tidal velocity - returns (range, ensemble-time, ensemble elements)
        U_reshaped = bnr.reshape(abs(U).values)
        # Take root-mean-square
        U_rms = np.sqrt(np.nanmean(U_reshaped**2, axis=-1))
        return xr.DataArray(
            U_rms, coords={"range": U.range, "time": bnr.mean(U["time"].values)}
        )
    elif function == "std":
        # Standard deviation
        return xr.DataArray(
            bnr.standard_deviation(U.values),
            coords={"range": U.range, "time": bnr.mean(U["time"].values)},
        )
    else:
        raise ValueError(
            f"Unknown function {function}. Should be one of 'mean', 'rms', or 'std'"
        )


def velocity_profiles(
    velocity,
    hub_height,
    water_depth,
    sampling_frequency,
    window_avg_time=600,
    function="mean",
    to_pandas=True,
):
    """
    Calculates profiles of the mean, root-mean-square (RMS), or standard
    deviation(std) of velocity. The chosen metric, specified by `function`,
    is calculated for each `window_avg_time` and bin-averaged based on
    ensemble velocity, as per IEC/TS 62600-200 sections 9.4 and 9.5.

    Parameters
    -------------
    velocity : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        1D or 2D streamwise sea water velocity or sea water speed.
    hub_height : numeric
        Turbine hub height altitude above the seabed. Assumes ADCP depth bins
        are referenced to the seafloor.
    water_depth : numeric
        Water depth to seafloor, in same units as velocity `range` coordinate.
    sampling_frequency : numeric
        ADCP sampling frequency in Hz.
    window_avg_time : int, optional
        Time averaging window in seconds. Defaults to 600.
    func : string
        Function to apply. One of 'mean','rms', or 'std'
    to_pandas: bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    iec_profiles: pandas.DataFrame
        Average velocity profiles based on ensemble mean velocity.
    """

    velocity = convert_to_dataarray(velocity, "velocity")
    if len(velocity.shape) != 2:
        raise ValueError(
            "Velocity should be 2 dimensional and have \
                         dimensions of 'time' (temporal) and 'range' (spatial)."
        )

    if function not in ["mean", "rms", "std"]:
        raise ValueError("`function` must be one of 'mean', 'rms', or 'std'.")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Streamwise data
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(
        n_bin=window_avg_time * sampling_frequency, fs=sampling_frequency
    )
    # Take velocity at hub height
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method="nearest").values)

    # Apply mean, root-mean-square, or standard deviation
    U_out = _apply_function(function, bnr, U)

    # Then reorganize into 0.5 m/s velocity bins and average
    profiles = _average_velocity_bins(U_out, mean_hub_vel, bin_size=0.5)

    # Extend top and bottom of profiles to the seafloor and sea surface
    # Clip off extra depth bins with nans
    rdx = profiles.isel(speed_bins=0).notnull().sum().values
    profiles = profiles.isel(range=slice(None, rdx + 1))
    # Set seafloor velocity to 0 m/s
    out_data = np.insert(profiles.data, 0, 0, axis=0)
    # Set max range to the user-provided water depth
    new_range = np.insert(profiles["range"].data[:-1], 0, 0)
    new_range = np.append(new_range, water_depth)
    # Create a profiles with new range
    iec_profiles = xr.DataArray(
        out_data, coords={"range": new_range, "speed_bins": profiles["speed_bins"]}
    )
    # Forward fill to surface
    iec_profiles = iec_profiles.ffill("range", limit=None)

    if to_pandas:
        iec_profiles = iec_profiles.to_pandas()

    return iec_profiles


def device_efficiency(
    power,
    velocity,
    water_density,
    capture_area,
    hub_height,
    sampling_frequency,
    window_avg_time=600,
    to_pandas=True,
):
    """
    Calculates marine energy device efficiency based on IEC/TS 62600-200 Section 9.7.

    Parameters
    -------------
    power : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        Device power output timeseries in Watts.
    velocity : numpy ndarray, pandas DataFrame, pandas Series, xarray DataArray, or xarray Dataset
        1D or 2D streamwise sea water velocity or sea water speed in m/s.
    water_density : float, pandas.Series or xarray.DataArray
        Sea water density in kg/m^3.
    capture_area : numeric
        Swept area of marine energy device.
    hub_height : numeric
        Turbine hub height altitude above the seabed. Assumes ADCP depth bins
        are referenced to the seafloor.
    sampling_frequency : numeric
        ADCP sampling frequency in Hz.
    window_avg_time : int, optional
        Time averaging window in seconds. Defaults to 600.
    to_pandas: bool, optional
        Flag to output pandas instead of xarray. Default = True.

    Returns
    ---------
    device_eta : pandas.Series or xarray.DataArray
        Device efficiency (power coefficient) in percent.
    """

    # Velocity should be a 2D xarray or pandas array and have dims (range, time)
    # Power should have a timestamp coordinate/index
    power = convert_to_dataarray(power, "power")
    velocity = convert_to_dataarray(velocity, "velocity")
    if len(velocity.shape) != 2:
        raise ValueError(
            "Velocity should be 2 dimensional and have \
                            dimensions of 'time' (temporal) and 'range' (spatial)."
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Streamwise data
    U = abs(velocity)
    time = U["time"].values

    # Power: Interpolate to velocity timeseries
    power.interp(time=U["time"], method="linear")

    # Create binner
    bnr = dolfyn.VelBinner(
        n_bin=window_avg_time * sampling_frequency, fs=sampling_frequency
    )
    # Hub-height velocity
    mean_hub_vel = xr.DataArray(
        bnr.mean(U.sel(range=hub_height, method="nearest").values),
        coords={"time": bnr.mean(time)},
    )
    vel_hub = _average_velocity_bins(mean_hub_vel, mean_hub_vel, bin_size=0.1)

    # Water density
    rho_vel = _calculate_density(water_density, bnr, mean_hub_vel, time)

    # Bin average power
    P_avg = xr.DataArray(bnr.mean(power.values), coords={"time": bnr.mean(time)})
    P_vel = _average_velocity_bins(P_avg, mean_hub_vel, bin_size=0.1)

    # Theoretical power resource
    P_resource = 1 / 2 * rho_vel * capture_area * vel_hub**3

    # Efficiency
    eta = P_vel / P_resource

    device_eta = xr.Dataset({"U_avg": vel_hub, "Efficiency": eta})
    device_eta = device_eta.rename({"speed_bins": "U_bins"})

    if to_pandas:
        device_eta = device_eta.to_pandas()

    return device_eta


def _calculate_density(water_density, bnr, mean_hub_vel, time):
    """
    Calculates the averaged density for the given time period.

    This function first checks if the water_density is a scalar or an array.
    If it is an array, the function calculates the mean density over the time
    period using the binner object 'bnr', and then averages it over velocity bins.
    If it is a scalar, it directly returns the input density.

    Parameters
    -------------
    water_density : numpy.ndarray or float
        Sea water density values in kg/m^3. It can be a scalar or a 1D array.
    bnr : dolfyn.VelBinner object
        Object for binning data over specified time periods.
    mean_hub_vel : xarray.DataArray
        Mean velocity at the hub height.
    time : numpy.ndarray
        Time data array.

    Returns
    ---------
    xarray.DataArray or float
        The averaged water density over velocity bins if water_density is an array,
        or the input scalar water_density.
    """

    if np.size(water_density) > 1:
        rho_avg = xr.DataArray(
            bnr.mean(water_density.values), coords={"time": bnr.mean(time)}
        )
        return _average_velocity_bins(rho_avg, mean_hub_vel, bin_size=0.1)
    else:
        return water_density
