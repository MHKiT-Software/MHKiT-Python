import numpy as np
import pandas as pd
import xarray as xr
import warnings

from mhkit import dolfyn
from mhkit.river.performance import (circular, ducted, rectangular, 
                                     multiple_circular, tip_speed_ratio,
                                     power_coefficient)


def _slice_circular_capture_area(diameter, hub_height, doppler_cell_size):
    """
    Slices a circle (capture area) based on ADCP depth bins mapped 
    across the face of the capture area

    Args:
        diameter (numeric): Diameter of the capture area.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        doppler_cell_size (numeric): ADCP depth bin size.

    Returns:
        xarray.DataArray: Capture area sliced into horizontal slices of 
        height `doppler_cell_size`, centered on `hub height`.
    """

    def area_of_circle_segment(radius, angle):
        # Calculating area of sector
        area_of_sector = np.pi * radius**2 * (angle/360)
        # Calculating area of triangle
        area_of_triangle = 0.5 * radius**2 * np.sin((np.pi*angle)/180)
        return area_of_sector - area_of_triangle

    def point_on_circle(y, r):
        return np.sqrt(r**2 - y**2)

    # Capture area - from mhkit.river.performance
    d = diameter
    cs = doppler_cell_size

    A_cap = np.pi*(d/2)**2  # m^2
    # Need to chop up capture area into slices based on bin size
    ## For a cirle:
    r_min = hub_height - d/2
    r_max = hub_height + d/2
    A_edge = np.arange(r_min, r_max+cs, cs)
    A_rng = A_edge[:-1] + cs/2 # Center of each slice

    # y runs from the bottom edge of the lower centerline slice to 
    # the top edge of the lowest slice
    # Will need to figure out y if the hub height isn't centered
    y = abs(A_edge - np.mean(A_edge))
    y[np.where(abs(y)>(d/2))] = d/2

    # Even vs odd number of slices
    if y.size % 2:
        odd = 1
    else:
        odd = 0
        y = y[:len(y)//2]
        y = np.append(y, 0)

    x = point_on_circle(y, d/2)
    radii = np.rad2deg(np.arctan(x/y)*2)
    # Segments go from outside of circle towards middle
    As = area_of_circle_segment(d/2, radii)
    # Subtract segments to get area of slices
    As_slc = As[1:] - As[:-1]

    if not odd:
        # Make middle slice half whole
        As_slc[-1] = As_slc[-1]*2
        # Copy-flip the other slices to get the whole circle
        As_slc = np.append(As_slc, np.flip(As_slc[:-1]))
    else:
        As_slc = abs(As_slc)

    # Make sure the circle was sliced correctly
    assert(round(A_cap,6)==round(As_slc.sum(),6))

    return xr.DataArray(As_slc, coords={'range': A_rng})


def _slice_rectangular_capture_area(height, width, hub_height, doppler_cell_size):
    """
    Slices a rectangular (capture area) based on ADCP depth bins mapped 
    across the face of the capture area

    Args:
        height (numeric): Height of the capture area.
        width (numeric): Width of the capture area.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        doppler_cell_size (numeric): ADCP depth bin size.

    Returns:
        xarray.DataArray: Capture area sliced into horizontal slices of 
        height `doppler_cell_size`, centered on `hub height`.
    """
    # Capture area - from mhkit.river.performance
    d_equiv, A_cap = rectangular(h=height, w=width)  # m^2
    cs = doppler_cell_size

    # Need to chop up capture area into slices based on bin size
    # For a rectangle it's pretty simple
    r_min = hub_height - height/2
    r_max = hub_height + height/2
    A_edge = np.arange(r_min, r_max+cs, cs)
    A_rng = A_edge[:-1] + cs/2 # Center of each slice

    As_slc = np.ones(len(A_rng))*width*cs

    # Make sure the rectangle was sliced correctly
    assert(round(A_cap,6)==round(As_slc.sum(),6))

    return xr.DataArray(As_slc, coords={'range': A_rng})


def power_curve(power, 
                velocity,
                hub_height,
                doppler_cell_size, 
                sampling_frequency, 
                window_avg_time=600,
                turbine_profile='circular',
                diameter=None,
                height=None,
                width=None):
    """
    Calculate power curve and power statistics for a marine energy device 
    based on IEC/TS 62600-200 section 9.3.

    Args:
        power (pandas.Series or xarray.DataArray (time)): Device power 
        output timeseries.
        velocity (pandas.Series or xarray.DataArray ([range,] time)): 
        Streamwise sea water velocity or sea water speed.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        doppler_cell_size (numeric): ADCP depth bin size.
        sampling_frequency (numeric): ADCP sampling frequency in Hz.
        window_avg_time (int, optional): Time averaging window in seconds. 
        Defaults to 600.
        turbine_profile ('circular' or 'rectangular'): Shape of swept area
        of the turbine. Defaults to 'circular'.

        diameter (numeric): Required for turbine_profile='circular'. 
        Defaults to None.
        height (numeric): Required for turbine_profile='rectangular'. 
        Defaults to None.
        width (numeric): Required for turbine_profile='rectangular'. 
        Defaults to None.

    Returns:
        pandas.DataFrame: Power-weighted velocity, mean power, power std dev,
         max and min power vs hub-height velocity.
    """

    # Assert velocity is 2D xarray or pandas and has dims range, time
    # Power should have a timestamp coordinate/index
    dtype = type(velocity)

    if turbine_profile=='rectangular':
        if height is None:
            raise Exception("`height` cannot be None for input `turbine_profile` = 'rectangular'.")
        if width is None:
            raise Exception("`width` cannot be None for input `turbine_profile` = 'rectangular'.")
        A_slc = _slice_rectangular_capture_area(height, width, hub_height, doppler_cell_size)
    elif turbine_profile=='circular':
        if diameter is None:
            raise Exception("`diameter` cannot be None for input `turbine_profile` = 'circular'.")
        A_slc = _slice_circular_capture_area(diameter, hub_height, doppler_cell_size)

    # Streamwise data
    U = abs(velocity)
    time = U['time'].values
    # Interpolate power to velocity timestamps
    P = power.interp(time=U['time'], method='linear')

    ## Power weighted velocity in capture area
    # Interpolate U range to capture area slices, then cube and multiply by area
    U_hat = U.interp(range=A_slc['range'], method='linear')**3 * A_slc
    # Average the velocity across the capture area and divide out area
    U_hat = (U_hat.mean('range') / A_slc.sum()) ** (-1/3)

    # Time-average velocity at hub-height
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Hub-height velocity mean
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)
    U_hub_bar = xr.DataArray(mean_hub_vel, coords={'time': bnr.mean(time)})
    
    # Power-weighted hub-height velocity mean
    U_hat_bar = xr.DataArray((bnr.mean(U_hat.values ** 3)) ** (-1/3), 
                              coords={'time': bnr.mean(time)})
    
    # Average power
    P_bar = xr.DataArray(bnr.mean(P.values),
                         coords={'time': bnr.mean(time)})

    # Then reorganize into 0.1 m velocity bins and average
    U_bins = np.arange(0, mean_hub_vel.max() + 0.1, 0.1)
    U_hub_vel = U_hub_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})
    U_hub_mean = U_hub_vel.groupby_bins("speed", U_bins).mean()
    U_hat_vel = U_hat_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})
    U_hat_mean = U_hat_vel.groupby_bins("speed", U_bins).mean()
    
    P_bar_vel = P_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})   
    P_bar_mean = P_bar_vel.groupby_bins("speed", U_bins).mean()
    P_bar_std = P_bar_vel.groupby_bins("speed", U_bins).std()
    P_bar_max = P_bar_vel.groupby_bins("speed", U_bins).max()
    P_bar_min = P_bar_vel.groupby_bins("speed", U_bins).min()

    out = pd.DataFrame((U_hub_mean.to_series(),
                        U_hat_mean.to_series(), 
                        P_bar_mean.to_series(), 
                        P_bar_std.to_series(),
                        P_bar_max.to_series(),
                        P_bar_min.to_series(),                        
                        )).T
    out.columns = ['U_mean','U_mean_power_weighted','P_mean','P_std','P_max','P_min']
    out.index.name = 'U_bins'

    return out


def _average_velocity_bins(U, U_hub, bin_size):
    """
    Group time-ensembles into velocity bins based on hub-height 
    velocity and average.

    Args:
        U (xarray.DataArray): Input variable to group by velocity.
        U_hub (xarray.DataArray): Sea water velocity at hub height.
        bin_size (numeric): Velocity averaging window size in m/s.

    Returns:
        xarray.DataArray: Data grouped into velocity bins
    """

    # Reorganize into velocity bins and average
    U_bins = np.arange(0, np.nanmax(U_hub) + bin_size, bin_size)

    # Group time-ensembles into velocity bins based on hub-height velocity and average
    out = U.assign_coords({"time": U_hub}).rename({"time": "speed"})
    out = out.groupby_bins("speed", U_bins).mean()
    
    return out


def mean_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """
    Calculates profiles of the velocity means based on IEC/TS 62600-200 
    section 9.4. A mean is calculated for each `window_avg_time` and 
    bin-averaged based on ensemble velocity.

    Args:
        velocity (pandas.Series or xarray.DataArray ([range,] time)): 
        Streamwise sea water velocity or sea water speed.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        sampling_frequency (numeric): ADCP sampling frequency in Hz.
        window_avg_time (int, optional): Time averaging window in seconds. 
        Defaults to 600.

    Returns:
        pandas.DataFrame: Average velocity profiles based on ensemble mean
        velocity.
    """

    # Streamwise data
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Take velocity at hub height
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)
    # Average data into 5-10 minute ensembles
    U_bar = xr.DataArray(bnr.mean(abs(U).values),
                         coords={'range': U.range,
                                 'time': bnr.mean(U['time'].values)})

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_bar, mean_hub_vel, bin_size=0.5)

    return out.to_pandas()


def rms_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """
    Calculates profiles of the root-mean-square (RMS) of velocity based on 
    IEC/TS 62600-200 section 9.5. A RMS is calculated for each `window_avg_time` 
    and bin-averaged based on ensemble velocity.

    Args:
        velocity (pandas.Series or xarray.DataArray ([range,] time)): 
        Streamwise sea water velocity or sea water speed.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        sampling_frequency (numeric): ADCP sampling frequency in Hz.
        window_avg_time (int, optional): Time averaging window in seconds. 
        Defaults to 600.

    Returns:
        pandas.DataFrame: Root-mean-square velocity profiles based on RMS of
        raw velocity.
    """

    # Streamwise data
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Take velocity at hub height from velocity profile
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)

    # Reshape tidal velocity - returns (range, ensemble-time, ensemble elements)
    U_reshaped = bnr.reshape(abs(U).values)
    # Take root-mean-square
    U_rms = np.sqrt(np.nanmean(U_reshaped**2, axis=-1))

    U_rms = xr.DataArray(U_rms, 
                         coords={'range': U.range, 
                                 'time':bnr.mean(U['time'].values)})

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_rms, mean_hub_vel, bin_size=0.5)

    return out.to_pandas()


def std_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """
    Calculates profiles of the standard deviation of velocity based on 
    IEC/TS 62600-200 section 9.5. A standard deviation is calculated for 
    each `window_avg_time` and bin-averaged based on ensemble velocity.

    Args:
        velocity (pandas.Series or xarray.DataArray ([range,] time)): 
        Streamwise sea water velocity or sea water speed.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        sampling_frequency (numeric): ADCP sampling frequency in Hz.
        window_avg_time (int, optional): Time averaging window in seconds. 
        Defaults to 600.

    Returns:
        pandas.DataFrame: Standard deviation of velocity profiles based on
        ensemble standard deviation.
    """

    # Streamwise data
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Take velocity at hub height from velocity profile
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)
    # Standard deviation
    U_std = xr.DataArray(bnr.standard_deviation(U.values),
                         coords={'range': U.range,
                                 'time':bnr.mean(U['time'].values)})

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_std, mean_hub_vel, bin_size=0.5)

    return out.to_pandas()


def device_efficiency(power, 
                      velocity, 
                      water_density, 
                      capture_area, 
                      hub_height, 
                      sampling_frequency, 
                      window_avg_time=600):
    """
    Calculates marine energy device efficiency based on IEC/TS 62600-200
    Section 9.7.

    Args:
        power (pandas.Series or xarray.DataArray (time)): Device power 
        output timeseries in Watts.
        velocity (pandas.Series or xarray.DataArray ([range,] time)): 
        Streamwise sea water velocity or sea water speed in m/s.
        water_density (float, pandas.Series or xarray.DataArray): Sea 
        water density in kg/m^3.
        capture_area (numeric): Swept area of marine energy device.
        hub_height (numeric): Altitude above the seabed. Assumes ADCP depth 
        bins are referenced to the seafloor.
        sampling_frequency (numeric): ADCP sampling frequency in Hz.
        window_avg_time (int, optional): Time averaging window in seconds. 
        Defaults to 600.

    Returns:
        pandas.Series: Device efficiency (power coefficient) in percent.
    """

    # Assert velocity is 2D xarray or pandas and has dims range, time
    # Power should have a timestamp coordinate/index

    # Streamwise data
    U = velocity

    # Power
    # Interpolate to velocity timeseries
    if 'xarray' in type(power).__module__:
        power = power.interp(time=U.time)
    elif 'pandas' in type(power).__module__ and isinstance(power.index, pd.DatetimeIndex):
        power = power.to_xarray().interp(time=U.time)
    else:
        warnings.warn("Assuming `power` timestamps match `velocity` timestamps")

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Hub-height velocity
    mean_hub_vel = xr.DataArray(bnr.mean(U.sel(range=hub_height, method='nearest').values), 
                                coords={'time': bnr.mean(U['time'].values)})
    vel_hub = _average_velocity_bins(mean_hub_vel, mean_hub_vel, bin_size=0.1)

    # Water density
    if np.size(water_density) > 1:
        rho_avg = xr.DataArray(bnr.mean(water_density.values), 
                        coords={'time': bnr.mean(U['time'].values)})
        rho_vel = _average_velocity_bins(rho_avg, mean_hub_vel, bin_size=0.1)
    else:
        rho_vel = water_density

    # Bin average power
    P_avg = xr.DataArray(bnr.mean(power.values), 
                         coords={'time': bnr.mean(power['time'].values)})
    P_vel = _average_velocity_bins(P_avg, mean_hub_vel, bin_size=0.1)

    # Theoretical power resource
    P_resource = 1/2 * rho_vel * capture_area * vel_hub**3

    # Efficiency
    out = P_vel / P_resource

    return out.to_pandas()
