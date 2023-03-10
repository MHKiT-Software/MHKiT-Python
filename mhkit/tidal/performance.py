import numpy as np
import pandas as pd
import xarray as xr

from mhkit import dolfyn
from mhkit.river.performance import (circular, ducted, rectangular, 
                                     multiple_circular, tip_speed_ratio,
                                     power_coefficient)


def _slice_circular_capture_area(diameter, hub_height, Doppler_cell_size):
    """_summary_

    Args:
        diameter (_type_): _description_
        hub_height (_type_): _description_
        Doppler_cell_size (_type_): _description_

    Returns:
        _type_: _description_
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
    # d = 5  # m
    # hub_height = 4.2
    d = diameter
    cs = Doppler_cell_size

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


def power_curve(velocity, 
                power, 
                diameter, 
                hub_height, 
                Doppler_cell_size, 
                sampling_frequency, 
                window_avg_time=600):
    """_summary_
    IEC 9.3

    Args:
        velocity (_type_): _description_
        diameter (_type_): _description_
        hub_height (_type_): _description_
        Doppler_cell_size (_type_): _description_
        sampling_frequency (_type_): _description_
        window_avg_time (int, optional): _description_. Defaults to 600.

    Returns:
        _type_: _description_
    """
    # assert velocity is 2D xarray or pandas and has dims range, time
    dtype = type(velocity)

    A_slc = _slice_circular_capture_area(diameter, hub_height, Doppler_cell_size)

    # Fetch streamwise data
    #U = ds_streamwise['vel'].sel(dir='streamwise')
    U = abs(velocity)
    # Interpolate power to velocity timestamps
    P = power.interp(time=U['time'], method='linear')

    ## Power weighted velocity in capture area
    # Interpolate U range to capture area slices, then cube and multiply by area
    U_hat = U.interp(range=A_slc['range'], method='linear')**3 * A_slc
    # Average the velocity across the capture area and divide out area
    U_hat = (U_hat.mean('range') / A_slc.sum()) ** (-1/3)

    # Time-average velocity at hub-height
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)
    
    # Power weighted velocity mean
    time = U_hat['time'].values
    U_hat_bar = xr.DataArray((bnr.mean(U_hat.values ** 3)) ** (-1/3), 
                              coords={'time': bnr.mean(time)})
    
    # Average power
    P_bar = xr.DataArray(bnr.mean(P.values),
                         coords={'time': bnr.mean(time)})

    # Then reorganize into 0.1 m velocity bins and average
    U_bins = np.arange(0, mean_hub_vel.max() + 0.1, 0.1)
    U_hat_vel = U_hat_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})
    U_hat_mean = U_hat_vel.groupby_bins("speed", U_bins).mean()
    
    P_bar_vel = P_bar.assign_coords({"time": mean_hub_vel}).rename({"time": "speed"})   
    P_bar_mean = P_bar_vel.groupby_bins("speed", U_bins).mean()
    P_bar_std = P_bar_vel.groupby_bins("speed", U_bins).std()
    P_bar_max = P_bar_vel.groupby_bins("speed", U_bins).max()
    P_bar_min = P_bar_vel.groupby_bins("speed", U_bins).min()

    out = pd.DataFrame((U_hat_mean.to_series(), 
                        P_bar_mean.to_series(), 
                        P_bar_std.to_series(),
                        P_bar_max.to_series(),
                        P_bar_min.to_series(),                        
                        ))
    out.columns = ['U_mean_power_weighted','P_mean','P_std','P_max','P_min']
    out.index.name = 'U_mean'

    return out


def _average_velocity_bins(U, U_hub, bin_size):
    """_summary_

    Args:
        U (_type_): _description_
        U_hub (_type_): _description_
        bin_size (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Reorganize into velocity bins and average
    U_bins = np.arange(0, U.max() + bin_size, bin_size)

    # Group time-ensembles into velocity bins based on hub-height velocity and average
    out = U.assign_coords({"time": U_hub}).rename({"time": "speed"})
    out = out.groupby_bins("speed", U_bins).mean()
    
    return out


def mean_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """_summary_
    IEC 9.4

    Args:
        velocity (_type_): _description_
        hub_height (_type_): _description_
        sampling_frequency (_type_): _description_
        window_avg_time (int, optional): _description_. Defaults to 600.

    Returns:
        _type_: _description_
    """

    # Fetch streamwise data
    #U = ds_streamwise['vel'].sel(dir='streamwise')
    U = velocity

    # Average data into 5-10 minute ensembles
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    U_bar = xr.DataArray(bnr.mean(abs(U).values), 
                         coords={'range': U.range,
                                 'time': bnr.mean(U['time'].values)})

    # Take velocity at hub height
    mean_hub_vel = U_bar.sel(range=hub_height, method='nearest').values

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_bar, mean_hub_vel, bin_size=0.5)

    return out.to_pandas()


def rms_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """_summary_
    IEC 9.5

    Args:
        velocity (_type_): _description_
        hub_height (_type_): _description_
        sampling_frequency (_type_): _description_
        window_avg_time (int, optional): _description_. Defaults to 600.

    Returns:
        _type_: _description_
    """

    # Fetch streamwise data
    #U = ds_streamwise['vel'].sel(dir='streamwise')
    U = velocity

    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)

    ## Detrend tidal velocity - returns (range, ensemble-time, ensemble)
    U_detrend = bnr.detrend(abs(U).values) # vs demean
    # Unravel detrended array from (range, ensemble-time, ensemble) into (range, time)
    new_time_size = U['time'].size//bnr.n_bin * bnr.n_bin
    U_rms = np.empty((U['range'].size, new_time_size))
    for i in range(U['range'].size):
        U_rms[i] = np.ravel(U_detrend[i], 'C')
    # Ignoring datapoints at end of array that get chopped off from reshaping
    U_rms = xr.DataArray(U_rms, coords={'range': U.range, 'time':U.time[:new_time_size]})

    # Take velocity at hub height from velocity profile
    hub_vel = U.sel(range=hub_height, method='nearest').values[:new_time_size]

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_rms, hub_vel, bin_size=0.5)

    return out.to_pandas()


def std_velocity_profiles(velocity, hub_height, sampling_frequency, window_avg_time=600):
    """_summary_
    IEC 9.5

    Args:
        velocity (_type_): _description_
        hub_height (_type_): _description_
        sampling_frequency (_type_): _description_
        window_avg_time (int, optional): _description_. Defaults to 600.

    Returns:
        _type_: _description_
    """

    # Fetch streamwise data
    #U = ds_streamwise['vel'].sel(dir='streamwise')
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # standard deviation
    U_std = np.nanstd(bnr.reshape(U.values))

    # Take velocity at hub height from velocity profile
    mean_hub_vel = bnr.mean(U.sel(range=hub_height, method='nearest').values)

    # Then reorganize into 0.5 m/s velocity bins and average
    out = _average_velocity_bins(U_std, mean_hub_vel, bin_size=0.5)

    return out.to_pandas()


def efficiency(power, 
               velocity, 
               water_density, 
               capture_area, 
               hub_height, 
               sampling_frequency, 
               window_avg_time=600):
    """_summary_
    IEC 9.7

    Args:
        power (_type_): _description_
        velocity (_type_): _description_
        water_density (_type_): _description_
        capture_area (_type_): _description_
        hub_height (_type_): _description_
        sampling_frequency (_type_): _description_
        window_avg_time (int, optional): _description_. Defaults to 600.

    Returns:
        _type_: _description_
    """

    # Fetch streamwise data
    #U = ds_streamwise['vel'].sel(dir='streamwise')
    U = velocity

    # Create binner
    bnr = dolfyn.VelBinner(n_bin=window_avg_time*sampling_frequency, fs=sampling_frequency)
    # Hub-height velocity
    mean_hub_vel = xr.DataArray(bnr.mean(U.sel(range=hub_height, method='nearest').values), 
                                coords={'time': bnr.mean(U['time'].values)})
    # Water density
    # TODO water density if a single value
    rho = xr.DataArray(bnr.mean(water_density.values), 
                       coords={'time': bnr.mean(U['time'].values)})
    # Power
    P_average = xr.DataArray(bnr.mean(power.values), 
                            coords={'time': bnr.mean(power['time'].values)})

    # Group and average
    rho_vel = _average_velocity_bins(rho, mean_hub_vel, bin_size=0.1)
    vel = _average_velocity_bins(mean_hub_vel, mean_hub_vel, bin_size=0.1)

    # Theoretical power resource
    P_resource = 1/2 * rho_vel * capture_area * vel**3

    # Efficiency
    # TODO will need to interpolate average_power time to P_resource
    out = P_average / P_resource.to_pandas()

    return out
