import numpy as np
import xarray as xr
import cartopy.crs as ccrs


def discharge(ds, water_depth, rho, mu=None, surface_offset=0, utm_zone=10):
    """Calculate discharge (volume flux), power (kinetic energy flux),
    kinetic energy, and Reynolds number from a dataset containing a
    boat survey with a down-looking ADCP. This function is built to
    natively handle ADCP datasets read in using the `dolfyn` module.

    Dataset velocity should already be corrected using ADCP-measured
    bottom track or GPS-measured velocity.
    This function linearly interpolates the lowest ADCP depth bin to
    the seafloor, and applies a constant extrapolation from the first
    ADCP bin to the surface.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing the following variables:
         - `vel`: (dir, range, time) motion-corrected velocity, in m/s
         - `latitude_gps`: (time_gps) latitude measured by GPS, in deg N
         - `longitude_gps`: (time_gps) longitude measured by GPS, in deg E
    water_depth: xarray.DataArray
        Total water depth measured by the ADCP or other input, in
        meters. If measured by the ADCP, add the ADCP's depth below
        the surface to this array.
        The "down" direction should be positive.
    rho: float
        Water density in kg/m^3
    mu: float
        Kinematic visocity based on water temperature and salinity, in Ns/m^2.
        Required for Reynolds Number.
    surface_offset: float
        Surface level offset due to changes in tidal level, in meters.
        Default: 0 m.
    utm_zone: int
        UTM zone that measurements were acquired in. Map of UTM zones for the
        contiguous US:
        https://www.usgs.gov/media/images/mapping-utm-grid-conterminous-48-united-states.
        Default: 10 (the US west coast).

    Returns
    -------
    out: dict(str, float)
        Dictionary containing computed parameters
    """

    def extrap2bottom(vel, bottom, rng):
        for idx in range(vel.shape[-1]):
            z_bot = bottom[idx]
            # Fetch lowest range index
            ind_bot = np.nonzero(rng > z_bot)[0][0]
            for idim in range(vel.shape[0]):
                vnow = vel[idim, :, idx]
                # Check that data exists in slice
                gd = np.isfinite(vnow) & (vnow != 0)
                if not gd.sum():
                    continue
                else:
                    ind = np.nonzero(gd)[0][-1]
                z_top = rng[ind]
                # linearly interpolate next lowest range bin based on 0 m/s at bottom
                vals = np.interp(rng[ind:ind_bot], [z_top, z_bot], [vnow[ind], 0])
                vel[idim, ind:ind_bot, idx] = vals

        return vel

    def latlon2utm(ds, proj):
        PlateC = ccrs.PlateCarree()
        proj.x0, proj.y0 = proj.transform_point(
            ds["longitude_gps"].mean(), ds["latitude_gps"].mean(), PlateC
        )
        xy = xr.DataArray(
            proj.transform_points(PlateC, ds["longitude_gps"], ds["latitude_gps"])[
                :, :2
            ].T,
            coords={"gps": ["x", "y"], "time_gps": ds["longitude_gps"]["time_gps"]},
        )

        # this seems to work for missing latlon
        xy = xy.interp(
            time_gps=ds["time"], kwargs={"fill_value": "extrapolate"}
        ).drop_vars("time_gps")
        return xy

    def _distance(proj, x, y):
        # GPS distance traveled in meters
        return np.sqrt((proj.x0 - x) ** 2 + (proj.y0 - y) ** 2)

    def calc_Q(vel, x, depth, surface_zoff=None):
        # depth and surface_zoff should be positive in down direction
        vel = vel.copy()
        vel = vel.fillna(0)
        if surface_zoff is not None:
            # Add a copy of the top row of data
            vel = np.vstack((vel[0], vel))
            depth = np.hstack((surface_zoff, depth))
        if x[0] > x[-1]:
            sign = -1
        else:
            sign = 1
        return sign * np.trapz(np.trapz(vel, depth, axis=0), x)

    # Extrapolate to bed
    vel = ds["vel"].copy()
    vel.values = extrap2bottom(ds["vel"].values, water_depth, ds["range"].values)
    vel_x = vel[0]
    # Get position at each timestep in UTM grid
    proj = ccrs.UTM(utm_zone)
    xy = latlon2utm(ds, proj)
    # Distance from UTM grid origin (mean of GPS points)
    _x = _distance(proj, xy[0], xy[1])
    # Set distance range for entire transect
    Q_x_range = [_x.min(), _x.max()]  # meters

    # Calculate discharge, power, kinetic energy, and reynolds number
    _xinds = (Q_x_range[0] < _x) & (_x < Q_x_range[1])
    out = {}
    if _xinds.any():
        U = vel_x[:, _xinds]  # m/s
        # Volume Flux, aka Discharge
        out["Q"] = calc_Q(
            U, xy[0][_xinds], ds["range"], surface_offset
        )  # m/s * m * m = m^3/s
        # Kinetic Energy Flux, aka Power
        out["P"] = (
            0.5 * rho * calc_Q(U**3, xy[0][_xinds], ds["range"], surface_offset)
        )  # kg/m^3 * m^3/s^3 * m * m = kg*m^2/s = W
        # Power Density
        out["J"] = (0.5 * rho * U**3).mean().item()  # kg/m^3 * m^3/s^3 = kg/s^3 = W/m^2
        # Hydraulic Depth
        L = abs(np.trapz((water_depth - surface_offset)[_xinds], xy[0][_xinds])) / (
            xy[0][_xinds].max() - xy[0][_xinds].min()
        )  # area / surface-width
        # Reynolds Number
        out["Re"] = ((rho * ds.velds.U_mag.mean() * L) / mu).item()
    else:
        out["Q"] = np.nan
        out["P"] = np.nan
        out["J"] = np.nan
        out["Re"] = np.nan

    return out
