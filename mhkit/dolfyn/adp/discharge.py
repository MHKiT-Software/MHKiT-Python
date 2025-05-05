import numpy as np
import xarray as xr
import cartopy.crs as ccrs


def discharge(ds, water_depth, rho, mu=None, surface_offset=0, utm_zone=10):
    """Calculate discharge (volume flux), power (kinetic energy flux),
    power density, and Reynolds number from a dataset containing a
    boat survey with a down-looking ADCP. This function is built to
    natively handle ADCP datasets read in using the `dolfyn` module.

    Dataset velocity should already be corrected using ADCP-measured
    bottom track or GPS-measured velocity. The first velocity direction
    is assumed to be the primary flow axis.

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
        Dynamic visocity based on water temperature and salinity, in Ns/m^2.
        If not provided, Reynolds Number will not be calculated.
        Default: None.
    surface_offset: float
        Surface level offset due to changes in tidal level, in meters.
        Positive is down. Default: 0 m.
    utm_zone: int
        UTM zone for coordinate transformations (e.g., to compute cross-sectional
        distances from GPS lat/lon data). Map of UTM zones for the contiguous US:
        https://www.usgs.gov/media/images/mapping-utm-grid-conterminous-48-united-states.
        Default: 10 (the US west coast).

    Returns
    -------
    out: xarray.Dataset
        Dataset containing the following variables:
         - `discharge`: (1) volume flux, in m^3/s
         - `power`: (1) power, in W
         - `power_density`: (1) power density, in W/m^2
         - `reynolds_number`: (1) Reynolds number, unitless
    """

    def _extrapolate_to_bottom(vel, bottom, rng):
        """
        Linearly extrapolate velocity values from the deepest valid bin down to zero at the seafloor.

        This function sets velocity to zero at the seafloor and linearly interpolates
        between the last valid velocity bin and this zero-velocity boundary. If no valid
        velocity is found in a particular profile, no update is performed for that profile.
        This function assumes `rng` extends at least to (or below) the deepest seafloor depth
        specified in `bottom`.

        Parameters
        ----------
        vel : numpy.ndarray
            A velocity array of shape (dir, range, time), typically containing:
                - `dir` : velocity component dimension (e.g., 2 or 3 for 2D or 3D flow).
                - `range` : vertical/bin dimension (positive downward).
                - `time` : time dimension corresponding to each profile.
            The array is modified in-place (the updated values are also returned).
        bottom : array-like
            Array of length equal to the time dimension in `vel`, specifying the seafloor
            depth (in the same coordinate system as `rng`) at each time step.
        rng : array-like
            The vertical/bin positions corresponding to `vel` along the `range` dimension,
            sorted in ascending order (e.g., depth from the water surface downward).

        Returns
        -------
        vel : numpy.ndarray
            The same array passed in, with updated values below the last valid velocity bin
            for each time step (linear extrapolation to zero at the seafloor).
        """

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

    def _convert_latlon_to_utm(ds, proj):
        """
        Convert latitude/longitude coordinates to UTM coordinates.

        This function uses the Cartopy `transform_point` and `transform_points` methods to
        project GPS latitude/longitude data into the specified UTM coordinate reference
        system. The resulting (x, y) coordinates are stored in an xarray DataArray that is
        interpolated onto the main time axis of `ds`.

        The function sets `proj.x0` and `proj.y0` to the UTM coordinates of the mean
        longitude and latitude from `ds`. This can be used as a reference origin.
        Missing or NaN lat/lon values are handled via interpolation and extrapolation
        onto the `ds["time"]` axis.
        This function modifies the `proj` object by adding `x0` and `y0` attributes,
        which may be used for subsequent coordinate transformations or offsets.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset that must contain at least the following variables:
            - "latitude_gps"  : (time_gps) latitude values in degrees North.
            - "longitude_gps" : (time_gps) longitude values in degrees East.
            - "time"          : time axis onto which the projected coordinates will be
                                interpolated.
        proj : cartopy.crs.Projection
            A Cartopy UTM projection or similar projection object. This is used both to
            store the reference origin (`x0`, `y0`) and to transform lat/lon coordinates
            into UTM.

        Returns
        -------
        xy : xarray.DataArray
            A DataArray of shape (gps=2, time), where:
            - The first dimension (indexed by "gps") corresponds to ["x", "y"] UTM
                coordinates.
            - The second dimension ("time") matches `ds["time"]`.
            The returned coordinates are interpolated in time using `ds["longitude_gps"]`
            and `ds["latitude_gps"]`, with values extrapolated if necessary.

        """

        plate_c = ccrs.PlateCarree()
        proj.x0, proj.y0 = proj.transform_point(
            ds["longitude_gps"].mean(), ds["latitude_gps"].mean(), plate_c
        )
        xy = xr.DataArray(
            proj.transform_points(plate_c, ds["longitude_gps"], ds["latitude_gps"])[
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
        """
        Compute the planar distance from the projection's reference origin.

        Parameters
        ----------
        proj : cartopy.crs.Projection
            A projection object with attributes `x0` and `y0`, which define the
            reference origin in the projected coordinate system.
        x : float or array-like
            One or more x-coordinates in the same units (m) as `proj.x0`.
        y : float or array-like
            One or more y-coordinates in the same units (m) as `proj.y0`.

        Returns
        -------
        dist : float or numpy.ndarray
            The distance(s) in m from the point(s) `(x, y)` to `(proj.x0, proj.y0)`.
            If `x` and `y` are arrays, the output is an array of the same shape.
        """

        return np.sqrt((proj.x0 - x) ** 2 + (proj.y0 - y) ** 2)

    def _calc_discharge(vel, x, depth, surface_zoff=None):
        """
        Calculate the integrated flux (e.g., discharge) by double integration of velocity
        over the cross-sectional area: depth and lateral distance.

        Missing (NaN) velocities are treated as zero.
        Ensure `depth` and `surface_zoff` are both positive downward.

        Parameters
        ----------
        vel : numpy.ndarray or xarray.DataArray
            A 2D array of shape (nz, nx) corresponding to velocity values (m/s).
            - `nz` is the number of vertical bins (downward).
            - `nx` is the number of horizontal points.
        x : array-like
            Horizontal positions (m) of length `nx`. If `x` is in descending order
            (i.e., `x[0] > x[-1]`), the resulting flux is assigned a negative sign to
            indicate reverse orientation.
        depth : array-like
            Vertical positions (m) of length `nz`, positive downward. This is used
            for integration along the vertical dimension.
        surface_zoff : float, optional
            Surface level offset due to changes in tidal level, in meters.
            Positive is down.

        Returns
        -------
        Q : float
            The integrated flux (e.g., discharge) in units of m^3/s

        """
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
    vel.values = _extrapolate_to_bottom(
        ds["vel"].values, water_depth, ds["range"].values
    )
    vel_x = vel[0]
    # Get position at each timestep in UTM grid
    proj = ccrs.UTM(utm_zone)
    xy = _convert_latlon_to_utm(ds, proj)
    # Distance from UTM grid origin (mean of GPS points)
    _x = _distance(proj, xy[0], xy[1])
    # Set distance range for entire transect
    q_x_range = [_x.min(), _x.max()]  # meters

    # Calculate discharge, power, kinetic energy, and reynolds number
    _xinds = (q_x_range[0] < _x) & (_x < q_x_range[1])
    out = {}
    if _xinds.any():
        speed = vel_x[:, _xinds]  # m/s
        # Volume Flux, aka Discharge
        out["Q"] = _calc_discharge(
            speed, xy[0][_xinds], ds["range"], surface_offset
        )  # m/s * m * m = m^3/s
        # Kinetic Energy Flux, aka Power
        out["P"] = (
            0.5
            * rho
            * _calc_discharge(speed**3, xy[0][_xinds], ds["range"], surface_offset)
        )  # kg/m^3 * m^3/s^3 * m * m = kg*m^2/s = W
        # Power Density
        out["J"] = (
            (0.5 * rho * speed**3).mean().item()
        )  # kg/m^3 * m^3/s^3 = kg/s^3 = W/m^2
        hydraulic_depth = abs(
            np.trapz((water_depth - surface_offset)[_xinds], xy[0][_xinds])
        ) / (
            xy[0][_xinds].max() - xy[0][_xinds].min()
        )  # area / surface-width
        # Reynolds Number
        out["Re"] = ((rho * ds.velds.U_mag.mean() * hydraulic_depth) / mu).item()
    else:
        out["Q"] = np.nan
        out["P"] = np.nan
        out["J"] = np.nan
        out["Re"] = np.nan

    ds["discharge"] = xr.DataArray(
        np.float32(out["Q"]),
        dims=[],
        attrs={
            "units": "m3 s-1",
            "long_name": "Discharge",
        },
    )
    ds["power"] = xr.DataArray(
        np.float32(out["P"]),
        dims=[],
        attrs={
            "units": "W",
            "long_name": "Power",
        },
    )
    ds["power_density"] = xr.DataArray(
        np.float32(out["J"]),
        dims=[],
        attrs={
            "units": "W m-2",
            "long_name": "Power Density",
        },
    )
    ds["reynolds_number"] = xr.DataArray(
        np.float32(out["Re"]),
        dims=[],
        attrs={
            "units": "1",
            "long_name": "Reynolds Number",
        },
    )

    return ds
