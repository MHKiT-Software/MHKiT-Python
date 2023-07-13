import xarray as xr


def lay_length(ds, depth, tolerance=0.25):
    """
    Calculate the laylength of a mooring line over time.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing x,y,z nodes (ie Node1px, Node1py, Node1pz)
    depth: float
        Depth of seabed (m)
    tolerance: float, optional
        Tolerance to detect first lift point from seabed, by default 0.25 meters

    Returns
    -------
    line_lay_length: xr.Dataset
        Array containing the laylength at each time step

    Raises
    ------
    ValueError
        Checks for mininum number of nodes necessary to calculate laylength
    TypeError
        Checks for correct input types for ds, depth, and tolerance
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError('ds must be of type xr.Dataset')
    if not isinstance(depth, (float, int)):
        raise TypeError('depth must be of type float or int')
    if not isinstance(tolerance, (float, int)):
        raise TypeError('tolerance must be of type float or int')

    # get channel names
    chans = list(ds.keys())
    nodes_x = [x for x in chans if 'x' in x]
    nodes_y = [y for y in chans if 'y' in y]
    nodes_z = [z for z in chans if 'z' in z]

    if len(nodes_z) < 3:
        raise ValueError(
            'This function requires at least 3 nodes to calculate laylength')
    # find name of first z point where tolerance is exceeded
    laypoint = ds[nodes_z].where(ds[nodes_z] > depth+abs(tolerance))
    laypoint = laypoint.to_dataframe().dropna(axis=1).columns[0]
    # get previous z-point
    lay_indx = nodes_z.index(laypoint) - 1
    lay_z = nodes_z[lay_indx]
    # get corresponding x-point and y-point node names
    lay_x = lay_z[:-1] + 'x'
    lay_y = lay_z[:-1] + 'y'
    lay_0x = nodes_x[0]
    lay_0y = nodes_y[0]
    # find distance between initial point and lay point
    laylength_x = ds[lay_x] - ds[lay_0x]
    laylength_y = ds[lay_y] - ds[lay_0y]
    line_lay_length = (laylength_x**2 + laylength_y**2) ** (1/2)

    return line_lay_length
