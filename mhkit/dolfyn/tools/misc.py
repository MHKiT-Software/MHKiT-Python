import numpy as np
from scipy.signal import medfilt2d, convolve2d


def _nans(*args, **kwargs):
    out = np.empty(*args, **kwargs)
    if np.issubdtype(out.flatten()[0], np.integer):
        out[:] = 0
        return out
    else:
        out[:] = np.nan
        return out


def _nans_like(*args, **kwargs):
    out = np.empty_like(*args, **kwargs)
    out[:] = np.nan
    return out


def _find(arr):
    return np.nonzero(np.ravel(arr))[0]


def fillgaps(a, maxgap=np.inf, dim=0, extrapFlg=False):
    """
    Linearly fill NaN value in an array.

    Parameters
    ----------
    a : |np.ndarray|
      The array to be filled.

    maxgap : |np.ndarray| (optional: inf)
      The maximum gap to fill.

    dim : int (optional: 0)
      The dimension to operate along.

    extrapFlg : bool (optional: False)
      Whether to extrapolate if NaNs are found at the ends of the
      array.

    See Also
    =====

    interpgaps : Linearly interpolates in time.

    Notes
    =====

    This function interpolates assuming spacing/timestep between
    successive points is constant. If the spacing is not constant, use
    interpgaps.

    """
    a = np.asarray(a)
    nd = a.ndim
    if dim < 0:
        dim += nd
    if (dim >= nd):
        raise ValueError("dim must be less than a.ndim; dim=%d, rank=%d."
                          % (dim, nd))
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(dim)
    i[dim] = slice(None, None)
    i.put(indlist, ind)
    # k = 0

    gd = np.nonzero(~np.isnan(a))[0]

    # Here we extrapolate the ends, if necessary:
    if extrapFlg and gd.__len__() > 0:
        if gd[0] != 0 and gd[0] <= maxgap:
            a[:gd[0]] = a[gd[0]]
        if gd[-1] != a.__len__() and (a.__len__() - (gd[-1] + 1)) <= maxgap:
            a[gd[-1]:] = a[gd[-1]]

    # Here is the main loop
    if gd.__len__() > 1:
        inds = np.nonzero((1 < np.diff(gd)) & (np.diff(gd) <= maxgap + 1))[0]
        for i2 in range(0, inds.__len__()):
            ii = list(range(gd[inds[i2]] + 1, gd[inds[i2] + 1]))
            a[ii] = (np.diff(a[gd[[inds[i2], inds[i2] + 1]]]) *
                      (np.arange(0, ii.__len__()) + 1) /
                      (ii.__len__() + 1) + a[gd[inds[i2]]]).astype(a.dtype)


def interpgaps(a, t, maxgap=np.inf, dim=0, extrapFlg=False):
    """
    Fill gaps (NaN values) in ``a`` by linear interpolation along
    dimension DIM with the point spacing specified in ``t``.

    Parameters
    ==========
    a : |np.ndarray|
      The array containing NaN values to be filled.

    t : |np.ndarray| (len(t) == a.shape[dim])
      The grid of the points in ``a``.

    maxgap : |np.ndarray| (optional: inf)
      The maximum gap to fill.

    dim : int (optional: 0)
      The dimension to operate along.

    extrapFlg : bool (optional: False)
      Whether to extrapolate if NaNs are found at the ends of the
      array.

    See Also
    =====

    fillgaps : Linearly interpolates in array-index space.

    """
    gd = _find(~np.isnan(a))
    # Here is the main loop
    if gd.__len__() > 1:
        inds = _find((1 < np.diff(gd)) &
                    (np.diff(gd) <= maxgap + 1))
        for i2 in range(0, inds.__len__()):
            ii = np.arange(gd[inds[i2]] + 1, gd[inds[i2] + 1])
            ti = (t[ii] - t[gd[inds[i2]]]) / np.diff(t[[gd[inds[i2]],
                                                        gd[inds[i2] + 1]]])
            a[ii] = (np.diff(a[gd[[inds[i2], inds[i2] + 1]]]) * ti +
                     a[gd[inds[i2]]]).astype(a.dtype)

    
def convert_degrees(deg, tidal_mode=True):
    """
    Converts between the 'cartesian angle' (counter-clockwise from East) and
    the 'polar angle' in (degrees clockwise from North)
    
    Parameters
    ----------
    deg: float or array-like
      Number or array in 'degrees CCW from East' or 'degrees CW from North'
    tidal_mode : bool
      If true, range is set from 0 to +/-180 degrees. If false, range is 0 to 
      360 degrees
      
    Returns
    -------
    out : float or array-like
      Input data transformed to 'degrees CW from North' or 
      'degrees CCW from East', respectively (based on `deg`)
      
    Notes
    -----
    The same algorithm is used to convert back and forth between 'CCW from E' 
    and 'CW from N'
    
    """
    out = -(deg - 90) % 360
    if tidal_mode:
        out[out > 180] -= 360
    return out
