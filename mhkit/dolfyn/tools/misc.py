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


def detrend_array(arr, axis=-1, in_place=False):
    """
    Remove a linear trend from arr.

    Parameters
    ----------
    arr : array_like
       The array from which to remove a linear trend.
    axis : int
       The axis along which to operate.

    Notes
    -----
    This method is copied from the matplotlib.mlab library, but
    implements the covariance calcs explicitly for added speed.

    This works much faster than mpl.mlab.detrend for multi-dimensional
    arrays, and is also faster than linalg.lstsq methods.
    """

    arr = np.asarray(arr)
    if not in_place:
        arr = arr.copy()
    sz = np.ones(arr.ndim, dtype=int)
    sz[axis] = arr.shape[axis]
    x = np.arange(sz[axis], dtype=np.float_).reshape(sz)
    x -= np.nanmean(x, axis=axis, keepdims=True)
    arr -= np.nanmean(arr, axis=axis, keepdims=True)
    b = np.nanmean((x * arr), axis=axis, keepdims=True) / np.nanmean(
        (x**2), axis=axis, keepdims=True
    )
    arr -= b * x
    return arr


def group(bl, min_length=0):
    """
    Find continuous segments in a boolean array.

    Parameters
    ----------
    bl : numpy.ndarray (dtype='bool')
      The input boolean array.
    min_length : int (optional)
      Specifies the minimum number of continuous points to consider a
      `group` (i.e. that will be returned).

    Returns
    -------
    out : np.ndarray(slices,)
      a vector of slice objects, which indicate the continuous
      sections where `bl` is True.

    Notes
    -----
    This function has funny behavior for single points.  It will
    return the same two indices for the beginning and end.
    """

    if not any(bl):
        return np.empty(0)
    vl = np.diff(bl.astype("int"))
    ups = np.nonzero(vl == 1)[0] + 1
    dns = np.nonzero(vl == -1)[0] + 1
    if bl[0]:
        if len(ups) == 0:
            ups = np.array([0])
        else:
            ups = np.concatenate((np.array([0]), [len(ups)]))
    if bl[-1]:
        if len(dns) == 0:
            dns = np.array([len(bl)])
        else:
            dns = np.concatenate((dns, [len(bl)]))
    out = np.empty(len(dns), dtype="O")
    idx = 0
    for u, d in zip(ups, dns):
        if d - u < min_length:
            continue
        out[idx] = slice(u, d)
        idx += 1
    return out[:idx]


def slice1d_along_axis(arr_shape, axis=0):
    """
    Return an iterator object for looping over 1-D slices, along ``axis``, of
    an array of shape arr_shape.

    Parameters
    ----------
    arr_shape : tuple,list
      Shape of the array over which the slices will be made.
    axis : integer
      Axis along which `arr` is sliced.

    Returns
    -------
    Iterator : object
      The iterator object returns slice objects which slices arrays of
      shape arr_shape into 1-D arrays.

    Examples
    --------
    >> out=np.empty(replace(arr.shape,0,1))
    >> for slc in slice1d_along_axis(arr.shape,axis=0):
    >>     out[slc]=my_1d_function(arr[slc])
    """

    nd = len(arr_shape)
    if axis < 0:
        axis += nd
    ind = [0] * (nd - 1)
    i = np.zeros(nd, "O")
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None)
    itr_dims = np.asarray(arr_shape).take(indlist)
    Ntot = np.prod(itr_dims)
    i.put(indlist, ind)
    k = 0
    while k < Ntot:
        # increment the index
        n = -1
        while (ind[n] >= itr_dims[n]) and (n > (1 - nd)):
            ind[n - 1] += 1
            ind[n] = 0
            n -= 1
        i.put(indlist, ind)
        yield tuple(i)
        ind[-1] += 1
        k += 1


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
      360 degrees. Default = True

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


def fillgaps(a, maxgap=np.inf, dim=0, extrapFlg=False):
    """
    Linearly fill NaN value in an array.

    Parameters
    ----------
    a : numpy.ndarray
      The array to be filled.
    maxgap : numpy.ndarray (optional: inf)
      The maximum gap to fill.
    dim : int (optional: 0)
      The dimension to operate along.
    extrapFlg : bool (optional: False)
      Whether to extrapolate if NaNs are found at the ends of the
      array.

    See Also
    --------
    mhkit.dolfyn.tools.misc._interpgaps : Linearly interpolates in time.

    Notes
    -----
    This function interpolates assuming spacing/timestep between
    successive points is constant. If the spacing is not constant, use
    _interpgaps.
    """

    # If this is a multi-dimensional array, operate along axis dim.
    if a.ndim > 1:
        for inds in slice1d_along_axis(a.shape, dim):
            fillgaps(a[inds], maxgap, 0, extrapFlg)
        return

    a = np.asarray(a)
    nd = a.ndim
    if dim < 0:
        dim += nd
    if dim >= nd:
        raise ValueError("dim must be less than a.ndim; dim=%d, rank=%d." % (dim, nd))
    ind = [0] * (nd - 1)
    i = np.zeros(nd, "O")
    indlist = list(range(nd))
    indlist.remove(dim)
    i[dim] = slice(None, None)
    i.put(indlist, ind)

    gd = np.nonzero(~np.isnan(a))[0]

    # Here we extrapolate the ends, if necessary:
    if extrapFlg and gd.__len__() > 0:
        if gd[0] != 0 and gd[0] <= maxgap:
            a[: gd[0]] = a[gd[0]]
        if gd[-1] != a.__len__() and (a.__len__() - (gd[-1] + 1)) <= maxgap:
            a[gd[-1] :] = a[gd[-1]]

    # Here is the main loop
    if gd.__len__() > 1:
        inds = np.nonzero((1 < np.diff(gd)) & (np.diff(gd) <= maxgap + 1))[0]
        for i2 in range(0, inds.__len__()):
            ii = list(range(gd[inds[i2]] + 1, gd[inds[i2] + 1]))
            a[ii] = (
                np.diff(a[gd[[inds[i2], inds[i2] + 1]]])
                * (np.arange(0, ii.__len__()) + 1)
                / (ii.__len__() + 1)
                + a[gd[inds[i2]]]
            ).astype(a.dtype)

    return a


def interpgaps(a, t, maxgap=np.inf, dim=0, extrapFlg=False):
    """
    Fill gaps (NaN values) in ``a`` by linear interpolation along
    dimension ``dim`` with the point spacing specified in ``t``.

    Parameters
    ----------
    a : numpy.ndarray
      The array containing NaN values to be filled.
    t : numpy.ndarray (len(t) == a.shape[dim])
      Independent variable of the points in ``a``, e.g. timestep
    maxgap : numpy.ndarray (optional: inf)
      The maximum gap to fill.
    dim : int (optional: 0)
      The dimension to operate along.
    extrapFlg : bool (optional: False)
      Whether to extrapolate if NaNs are found at the ends of the
      array.

    See Also
    --------
    mhkit.dolfyn.tools.misc.fillgaps : Linearly interpolates in array-index space.
    """

    # If this is a multi-dimensional array, operate along dim dim.
    if a.ndim > 1:
        for inds in slice1d_along_axis(a.shape, dim):
            interpgaps(a[inds], t, maxgap, 0, extrapFlg)
        return

    gd = _find(~np.isnan(a))

    # Here we extrapolate the ends, if necessary:
    if extrapFlg and gd.__len__() > 0:
        if gd[0] != 0 and gd[0] <= maxgap:
            a[: gd[0]] = a[gd[0]]
        if gd[-1] != a.__len__() and (a.__len__() - (gd[-1] + 1)) <= maxgap:
            a[gd[-1] :] = a[gd[-1]]

    # Here is the main loop
    if gd.__len__() > 1:
        inds = _find((1 < np.diff(gd)) & (np.diff(gd) <= maxgap + 1))
        for i2 in range(0, inds.__len__()):
            ii = np.arange(gd[inds[i2]] + 1, gd[inds[i2] + 1])
            ti = (t[ii] - t[gd[inds[i2]]]) / np.diff(
                t[[gd[inds[i2]], gd[inds[i2] + 1]]]
            )
            a[ii] = (
                np.diff(a[gd[[inds[i2], inds[i2] + 1]]]) * ti + a[gd[inds[i2]]]
            ).astype(a.dtype)

    return a


def medfiltnan(a, kernel, thresh=0):
    """
    Do a running median filter of the data. Regions where more than
    ``thresh`` fraction of the points are NaN are set to NaN.

    Parameters
    ----------
    a : numpy.ndarray
      2D array containing data to be filtered.
    kernel_size : numpy.ndarray or list, optional
      A scalar or a list of length 2, giving the size of the median
      filter window in each dimension. Elements of kernel_size should
      be odd. If kernel_size is a scalar, then this scalar is used as
      the size in each dimension.
    thresh : int
      Maximum gap in *a* to filter over

    Returns
    -------
    out : numpy.ndarray
      2D array of same size containing filtered data

    See Also
    --------
    scipy.signal.medfilt2d
    """

    flag_1D = False
    if a.ndim == 1:
        a = a[None, :]
        flag_1D = True
    try:
        len(kernel)
    except:
        kernel = [1, kernel]
    out = medfilt2d(a, kernel)
    if thresh > 0:
        out[
            convolve2d(np.isnan(a), np.ones(kernel) / np.prod(kernel), "same") > thresh
        ] = np.NaN
    if flag_1D:
        return out[0]
    return out
