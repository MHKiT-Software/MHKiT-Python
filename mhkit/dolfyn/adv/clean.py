"""Module containing functions to clean data
"""
import numpy as np
import warnings
from ..tools.misc import group, slice1d_along_axis
warnings.filterwarnings('ignore', category=np.RankWarning)

sin = np.sin
cos = np.cos


def clean_fill(u, mask, npt=12, method='cubic', max_gap=None):
    """
    Interpolate over mask values in timeseries data using the specified method

    Parameters
    ----------
    u : xarray.DataArray
      The dataArray to clean.
    mask : bool
      Logical tensor of elements to "nan" out (from `spikeThresh`, `rangeLimit`,
      or `GN2002`) and replace
    npt : int
      The number of points on either side of the bad values that 
      interpolation occurs over
    method : string
      Interpolation scheme to use (linear, cubic, pchip, etc)
    max_gap : int
      Max number of consective nan's to interpolate across, must be <= npt/2

    Returns
    -------
    da : xarray.DataArray
      The dataArray with nan's filled in

    See Also
    --------
    xarray.DataArray.interpolate_na()

    """
    if max_gap:
        assert max_gap <= npt, 'Max_gap must be less than half of npt'

    # Apply mask
    u.values[..., mask] = np.nan

    # Remove bad data for 2D+ and 1D timeseries variables
    if 'dir' in u.dims:
        for i in range(u.shape[0]):
            u[i] = _interp_nan(u[i], npt, method, max_gap)
    else:
        u = _interp_nan(u, npt, method, max_gap)

    return u


def _interp_nan(da, npt, method, max_gap):
    """
    Interpolate over the points in `bad` that are True.

    Parameters
    ----------
    da : xarray.DataArray
      The field to be cleaned
    npt : int
      The number of points on either side of the gap that the fit
      occurs over

    """
    searching = True
    bds = da.isnull().values
    ntail = 0
    pos = 0
    # The index array:
    i = np.arange(len(da), dtype=np.uint32)

    while pos < len(da):
        if searching:
            # Check the point
            if bds[pos]:
                # If it's bad, mark the start
                start = max(pos - npt, 0)
                # And stop searching.
                searching = False
            pos += 1
            # Continue...
        else:
            # Another bad point?
            if bds[pos]:  # Yes
                # Reset ntail
                ntail = 0
            else:  # No
                # Add to the tail of the block.
                ntail += 1
            pos += 1

            if (ntail == npt or pos == len(da)):
                # This is the block we are interpolating over
                i_int = i[start:pos]
                da[i_int] = da[i_int].interpolate_na(dim='time',
                                                     method=method,
                                                     use_coordinate=True,
                                                     max_gap=max_gap)
                # Reset
                searching = True
                ntail = 0
    return da


def spike_thresh(u, thresh=10):
    """
    Returns a logical vector where a spike in `u` of magnitude greater than
    `thresh` occurs. Both 'Negative' and 'positive' spikes are found.

    Parameters
    ----------
    u : xarray.DataArray
      The timeseries data to clean.
    thresh : int
       Magnitude of velocity spike, must be positive.

    Returns
    -------
    mask : |np.ndarray|
      Logical vector with spikes labeled as 'True'

    """
    du = np.diff(u.values, prepend=0)
    bds1 = ((du > thresh) & (du < -thresh))
    bds2 = ((du < -thresh) & (du > thresh))

    return bds1 + bds2


def range_limit(u, range=[-5, 5]):
    """
    Returns a logical vector that is True where the values of `u` are 
    outside of `range`.

    Parameters
    ----------
    u : xarray.DataArray
      The timeseries data to clean.
    range : list
       Min and max magnitudes beyond which are masked

    Returns
    -------
    mask : |np.ndarray|
      Logical vector with spikes labeled as 'True'

    """
    return ~((range[0] < u.values) & (u.values < range[1]))


def _calcab(al, Lu_std_u, Lu_std_d2u):
    """Solve equations 10 and 11 of Goring+Nikora2002
    """
    return tuple(np.linalg.solve(
        np.array([[cos(al) ** 2, sin(al) ** 2],
                  [sin(al) ** 2, cos(al) ** 2]]),
        np.array([(Lu_std_u) ** 2, (Lu_std_d2u) ** 2])))


def _phaseSpaceThresh(u):
    if u.ndim == 1:
        u = u[:, None]
    u = np.array(u)
    Lu = (2 * np.log(u.shape[0])) ** 0.5
    u = u - u.mean(0)
    du = np.zeros_like(u)
    d2u = np.zeros_like(u)
    # Take the centered difference.
    du[1:-1] = (u[2:] - u[:-2]) / 2
    # And again.
    d2u[2:-2] = (du[1:-1][2:] - du[1:-1][:-2]) / 2
    p = (u ** 2 + du ** 2 + d2u ** 2)
    std_u = np.std(u, axis=0)
    std_du = np.std(du, axis=0)
    std_d2u = np.std(d2u, axis=0)
    alpha = np.arctan2(np.sum(u * d2u, axis=0), np.sum(u ** 2, axis=0))
    a = np.empty_like(alpha)
    b = np.empty_like(alpha)
    with warnings.catch_warnings() as w:
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in ')
        for idx, al in enumerate(alpha):
            a[idx], b[idx] = _calcab(al, Lu * std_u[idx], Lu * std_d2u[idx])
            if np.any(np.isnan(a)) or np.any(np.isnan(a[idx])):
                print('Coefficient calculation error')
        theta = np.arctan2(du, u)
        phi = np.arctan2((du ** 2 + u ** 2) ** 0.5, d2u)
        pe = (((sin(phi) * cos(theta) * cos(alpha) +
                cos(phi) * sin(alpha)) ** 2) / a +
              ((sin(phi) * cos(theta) * sin(alpha) -
                cos(phi) * cos(alpha)) ** 2) / b +
              ((sin(phi) * sin(theta)) ** 2) / (Lu * std_du) ** 2) ** -1
    pe[:, np.isnan(pe[0, :])] = 0
    return (p > pe).flatten('F')


def GN2002(u, npt=5000):
    """
    The Goring & Nikora 2002 'despiking' method, with Wahl2003 correction.
    Returns a logical vector that is true where spikes are identified.

    Parameters
    ----------
    u : xarray.DataArray
      The velocity array (1D or 3D) to clean.
    npt : int
      The number of points over which to perform the method.

    Returns
    -------
    mask : |np.ndarray|
      Logical vector with spikes labeled as 'True'

    """
    if not isinstance(u, np.ndarray):
        return GN2002(u.values, npt=npt)

    if u.ndim > 1:
        mask = np.zeros(u.shape, dtype='bool')
        for slc in slice1d_along_axis(u.shape, -1):
            mask[slc] = GN2002(u[slc], npt=npt)
        return mask

    mask = np.zeros(len(u), dtype='bool')

    # Find large bad segments (>npt/10):
    # group returns a vector of slice objects.
    bad_segs = group(np.isnan(u), min_length=int(npt//10))
    if bad_segs.size > 0:
        # Break them up into separate regions:
        sp = 0
        ep = len(u)

        # Skip start and end bad_segs:
        if bad_segs[0].start == sp:
            sp = bad_segs[0].stop
            bad_segs = bad_segs[1:]
        if bad_segs[-1].stop == ep:
            ep = bad_segs[-1].start
            bad_segs = bad_segs[:-1]

        for ind in range(len(bad_segs)):
            bs = bad_segs[ind]  # bs is a slice object.
            # Clean the good region:
            mask[sp:bs.start] = GN2002(u[sp:bs.start], npt=npt)
            sp = bs.stop
        # Clean the last good region.
        mask[sp:ep] = GN2002(u[sp:ep], npt=npt)
        return mask

    c = 0
    ntot = len(u)
    nbins = int(ntot // npt)
    mask_last = np.zeros_like(mask) + np.inf
    mask[0] = True  # make sure we start.
    while mask.any():
        mask[:nbins * npt] = _phaseSpaceThresh(
            np.array(np.reshape(u[:(nbins * npt)], (npt, nbins), order='F')))
        mask[-npt:] = _phaseSpaceThresh(u[-npt:])
        c += 1
        if c >= 100:
            raise Exception('GN2002 loop-limit exceeded.')
        if mask.sum() >= mask_last.sum():
            break
        mask_last = mask.copy()
    return mask
