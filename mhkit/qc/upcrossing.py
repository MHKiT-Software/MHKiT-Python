import numpy as np


def _apply(t, data, f, inds):
    if inds is None:
        inds = upcrossing(t, data)

    n = inds.size - 1

    vals = np.empty(n)
    for i in range(n):
        vals[i] = f(inds[i], inds[i+1])

    return vals


def upcrossing(t, data):
    """Find the zero upcrossing points.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time series.  

    Returns
    -------
    inds: np.array
        Zero crossing indicies
    """
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(t, np.ndarray), 'data must be of type np.ndarray'
    assert len(data.shape) == 1, 'only 1D data supported, try calling squeeze()'

    # eliminate zeros
    zeroMask = (data == 0)
    data[zeroMask] = 0.5 * np.min(np.abs(data))
    # zero up-crossings
    diff = np.diff(np.sign(data))
    zeroUpCrossings_mask = (diff == 2) | (diff == 1)
    zeroUpCrossings_index = np.where(zeroUpCrossings_mask)[0]

    return zeroUpCrossings_index


def peaks(t, data, inds=None):
    """Find the peaks between zero crossings.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time-series.
    inds: np.array
        Optional indicies for the upcrossing. Useful
        when using several of the upcrossing methods
        to avoid repeating the upcrossing analysis 
        each time.

    Returns
    -------
    peaks: np.array
        Peak values of the time-series

    """
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    return _apply(t, data, lambda ind1, ind2: np.max(data[ind1:ind2]), inds)


def troughs(t, data, inds=None):
    """Find the troughs between zero crossings.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time-series.
    inds: np.array
        Optional indicies for the upcrossing. Useful
        when using several of the upcrossing methods
        to avoid repeating the upcrossing analysis 
        each time.

    Returns
    -------
    troughs: np.array
        Trough values of the time-series

    """
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    return _apply(t, data, lambda ind1, ind2: np.min(data[ind1:ind2]), inds)


def heights(t, data, inds=None):
    """Find the heights between zero crossings.

    The height is defined as the max value - min value
    between the zero crossing points.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time-series.
    inds: np.array
        Optional indicies for the upcrossing. Useful
        when using several of the upcrossing methods
        to avoid repeating the upcrossing analysis 
        each time.

    Returns
    -------
    heights: np.array
        Height values of the time-series
    """
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    def func(ind1, ind2): return np.max(
        data[ind1:ind2]) - np.min(data[ind1:ind2])
    return _apply(t, data, func, inds)


def periods(t, data, inds=None):
    """Find the period between zero crossings.

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time-series.
    inds: np.array
        Optional indicies for the upcrossing. Useful
        when using several of the upcrossing methods
        to avoid repeating the upcrossing analysis 
        each time.

    Returns
    -------
    periods: np.array
        Period values of the time-series
    """
    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'

    return _apply(t, data, lambda ind1, ind2: t[ind2] - t[ind1], inds)


def custom(t, data, func, inds=None):
    """Apply custom function to the upcrossing analysis

    Parameters
    ----------
    t: np.array
        Time array.
    data: np.array
        Signal time-series.
    func: f(ind1, ind2) -> np.array
        Function to apply between the zero crossing periods
        given t[ind1], t[ind2], where ind1 < ind2, correspond
        to the start and end of an upcrossing section.
    inds: np.array
        Optional indicies for the upcrossing. Useful
        when using several of the upcrossing methods
        to avoid repeating the upcrossing analysis 
        each time.

    Returns
    -------
    values: np.array
        Custom values of the time-series
    """

    assert isinstance(t, np.ndarray), 't must be of type np.ndarray'
    assert isinstance(data, np.ndarray), 'data must of type np.ndarray'
    assert callable(func), 'f must be callable'

    return _apply(t, data, func, inds)
