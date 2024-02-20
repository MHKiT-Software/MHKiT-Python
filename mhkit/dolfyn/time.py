from datetime import datetime, timedelta, timezone
import numpy as np
from .tools.misc import fillgaps


def _fullyear(year):
    if year > 100:
        return year
    year += 1900 + 100 * (year < 90)
    return year


def epoch2dt64(ep_time):
    """
    Convert from epoch time (seconds since 1/1/1970 00:00:00) to
    numpy.datetime64 array

    Parameters
    ----------
    ep_time : xarray.DataArray
      Time coordinate data-array or single time element

    Returns
    -------
    time : numpy.datetime64
      The converted datetime64 array
    """

    # assumes t0=1970-01-01 00:00:00
    out = np.array(ep_time.astype("int")).astype("datetime64[s]")
    out = out + ((ep_time % 1) * 1e9).astype("timedelta64[ns]")
    return out


def dt642epoch(dt64):
    """
    Convert numpy.datetime64 array to epoch time
    (seconds since 1/1/1970 00:00:00)

    Parameters
    ----------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)

    Returns
    -------
    time : float
      Epoch time (seconds since 1/1/1970 00:00:00)
    """

    return dt64.astype("datetime64[ns]").astype("float") / 1e9


def date2dt64(dt):
    """
    Convert numpy.datetime64 array to list of datetime objects

    Parameters
    ----------
    time : datetime.datetime
      The converted datetime object

    Returns
    -------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)
    """

    return np.array(dt).astype("datetime64[ns]")


def dt642date(dt64):
    """
    Convert numpy.datetime64 array to list of datetime objects

    Parameters
    ----------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)

    Returns
    -------
    time : datetime.datetime
      The converted datetime object
    """

    return epoch2date(dt642epoch(dt64))


def epoch2date(ep_time, offset_hr=0, to_str=False):
    """
    Convert from epoch time (seconds since 1/1/1970 00:00:00) to a list
    of datetime objects

    Parameters
    ----------
    ep_time : xarray.DataArray
      Time coordinate data-array or single time element
    offset_hr : int
      Number of hours to offset time by (e.g. UTC -7 hours = PDT)
    to_str : logical
      Converts datetime object to a readable string

    Returns
    -------
    time : datetime.datetime
      The converted datetime object or list(strings)

    Notes
    -----
    The specific time instance is set during deployment, usually sync'd to the
    deployment computer. The time seen by DOLfYN is in the timezone of the
    deployment computer, which is unknown to DOLfYN.
    """

    try:
        ep_time = ep_time.values
    except AttributeError:
        pass

    if isinstance(ep_time, (np.ndarray)) and ep_time.ndim == 0:
        ep_time = [ep_time.item()]
    elif not isinstance(ep_time, (np.ndarray, list)):
        ep_time = [ep_time]

    ######### IMPORTANT #########
    # Note the use of `utcfromtimestamp` here, rather than `fromtimestamp`
    # This is CRITICAL! See the difference between those functions here:
    #    https://docs.python.org/3/library/datetime.html#datetime.datetime.fromtimestamp
    # Long story short: `fromtimestamp` used system-specific timezone
    # info to calculate the datetime object, but returns a
    # timezone-agnostic object.
    if offset_hr != 0:
        delta = timedelta(hours=offset_hr)
        time = [datetime.utcfromtimestamp(t) + delta for t in ep_time]
    else:
        time = [datetime.utcfromtimestamp(t) for t in ep_time]

    if to_str:
        time = date2str(time)

    return time


def date2str(dt, format_str=None):
    """
    Convert list of datetime objects to legible strings

    Parameters
    ----------
    dt : datetime.datetime
      Single or list of datetime object(s)
    format_str : string
      Timestamp string formatting. Default is '%Y-%m-%d %H:%M:%S.%f'
      See datetime.strftime documentation for timestamp string formatting.

    Returns
    -------
    time : string
      Converted timestamps
    """

    if format_str is None:
        format_str = "%Y-%m-%d %H:%M:%S.%f"

    if not isinstance(dt, list):
        dt = [dt]

    return [t.strftime(format_str) for t in dt]


def date2epoch(dt):
    """
    Convert list of datetime objects to epoch time

    Parameters
    ----------
    dt : datetime.datetime
      Single or list of datetime object(s)

    Returns
    -------
    time : float
      Datetime converted to epoch time (seconds since 1/1/1970 00:00:00)
    """

    if not isinstance(dt, list):
        dt = [dt]

    return [t.replace(tzinfo=timezone.utc).timestamp() for t in dt]


def date2matlab(dt):
    """
    Convert list of datetime objects to MATLAB datenum

    Parameters
    ----------
    dt : datetime.datetime
      List of datetime objects

    Returns
    -------
    time : float
      List of timestamps in MATLAB datnum format
    """

    time = list()
    for i in range(len(dt)):
        mdn = dt[i] + timedelta(days=366)
        frac_seconds = (
            dt[i] - datetime(dt[i].year, dt[i].month, dt[i].day, 0, 0, 0)
        ).seconds / (24 * 60 * 60)
        frac_microseconds = dt[i].microsecond / (24 * 60 * 60 * 1000000)
        time.append(mdn.toordinal() + frac_seconds + frac_microseconds)

    return time


def matlab2date(matlab_dn):
    """
    Convert MATLAB datenum to list of datetime objects

    Parameters
    ----------
    matlab_dn : float
      List of timestamps in MATLAB datnum format

    Returns
    -------
    dt : datetime.datetime
      List of datetime objects
    """

    time = list()
    for i in range(len(matlab_dn)):
        day = datetime.fromordinal(int(matlab_dn[i]))
        dayfrac = timedelta(days=matlab_dn[i] % 1) - timedelta(days=366)
        time.append(day + dayfrac)

        # Datenum is precise down to 100 microseconds - add difference to round
        us = int(round(time[i].microsecond / 100, 0)) * 100
        time[i] = time[i].replace(microsecond=time[i].microsecond) + timedelta(
            microseconds=us - time[i].microsecond
        )

    return time


def _fill_time_gaps(epoch, sample_rate_hz):
    """
    Fill gaps (NaN values) in the timeseries by simple linear
    interpolation.  The ends are extrapolated by stepping
    forward/backward by 1/sample_rate_hz.
    """

    # epoch is seconds since 1970
    dt = 1.0 / sample_rate_hz
    epoch = fillgaps(epoch)
    if np.isnan(epoch[0]):
        i0 = np.nonzero(~np.isnan(epoch))[0][0]
        delta = np.arange(-i0, 0, 1) * dt
        epoch[:i0] = epoch[i0] + delta
    if np.isnan(epoch[-1]):
        # Search backward through the array to get the 'negative index'
        ie = -np.nonzero(~np.isnan(epoch[::-1]))[0][0] - 1
        delta = np.arange(1, -ie, 1) * dt
        epoch[(ie + 1) :] = epoch[ie] + delta

    return epoch
