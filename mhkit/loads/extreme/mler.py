import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats, optimize, signal
from mhkit.wave.resource import frequency_moment
from mhkit.utils import upcrossing, custom


def mler_coefficients(
    rao, wave_spectrum, response_desired, frequency_dimension="", to_pandas=True
):
    """
    Calculate MLER (most likely extreme response) coefficients from a
    sea state spectrum and a response RAO.

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    wave_spectrum: pandas Series, pandas DataFrame, xarray DataArray, or xarray Dataset
        Wave spectral density [m^2/Hz] indexed by frequency [Hz].
        DataFrame and Dataset inputs should only have one data variable
    response_desired: int or float
        Desired response, units should correspond to a motion RAO or
        units of force for a force RAO.
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler: pandas DataFrame or xarray Dataset
        DataFrame containing conditioned wave spectral amplitude
        coefficient [m^2-s], and Phase [rad] indexed by freq [Hz].
    """
    try:
        rao = np.array(rao)
    except:
        pass

    if not isinstance(rao, np.ndarray):
        raise TypeError(f"rao must be of type np.ndarray. Got: {type(rao)}")
    if not isinstance(
        wave_spectrum, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            f"wave_spectrum must be of type pd.Series, pd.DataFrame, xr.DataArray, or xr.Dataset. Got: {type(wave_spectrum)}"
        )
    if not isinstance(response_desired, (int, float)):
        raise TypeError(
            f"response_desired must be of type int or float. Got: {type(response_desired)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # Convert input to xarray DataArray
    if isinstance(wave_spectrum, (pd.Series, pd.DataFrame)):
        wave_spectrum = wave_spectrum.squeeze().to_xarray()

    if isinstance(wave_spectrum, xr.Dataset):
        if len(wave_spectrum.data_vars) > 1:
            raise ValueError(
                f"wave_spectrum can only contain one variable. Got {list(wave_spectrum.data_vars)}."
            )
        wave_spectrum = wave_spectrum.to_array()

    if frequency_dimension == "":
        frequency_dimension = list(wave_spectrum.coords)[0]

    # convert from Hz to rad/s
    freq_hz = wave_spectrum.coords[frequency_dimension].values
    freq = freq_hz * (2 * np.pi)
    wave_spectrum = wave_spectrum.to_numpy() / (2 * np.pi)

    # get frequency step
    dw = 2.0 * np.pi / (len(freq) - 1)

    # Note: waves.A is "S" in Quon2016; 'waves' naming convention
    # matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    spectrum_r = np.abs(rao) ** 2 * (2 * wave_spectrum)

    # calculate spectral moments and other important spectral values.
    m0 = (frequency_moment(pd.Series(spectrum_r, index=freq), 0)).iloc[0, 0]
    m1 = (frequency_moment(pd.Series(spectrum_r, index=freq), 1)).iloc[0, 0]
    m2 = (frequency_moment(pd.Series(spectrum_r, index=freq), 2)).iloc[0, 0]
    wBar = m1 / m0

    # calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
    # Drummen version.  Dietz has negative of this.
    _coeff_a_rn = (
        np.abs(rao)
        * np.sqrt(2 * wave_spectrum * dw)
        * ((m2 - freq * m1) + wBar * (freq * m0 - m1))
        / (m0 * m2 - m1**2)
    )

    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase = -np.unwrap(np.angle(rao))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift, then flip sign on
    # negative Amplitudes
    _phase[_coeff_a_rn < 0] -= np.pi
    _coeff_a_rn[_coeff_a_rn < 0] *= -1

    # calculate the conditioned spectrum [m^2-s/rad]
    _s = wave_spectrum * _coeff_a_rn**2 * response_desired**2
    _a = 2 * wave_spectrum * _coeff_a_rn**2 * response_desired**2

    # if the response amplitude we ask for is negative, we will add
    # a pi phase shift to the phase information.  This is because
    # the sign of self.desiredRespAmp is lost in the squaring above.
    # Ordinarily this would be put into the final equation, but we
    # are shaping the wave information so that it is buried in the
    # new spectral information, S. (AP)
    if response_desired < 0:
        _phase += np.pi

    mler = xr.Dataset(
        data_vars={
            "WaveSpectrum": (["frequency"], _s),
            "Phase": (["frequency"], _phase),
        },
        coords={"frequency": freq_hz},
    )
    mler.fillna(0)

    if to_pandas:
        mler = mler.to_pandas()

    return mler


def mler_simulation(parameters=None):
    """
    Define the simulation parameters that are used in various MLER
    functionalities.

    See `extreme_response_contour_example.ipynb` example for how this is
    useful. If no input is given, then default values are returned.

    Parameters
    ----------
    parameters: dict (optional)
        Simulation parameters.
        Keys:
        -----
        'startTime': starting time [s]
        'endTime': ending time [s]
        'dT': time-step size [s]
        'T0': time of maximum event [s]
        'startx': start of simulation space [m]
        'endX': end of simulation space [m]
        'dX': horizontal spacing [m]
        'X': position of maximum event [m]

    Returns
    -------
    sim: dict
        Simulation parameters including spatial and time calculated
        arrays.
    """
    if not isinstance(parameters, (type(None), dict)):
        raise TypeError(
            f"If specified, parameters must be of type dict. Got: {type(parameters)}"
        )

    sim = {}

    if parameters == None:
        sim["startTime"] = -150.0  # [s] Starting time
        sim["endTime"] = 150.0  # [s] Ending time
        sim["dT"] = 1.0  # [s] Time-step size
        sim["T0"] = 0.0  # [s] Time of maximum event

        sim["startX"] = -300.0  # [m] Start of simulation space
        sim["endX"] = 300.0  # [m] End of simulation space
        sim["dX"] = 1.0  # [m] Horiontal spacing
        sim["X0"] = 0.0  # [m] Position of maximum event
    else:
        sim = parameters

    # maximum timestep index
    sim["maxIT"] = int(np.ceil((sim["endTime"] - sim["startTime"]) / sim["dT"] + 1))
    sim["T"] = np.linspace(sim["startTime"], sim["endTime"], sim["maxIT"])

    sim["maxIX"] = int(np.ceil((sim["endX"] - sim["startX"]) / sim["dX"] + 1))
    sim["X"] = np.linspace(sim["startX"], sim["endX"], sim["maxIX"])

    return sim


def mler_wave_amp_normalize(
    wave_amp, mler, sim, k, frequency_dimension="", to_pandas=True
):
    """
    Function that renormalizes the incoming amplitude of the MLER wave
    to the desired peak height (peak to MSL).

    Parameters
    ----------
    wave_amp: float
        Desired wave amplitude (peak to MSL).
    mler: pandas DataFrame or xarray Dataset
        MLER coefficients generated by 'mler_coefficients' function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler_norm : pandas DataFrame or xarray Dataset
        MLER coefficients
    """
    try:
        k = np.array(k)
    except:
        pass
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}"
        )
    if not isinstance(wave_amp, (int, float)):
        raise TypeError(f"wave_amp must be of type int or float. Got: {type(wave_amp)}")
    if not isinstance(sim, dict):
        raise TypeError(f"sim must be of type dict. Got: {type(sim)}")
    if not isinstance(k, np.ndarray):
        raise TypeError(f"k must be of type ndarray. Got: {type(k)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # If input is pandas, convert to xarray
    if isinstance(mler, pd.DataFrame):
        mler = mler.to_xarray()

    if frequency_dimension == "":
        frequency_dimension = list(mler.coords)[0]
    freq = mler.coords[frequency_dimension].values * 2 * np.pi
    dw = (max(freq) - min(freq)) / (len(freq) - 1)  # get delta

    wave_amp_time = np.zeros((sim["maxIX"], sim["maxIT"]))
    for ix, x in enumerate(sim["X"]):
        for it, t in enumerate(sim["T"]):
            # conditioned wave
            wave_amp_time[ix, it] = np.sum(
                np.sqrt(2 * mler["WaveSpectrum"] * dw)
                * np.cos(freq * (t - sim["T0"]) - k * (x - sim["X0"]) + mler["Phase"])
            )

    tmp_max_amp = np.max(np.abs(wave_amp_time))

    # renormalization of wave amplitudes
    rescale_fact = np.abs(wave_amp) / np.abs(tmp_max_amp)

    # rescale the wave spectral amplitude coefficients
    mler_norm = mler["WaveSpectrum"] * rescale_fact**2
    mler_norm = mler_norm.to_dataset()
    mler_norm = mler_norm.assign({"Phase": (frequency_dimension, mler["Phase"].data)})

    if to_pandas:
        mler_norm = mler_norm.to_pandas()

    return mler_norm


def mler_export_time_series(rao, mler, sim, k, frequency_dimension="", to_pandas=True):
    """
    Generate the wave amplitude time series at X0 from the calculated
    MLER coefficients

    Parameters
    ----------
    rao: numpy ndarray
        Response amplitude operator.
    mler: pandas DataFrame or xarray Dataset
        MLER coefficients dataframe generated from an MLER function.
    sim: dict
        Simulation parameters formatted by output from
        'mler_simulation'.
    k: numpy ndarray
        Wave number.
    frequency_dimension: string (optional)
        Name of the xarray dimension corresponding to frequency. If not supplied,
        defaults to the first dimension. Does not affect pandas input.
    to_pandas: bool (optional)
        Flag to output pandas instead of xarray. Default = True.

    Returns
    -------
    mler_ts: pandas DataFrame or xarray Dataset
        Time series of wave height [m] and linear response [*] indexed
        by time [s].

    """
    try:
        rao = np.array(rao)
    except:
        pass
    try:
        k = np.array(k)
    except:
        pass
    if not isinstance(rao, np.ndarray):
        raise TypeError(f"rao must be of type ndarray. Got: {type(rao)}")
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}"
        )
    if not isinstance(sim, dict):
        raise TypeError(f"sim must be of type dict. Got: {type(sim)}")
    if not isinstance(k, np.ndarray):
        raise TypeError(f"k must be of type ndarray. Got: {type(k)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # If input is pandas, convert to xarray
    if isinstance(mler, pd.DataFrame):
        mler = mler.to_xarray()

    if frequency_dimension == "":
        frequency_dimension = list(mler.coords)[0]
    freq = mler.coords[frequency_dimension].values * 2 * np.pi
    dw = (max(freq) - min(freq)) / (len(freq) - 1)  # get delta

    # calculate the series
    wave_amp_time = np.zeros((sim["maxIT"], 2))
    xi = sim["X0"]
    for i, ti in enumerate(sim["T"]):
        # conditioned wave
        wave_amp_time[i, 0] = np.sum(
            np.sqrt(2 * mler["WaveSpectrum"] * dw)
            * np.cos(freq * (ti - sim["T0"]) + mler["Phase"] - k * (xi - sim["X0"]))
        )
        # Response calculation
        wave_amp_time[i, 1] = np.sum(
            np.sqrt(2 * mler["WaveSpectrum"] * dw)
            * np.abs(rao)
            * np.cos(freq * (ti - sim["T0"]) - k * (xi - sim["X0"]))
        )

    mler_ts = xr.Dataset(
        data_vars={
            "WaveHeight": (["time"], wave_amp_time[:, 0]),
            "LinearResponse": (["time"], wave_amp_time[:, 1]),
        },
        coords={"time": sim["T"]},
    )

    if to_pandas:
        mler_ts = mler_ts.to_pandas()

    return mler_ts
