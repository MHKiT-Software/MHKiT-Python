"""
This module provides functionalities to calculate and analyze Most 
Likely Extreme Response (MLER) coefficients for wave energy converter
design and risk assessment. It includes functions to:

  - Calculate MLER coefficients (`mler_coefficients`) from a sea state
    spectrum and a response Amplitude Response Operator (ARO).
  - Define and manipulate simulation parameters (`mler_simulation`) used
    across various MLER analyses.
  - Renormalize the incoming amplitude of the MLER wave 
    (`mler_wave_amp_normalize`) to match the desired peak height for more
    accurate modeling and analysis.
  - Export the wave amplitude time series (`mler_export_time_series`) 
    based on the calculated MLER coefficients for further analysis or
    visualization.
"""

from typing import Union, List, Optional, Dict

import pandas as pd
import xarray as xr
import numpy as np
from numpy.typing import NDArray

from mhkit.wave.resource import frequency_moment

SimulationParameters = Dict[str, Union[float, int, np.ndarray]]


def mler_coefficients(
    rao: Union[NDArray[np.float_], pd.Series, List[float], List[int], xr.DataArray],
    wave_spectrum: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
    response_desired: Union[int, float],
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
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

    if isinstance(rao, (list, pd.Series, xr.DataArray)):
        rao_array = np.array(rao)
    elif isinstance(rao, np.ndarray):
        rao_array = rao
    else:
        raise TypeError(
            "Unsupported type for 'rao'. Must be one of: list, pd.Series, \
            np.ndarray, xr.DataArray."
        )

    if not isinstance(rao_array, np.ndarray):
        raise TypeError(f"rao must be of type np.ndarray. Got: {type(rao_array)}")
    if not isinstance(
        wave_spectrum, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(
            f"wave_spectrum must be of type pd.Series, pd.DataFrame, "
            f"xr.DataArray, or xr.Dataset. Got: {type(wave_spectrum)}"
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
    freq_hz = wave_spectrum.coords[frequency_dimension].values * (2 * np.pi)
    wave_spectrum = wave_spectrum.to_numpy() / (2 * np.pi)

    # get frequency step
    dw = 2.0 * np.pi / (len(freq_hz) - 1)

    # Note: waves.A is "S" in Quon2016; 'waves' naming convention
    # matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    spectrum_r = np.abs(rao_array) ** 2 * (2 * wave_spectrum)

    # calculate spectral moments and other important spectral values.
    m0 = frequency_moment(pd.Series(spectrum_r, index=freq_hz), 0).iloc[0, 0]
    m1_m2 = (
        frequency_moment(pd.Series(spectrum_r, index=freq_hz), 1).iloc[0, 0],
        frequency_moment(pd.Series(spectrum_r, index=freq_hz), 2).iloc[0, 0],
    )

    # calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
    # Drummen version.  Dietz has negative of this.
    _coeff_a_rn = (
        np.abs(rao)
        * np.sqrt(2 * wave_spectrum * dw)
        * (
            (m1_m2[1] - freq_hz * m1_m2[0])
            + (m1_m2[0] / m0) * (freq_hz * m0 - m1_m2[0])
        )
        / (m0 * m1_m2[1] - m1_m2[0] ** 2)
    )
    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase = -np.unwrap(np.angle(rao_array))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift, then flip sign on
    # negative Amplitudes
    _phase[_coeff_a_rn < 0] -= np.pi
    _coeff_a_rn[_coeff_a_rn < 0] *= -1

    # calculate the conditioned spectrum [m^2-s/rad]
    conditioned_spectrum = wave_spectrum * _coeff_a_rn**2 * response_desired**2

    # if the response amplitude we ask for is negative, we will add
    # a pi phase shift to the phase information.  This is because
    # the sign of self.desiredRespAmp is lost in the squaring above.
    # Ordinarily this would be put into the final equation, but we
    # are shaping the wave information so that it is buried in the
    # new spectral information, S. (AP)
    if response_desired < 0:
        _phase += np.pi

    mler = xr.Dataset(
        {
            "WaveSpectrum": (["frequency"], conditioned_spectrum),
            "Phase": (["frequency"], _phase + np.pi * (response_desired < 0)),
        },
        coords={"frequency": freq_hz},
    )
    mler.fillna(0)

    return mler.to_pandas() if to_pandas else mler


def mler_simulation(
    parameters: Optional[SimulationParameters] = None,
) -> SimulationParameters:
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
        - 'startTime': starting time [s]
        - 'endTime': ending time [s]
        - 'dT': time-step size [s]
        - 'T0': time of maximum event [s]
        - 'startx': start of simulation space [m]
        - 'endX': end of simulation space [m]
        - 'dX': horizontal spacing [m]
        - 'X': position of maximum event [m]
        The following keys are calculated from the above parameters:
        - 'maxIT': int, maximum timestep index
        - 'T': np.ndarray, time array
        - 'maxIX': int, maximum index for space
        - 'X': np.ndarray, space array

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

    if parameters is None:
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
    wave_amp: float,
    mler: Union[pd.DataFrame, xr.Dataset],
    sim: SimulationParameters,
    k: Union[NDArray[np.float_], List[float], pd.Series],
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
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
    if not isinstance(k, np.ndarray):
        k = np.array(k, dtype=float)
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


def mler_export_time_series(
    rao: Union[NDArray[np.float_], List[float], pd.Series],
    mler: Union[pd.DataFrame, xr.Dataset],
    sim: SimulationParameters,
    k: Union[NDArray[np.float_], List[float], pd.Series],
    frequency_dimension: str = "",
    to_pandas: bool = True,
) -> Union[pd.DataFrame, xr.Dataset]:
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
    rao_array = np.array(rao, dtype=float) if not isinstance(rao, np.ndarray) else rao
    k_array = np.array(k, dtype=float) if not isinstance(k, np.ndarray) else k

    if not isinstance(rao_array, np.ndarray):
        raise TypeError(f"rao must be of type ndarray. Got: {type(rao_array)}")
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}"
        )
    if not isinstance(sim, dict):
        raise TypeError(f"sim must be of type dict. Got: {type(sim)}")
    if not isinstance(k_array, np.ndarray):
        raise TypeError(f"k must be of type ndarray. Got: {type(k_array)}")
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
            * np.cos(
                freq * (ti - sim["T0"]) + mler["Phase"] - k_array * (xi - sim["X0"])
            )
        )
        # Response calculation
        wave_amp_time[i, 1] = np.sum(
            np.sqrt(2 * mler["WaveSpectrum"] * dw)
            * np.abs(rao_array)
            * np.cos(freq * (ti - sim["T0"]) - k_array * (xi - sim["X0"]))
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
