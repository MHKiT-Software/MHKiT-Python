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

from typing import Union, List, Optional, Dict, Any

import pandas as pd
import xarray as xr
import numpy as np
from numpy.typing import NDArray

from mhkit.wave.resource import frequency_moment

SimulationParameters = Dict[str, Union[float, int, np.ndarray]]


def _calculate_spectral_values(
    freq_hz: Union[np.ndarray, pd.Series],
    rao_array: np.ndarray,
    wave_spectrum: Union[pd.Series, pd.DataFrame, np.ndarray],
    d_w: float,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculates spectral moments and the coefficient A_{R,n} from a given sea state spectrum
    and a response RAO.

    Parameters
    ----------
    spectrum_r : Union[np.ndarray, pd.Series]
        Real part of the spectrum.
    freq_hz : Union[np.ndarray, pd.Series]
        Frequencies in Hz corresponding to spectrum_r.
    rao : numpy ndarray
        Response Amplitude Operator (RAO) of the system.
    wave_spectrum : Union[pd.Series, pd.DataFrame, np.ndarray]
        Wave spectrum values corresponding to freq_hz.
    d_w : float
        Delta omega, the frequency interval.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        A dictionary containing spectral moments (m_0, m_1, m_2) and the coefficient A_{R,n}.
    """
    # Note: waves.A is "S" in Quon2016; 'waves' naming convention
    # matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    spectrum_r = np.abs(rao_array) ** 2 * (2 * wave_spectrum)

    # Calculate spectral moments
    m_0 = frequency_moment(pd.Series(spectrum_r, index=freq_hz), 0).iloc[0, 0]
    m_1 = frequency_moment(pd.Series(spectrum_r, index=freq_hz), 1).iloc[0, 0]
    m_2 = frequency_moment(pd.Series(spectrum_r, index=freq_hz), 2).iloc[0, 0]

    # Calculate coefficient A_{R,n}
    coeff_a_rn = (
        np.abs(rao_array)
        * np.sqrt(2 * wave_spectrum * d_w)
        * ((m_2 - freq_hz * m_1) + (m_1 / m_0) * (freq_hz * m_0 - m_1))
        / (m_0 * m_2 - m_1**2)
    )

    return {
        "m_0": m_0,
        "m_1": m_1,
        "m_2": m_2,
        "coeff_a_rn": coeff_a_rn,
    }


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
    d_w = 2.0 * np.pi / (len(freq_hz) - 1)

    spectral_values = _calculate_spectral_values(freq_hz, rao_array, wave_spectrum, d_w)

    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase = -np.unwrap(np.angle(rao_array))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift, then flip sign on
    # negative Amplitudes
    _phase[spectral_values["coeff_a_rn"] < 0] -= np.pi
    spectral_values["coeff_a_rn"][spectral_values["coeff_a_rn"] < 0] *= -1

    # calculate the conditioned spectrum [m^2-s/rad]
    conditioned_spectrum = (
        wave_spectrum * spectral_values["coeff_a_rn"] ** 2 * response_desired**2
    )

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
            "WaveSpectrum": (["frequency"], np.array(conditioned_spectrum)),
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
    **kwargs: Any,
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
    frequency_dimension = kwargs.get("frequency_dimension", "")
    to_pandas = kwargs.get("to_pandas", True)

    k_array = np.array(k, dtype=float) if not isinstance(k, np.ndarray) else k

    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}"
        )
    if not isinstance(wave_amp, (int, float)):
        raise TypeError(f"wave_amp must be of type int or float. Got: {type(wave_amp)}")
    if not isinstance(sim, dict):
        raise TypeError(f"sim must be of type dict. Got: {type(sim)}")
    if not isinstance(frequency_dimension, str):
        raise TypeError(
            "frequency_dimension must be of type bool."
            + f"Got: {type(frequency_dimension)}"
        )
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")

    # If input is pandas, convert to xarray
    mler_xr = mler.to_xarray() if isinstance(mler, pd.DataFrame) else mler()

    # Determine frequency dimension
    freq_dim = frequency_dimension or list(mler_xr.coords)[0]
    # freq = mler_xr.coords[freq_dim].values * 2 * np.pi
    # d_w = np.diff(freq).mean()

    wave_amp_time = np.array(
        [
            np.sum(
                np.sqrt(
                    2
                    * mler_xr["WaveSpectrum"].values
                    * np.diff(mler_xr.coords[freq_dim].values * 2 * np.pi).mean()
                )
                * np.cos(
                    mler_xr.coords[freq_dim].values * 2 * np.pi * (t - sim["T0"])
                    - k_array * (x - sim["X0"])
                    + mler_xr["Phase"].values
                )
            )
            for x in np.linspace(sim["startX"], sim["endX"], sim["maxIX"])
            for t in np.linspace(sim["startTime"], sim["endTime"], sim["maxIT"])
        ]
    ).reshape(sim["maxIX"], sim["maxIT"])

    rescale_fact = np.abs(wave_amp) / np.max(np.abs(wave_amp_time))

    # Rescale the wave spectral amplitude coefficients and assign phase
    mler_norm = xr.Dataset(
        {
            "WaveSpectrum": (
                ["frequency"],
                mler_xr["WaveSpectrum"].data * rescale_fact**2,
            ),
            "Phase": (["frequency"], mler_xr["Phase"].data),
        },
        coords={"frequency": (["frequency"], mler_xr.coords[freq_dim].data)},
    )
    return mler_norm.to_pandas() if to_pandas else mler_norm


def mler_export_time_series(
    rao: Union[NDArray[np.float_], List[float], pd.Series],
    mler: Union[pd.DataFrame, xr.Dataset],
    sim: SimulationParameters,
    k: Union[NDArray[np.float_], List[float], pd.Series],
    **kwargs: Any,
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
    frequency_dimension = kwargs.get("frequency_dimension", "")
    to_pandas = kwargs.get("to_pandas", True)

    if not isinstance(rao, np.ndarray):
        raise TypeError(f"rao must be of type ndarray. Got: {type(rao)}")
    if not isinstance(mler, (pd.DataFrame, xr.Dataset)):
        raise TypeError(
            f"mler must be of type pd.DataFrame or xr.Dataset. Got: {type(mler)}"
        )
    if not isinstance(sim, dict):
        raise TypeError(f"sim must be of type dict. Got: {type(sim)}")
    if not isinstance(k, (np.ndarray, list, pd.Series)):
        raise TypeError(f"k must be of type ndarray. Got: {type(k)}")
    if not isinstance(to_pandas, bool):
        raise TypeError(f"to_pandas must be of type bool. Got: {type(to_pandas)}")
    if not isinstance(frequency_dimension, str):
        raise TypeError(
            f"frequency_dimension must be of type str. Got: {type(frequency_dimension)}"
        )

    rao = np.array(rao, dtype=float) if not isinstance(rao, np.ndarray) else rao
    k = np.array(k, dtype=float) if not isinstance(k, np.ndarray) else k
    # If input is pandas, convert to xarray
    mler = mler if isinstance(mler, xr.Dataset) else mler.to_xarray()

    # Handle optional frequency dimension
    frequency_dimension = (
        frequency_dimension if frequency_dimension else list(mler.coords)[0]
    )
    freq = mler.coords[frequency_dimension].values * 2 * np.pi
    d_w = np.diff(freq).mean()

    wave_height = np.zeros(len(sim["T"]))
    linear_response = np.zeros(len(sim["T"]))
    for i, t_i in enumerate(sim["T"]):
        cos_terms = np.cos(
            freq * (t_i - sim["T0"])
            - k * (sim["X0"] - sim["X0"])
            + mler["Phase"].values
        )
        wave_height[i] = np.sum(np.sqrt(2 * mler["WaveSpectrum"] * d_w) * cos_terms)

        linear_response[i] = np.sum(
            np.sqrt(2 * mler["WaveSpectrum"] * d_w)
            * np.abs(rao)
            * np.cos(freq * (t_i - sim["T0"]) - k * (sim["X0"] - sim["X0"]))
        )

    # Construct the output dataset
    mler_ts = xr.Dataset(
        {
            "WaveHeight": (["time"], wave_height),
            "LinearResponse": (["time"], linear_response),
        },
        coords={"time": sim["T"]},
    )

    # Convert to pandas DataFrame if requested
    return mler_ts.to_dataframe() if to_pandas else mler_ts
