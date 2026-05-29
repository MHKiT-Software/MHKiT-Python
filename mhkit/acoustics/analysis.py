"""
This module contains key functions for passive acoustics analysis, designed to process
and analyze sound pressure data from .wav files in the frequency and time domains.
The functions herein build on each other, with a structured flow that facilitates the
calculation of sound pressure spectral densities and banded averages based on
input audio data.

The following functionality is provided:

1. **Frequency Validation and Warning**:

   - `_fmax_warning`: Ensures specified maximum frequency does not exceed the Nyquist frequency,
     adjusting if necessary to avoid aliasing.

2. **Shallow Water Cutoff Frequency**:

   - `minimum_frequency`: Calculates the minimum frequency cutoff based on water depth and the
     speed of sound in water and seabed materials.

3. **Calculation of Frequency Bands**:

    - `create_frequency_bands`: Generates frequency bands based on specified octave divisions,
      minimum and maximum frequency limits, and the chosen base (e.g., 2 for octaves, 10 for decades).

4. **Sound Pressure Spectral Density Calculation**:

    - `sound_pressure_spectral_density`: Computes the mean square sound pressure spectral density
      using FFT binning with Hanning windowing and 50% overlap.

5. **Calibration**:

   - `apply_calibration`: Applies calibration adjustments to the spectral density data using
     a sensitivity curve, filling missing values as specified.

6. **Band-Averaged Spectral Density**:

    - `_get_band_table`: Generates a table of frequency bands for logarithmically spaced divisions.
    - `_band_mean_power_spectral_density`: Computes the mean power spectral density within specified bands.
    - `convert_to_millidecade`, `convert_to_decidecade`, `convert_to_third_octave`: Convenience functions to convert
      spectral density to millidecade, decidecade, and third octave banded spectral densities, respectively.

"""

from typing import Union, Optional
import warnings
import numpy as np
import xarray as xr

from mhkit.dolfyn import VelBinner


def _check_numeric(value, name: str):
    if np.issubdtype(type(value), np.ndarray):
        value = value.item()
    if not (
        isinstance(value, (int, float))
        or np.issubdtype(type(value), np.integer)
        or np.issubdtype(type(value), np.floating)
    ):
        raise TypeError(f"{name} must be a numeric type (int or float).")


def _fmax_warning(
    fn: Union[int, float, np.ndarray], fmax: Union[int, float, np.ndarray]
) -> Union[int, float, np.ndarray]:
    """
    Checks that the maximum frequency limit isn't greater than the Nyquist frequency.

    Parameters
    ----------
    fn: int, float, or numpy.ndarray
        The Nyquist frequency in Hz.
    fmax: float
        The maximum frequency limit in Hz.

    Returns
    -------
    fmax: float
        The adjusted maximum frequency limit, ensuring it does not exceed the Nyquist frequency.
    """

    if fmax > fn:
        warnings.warn(
            f"`fmax` = {fmax} is greater than the Nyquist frequency. Setting"
            f"fmax = {fn}"
        )
        fmax = fn

    return fmax


def minimum_frequency(
    water_depth: Union[int, float, np.ndarray, list],
    c: Union[int, float] = 1500,
    c_seabed: Union[int, float] = 1700,
) -> Union[float, np.ndarray]:
    """
    Estimate the shallow water cutoff frequency based on the speed of
    sound in the water column and the speed of sound in the seabed
    material (generally ranges from 1450 - 1800 m/s)

    Parameters
    ----------
    water_depth: int, float or array-like
        Depth of the water column in meters.
    c: float, optional
        Speed of sound in the water column in meters per second. Default is 1500 m/s.
    c_seabed: float, optional
        Speed of sound in the seabed material in meters per second. Default is 1700 m/s.

    Returns
    -------
    f_min: float or numpy.ndarray
        The minimum cutoff frequency in Hz.

    Reference
    ---------
    Jennings 2011 - Computational Ocean Acoustics, 2nd ed.
    """

    # Convert water_depth to a NumPy array for vectorized operations
    water_depth = np.asarray(water_depth)

    # Validate water_depth
    if not np.issubdtype(water_depth.dtype, np.number):
        raise TypeError("'water_depth' must be a numeric type or array of numerics.")

    _check_numeric(c, "c")
    _check_numeric(c_seabed, "c_seabed")

    if np.any(water_depth <= 0):
        raise ValueError("All elements of 'water_depth' must be positive numbers.")
    if c <= 0:
        raise ValueError("'c' must be a positive number.")
    if c_seabed <= 0:
        raise ValueError("'c_seabed' must be a positive number.")
    if c_seabed <= c:
        raise ValueError("'c_seabed' must be greater than 'c'.")

    fmin = c / (4 * water_depth * np.sqrt(1 - (c / c_seabed) ** 2))

    return fmin


def create_frequency_bands(octave, base, fmin, fmax):
    """
    Calculates frequency bands based on the specified octave, minimum and
    maximum frequency limits.

    Parameters
    ----------
    octave: int
        Octave to subdivide spectral density level by.
    base : int, optional
        Octave base. Set to 2 for the true octave band; set to base 10 for
        the decidecade octave band. Default: 2
    fmin : int, optional
        Lower frequency band limit (lower limit of the hydrophone). Default is 10 Hz.
    fmax : int, optional
        Upper frequency band limit (Nyquist frequency). Default is 100,000 Hz.

    Returns
    -------
    octave_bins: numpy.array
        Array of octave bin edges
    band: dict(str, numpy.array)
        Dictionary containing the frequency band edges and center frequency
    """

    bandwidth = base ** (1 / octave)
    half_bandwidth = base ** (1 / (octave * 2))

    band = {}
    band["center_freq"] = 10 ** np.arange(
        np.log10(fmin),
        np.log10(fmax * bandwidth),
        step=np.log10(bandwidth),
    )
    band["lower_limit"] = band["center_freq"] / half_bandwidth
    band["upper_limit"] = band["center_freq"] * half_bandwidth
    octave_bins = np.append(band["lower_limit"], band["upper_limit"][-1])

    return octave_bins, band


def sound_pressure_spectral_density(
    pressure: xr.DataArray,
    fs: Union[int, float],
    bin_length: Union[int, float] = 1,
    fft_length: Optional[Union[int, float]] = None,
    rms: bool = True,
) -> xr.DataArray:
    """
    Calculates the sound pressure spectral density (SPSD) from audio
    samples split into FFTs with a specified bin length in seconds,
    using Hanning windowing with 50% overlap.

    By default (`rms=True`), this function returns the mean-squared SPSD,
    which found by scaling the total spectral power (frequency domain) with
    the time-domain averaged mean-squared power, in accordance with
    Parseval's theorem.

    Setting `rms=False` disables this scaling and returns the
    power spectral density of the sound pressure signal.
    Both forms have units of [Pa^2/Hz] or [V^2/Hz].

    Parameters
    ----------
    pressure: xarray.DataArray (time)
        Sound pressure in [Pa] or voltage [V]
    fs: int or float
        Data collection sampling rate [Hz]
    bin_length: int or float
        Length of time in seconds to create FFTs. Default: 1.
    fft_length: int or float, optional
        Length of FFT to use. If None, uses bin_length * fs. Default: None.
    rms: bool
        If True, calculates the mean-squared SPSD. Set to False to
        calculate standard SPSD. Default: True.

    Returns
    -------
    spsd: xarray.DataArray (time, freq)
        Spectral density [Pa^2/Hz] indexed by time and frequency
    """

    # Type checks
    if not isinstance(pressure, xr.DataArray):
        raise TypeError("'pressure' must be an xarray.DataArray.")
    _check_numeric(fs, "fs")
    _check_numeric(bin_length, "bin_length")

    # Ensure that 'pressure' has a 'time' coordinate
    if "time" not in pressure.dims:
        raise ValueError("'pressure' must be indexed by 'time' dimension.")

    # window length of each time series
    nbin = bin_length * fs
    if fft_length is not None:
        _check_numeric(fft_length, "fft_length")
        nfft = fft_length
    else:
        nfft = nbin

    # Use dolfyn PSD functionality
    binner = VelBinner(n_bin=nbin, fs=fs, n_fft=nfft)
    # Always 50% overlap if numbers reshape perfectly
    # Mean square sound pressure
    psd = binner.power_spectral_density(pressure, freq_units="Hz")
    if rms:
        # Scale PSD by mean square of original signal
        samples = (
            binner.reshape(pressure.values) - binner.mean(pressure.values)[:, None]
        )
        # mean squared pressure ("power") in time domain
        t_power = np.sum(samples**2, axis=1) / nbin
        # pressure ("power") in frequency domain
        f_power = psd.sum("freq") * (fs / nbin)
        # Adjust the amplitude of the PSD to return the mean-squared PSD
        # based on Parseval's theorem: total energy computed in the time
        # domain must equal the total energy computed in the frequency domain
        psd = psd * t_power[:, None] / f_power
        long_name = "Mean Square Sound Pressure Spectral Density"
    else:
        long_name = "Sound Pressure Spectral Density"

    out = xr.DataArray(
        psd,
        coords={"time": psd["time"], "freq": psd["freq"]},
        attrs={
            "units": pressure.units + "^2/Hz",
            "long_name": long_name,
            "fs": fs,
            "bin_length": bin_length,
            "overlap": "50%",
            "n_fft": nfft,
        },
    )

    return out


def apply_calibration(
    spsd: xr.DataArray,
    sensitivity_curve: xr.DataArray,
    fill_value: Union[float, int, np.ndarray],
    interp_method: str = "linear",
) -> xr.DataArray:
    """
    Applies custom calibration to spectral density values.

    Parameters
    ----------
    spsd: xarray.DataArray (time, freq)
        Mean square sound pressure spectral density in V^2/Hz.
    sensitivity_curve: xarray.DataArray (freq)
        Calibrated sensitivity curve in units of dB rel 1 V^2/uPa^2.
        First column should be frequency, second column should be calibration values.
    fill_value: float or int
        Value with which to fill missing values from the calibration curve,
        in units of dB rel 1 V^2/uPa^2.
    interp_method: str
        Interpolation method to use when interpolating the calibration curve
        to the frequencies in 'spsd'. Default is 'linear'.

    Returns
    -------
    spsd_calibrated: xarray.DataArray (time, freq)
        Spectral density in Pa^2/Hz, indexed by time and frequency.
    """

    if not isinstance(spsd, xr.DataArray):
        raise TypeError("'spsd' must be an xarray.DataArray.")
    if not isinstance(sensitivity_curve, xr.DataArray):
        raise TypeError("'sensitivity_curve' must be an xarray.DataArray.")
    _check_numeric(fill_value, "fill_value")

    # Ensure 'freq' dimension exists in 'spsd'
    if "freq" not in spsd.dims:
        if len(spsd.dims) > 1:
            # Issue a warning and assign the 2nd dimension as 'freq'
            warnings.warn(
                f"'spsd' does not have 'freq' as a dimension and has multiple dimensions. "
                f"Using the second dimension '{spsd.dims[1]}' as 'freq'."
            )
        # Assign the 2nd dimension as 'freq'
        spsd = spsd.rename({spsd.dims[1]: "freq"})

    # Ensure 'freq' dimension exists in 'sensitivity_curve'
    if "freq" not in sensitivity_curve.dims:
        if len(sensitivity_curve.dims) > 1:
            # Issue a warning and assign the 1st dimension as 'freq'
            warnings.warn(
                f"'sensitivity_curve' does not have 'freq' as a dimension \
                      and has multiple dimensions. "
                f"Using the first dimension '{sensitivity_curve.dims[0]}' as 'freq'."
            )
        # Assign the 0th dimension as 'freq'
        sensitivity_curve = sensitivity_curve.rename(
            {sensitivity_curve.dims[0]: "freq"}
        )

    # Create a copy of spsd to avoid in-place modification
    spsd_calibrated = spsd.copy(deep=True)
    attrs = spsd.attrs  # recover attrs

    # Read calibration curve
    freq = sensitivity_curve.dims[0]
    # Interpolate calibration curve to desired value
    calibration = sensitivity_curve.interp(
        {freq: spsd_calibrated["freq"]}, method=interp_method
    )
    # Fill missing with provided value
    calibration = calibration.fillna(fill_value)

    # Subtract from sound pressure spectral density
    sensitivity_ratio = 10 ** (calibration / 10)  # V^2/uPa^2
    spsd_calibrated = spsd_calibrated / sensitivity_ratio / 1e12  # Pa^2/Hz
    attrs.update(
        {"long_name": "Calibrated Sound Pressure Spectral Density", "units": "Pa^2/Hz"}
    )
    spsd_calibrated.attrs = attrs

    return spsd_calibrated


def _get_band_table(
    fft_bin_size: int,
    bin1_center_freq: float,
    fs: float,
    base: int,
    bands_per_division: int,
    first_output_band_center_freq: float,
    use_fft_res_at_bottom: bool,
):
    """
    Returns a three column array with the start, center, stop frequecies for
    logarthimically spaced frequency bands such as millidecades, decidecades,
    or third octaves base 2. These tables are passed to
    `band_squared_sound_pressure` to convert square spectra to band
    levels. The squared pressure can be converted to power spectral
    density by dividing by the band_widths.

    Inputs:
        "fft_bin_size" - the size of the FFT bins in Hz that subsequent
                  processing of data will use.
        "bin1_center_freq" - this is the center frequency in Hz of the FFT
            spectra that will be passed to subsequent processing, normally
                  this should be zero.
        "fs" - data sampling frequency in Hz
        "base" - base for the band levels, generally 10 or 2.
        "bands_per_division" - the number of bands to divide the spectrum into
            per increase by a factor of 'base'. A base of 2 and
            "bands_per_division" of 3 results in third octaves base 2. Base 10
            and "bands_per_division" of 1000 results in millidecades.
        "first_output_band_center_freq": this is the frequency where the output
                  bands will start.
        "use_fft_res_at_bottom": In some cases, like millidecades, we do not want
            to have logarithmically spaced frequency bands across the full
            spectrum, instead we have the option to have bands that are equal
            'fft_bin_size'. The switch to log spacing is made at the band that
                  has a bandwidth greater than 'fft_bin_size' and such that the
            frequency space between band center frequencies is at least
                  'fft_bin_size'.

    Outputs:
          Three column array where column 1 is the lowest frequency of the
          band, column 2 is the center frequency, and 3 is the highest
          frequency
    Example Usage:
       fft_bin_size = fs/fftSize # fs is sample rate,
             fftSize is number of points in your FFT
       millidecade_bands = get_band_table(fft_bin_size, 0, fs, 10, 1000, 1, 1)
       decidecade_bands = get_band_table(fft_bin_size, 0, fs, 10, 10, 1, 0)
       third_octave_bands = get_band_table(fft_bin_size, 0, fs, 2, 3, 1, 0)

    Author: Bruce Martin, JASCO Applied Sciences, Feb 2020.
             bruce.martin@jasco.com.
    get_band_table(1, 0, 50000, 10, 1000, 1, 1)

    Converted and adapted to Python by MHKiT Team, May 28, 2026
    """

    band_count = 0
    maxFreq = fs / 2
    low_side_multiplier = base ** (-1 / (2 * bands_per_division))
    high_side_multiplier = base ** (1 / (2 * bands_per_division))

    # Count the number of bands:
    linear_bin_count = 0
    log_bin_count = 0
    center_freq = 0

    # For millidecades
    if use_fft_res_at_bottom:
        bin_width = 0
        while bin_width < fft_bin_size:
            band_count += 1
            center_freq = first_output_band_center_freq * base ** (
                band_count / bands_per_division
            )
            bin_width = (
                high_side_multiplier * center_freq - low_side_multiplier * center_freq
            )

        # Now keep counting until the difference between the log spaced
        # center frequency and new frequency is greater than .025
        center_freq = first_output_band_center_freq * base ** (
            band_count / bands_per_division
        )
        linear_bin_count = np.ceil(center_freq / fft_bin_size)
        while (linear_bin_count * fft_bin_size - center_freq) > 0:
            band_count = band_count + 1
            linear_bin_count = linear_bin_count + 1
            center_freq = first_output_band_center_freq * base ** (
                band_count / bands_per_division
            )

        if (fft_bin_size * linear_bin_count) > maxFreq:
            linear_bin_count = maxFreq / fft_bin_size + 1

    linear_bin_count = int(linear_bin_count)
    log_band1 = band_count

    # Count the log space frequencies
    while maxFreq > center_freq:
        band_count = band_count + 1
        log_bin_count = log_bin_count + 1
        center_freq = first_output_band_center_freq * base ** (
            band_count / bands_per_division
        )

    # Generate the linear frequencies (For millidecades)
    bands = np.zeros(((linear_bin_count + log_bin_count), 3))
    for i in range(linear_bin_count):
        bands[i, 1] = bin1_center_freq + i * fft_bin_size
        bands[i, 0] = bands[i, 1] - fft_bin_size / 2
        bands[i, 2] = bands[i, 1] + fft_bin_size / 2

    # Generate the log-spaced bands
    for i in range(log_bin_count):
        out_band_num = linear_bin_count + i
        m_dec_num = log_band1 + i
        bands[out_band_num, 1] = first_output_band_center_freq * base ** (
            m_dec_num / bands_per_division
        )
        bands[out_band_num, 0] = bands[out_band_num, 1] * low_side_multiplier
        bands[out_band_num, 2] = bands[out_band_num, 1] * high_side_multiplier

    if log_bin_count > 0:
        bands[out_band_num, 2] = maxFreq

    return bands


def _band_mean_power_spectral_density(
    input_spsd,
    fft_bin_size,
    bin1_center_freq,
    first_band_idx,
    last_band_idx,
    freq_table,
):
    """
    Sums squared sound pressures to determine the in-band totals then divides by the
    band_widths to get PSD. The band edges are normally obtained from a call to `get_band_table`
    `band_squared_sound_pressure` is called to get the band SPLs

    Note that the output of `band_squared_sound_pressure` should satisfy
    Parseval's theorem, but the output of `band_mean_power_spectral_density`
    will not unless the bands are re-multiplied by the band widths.
    Results are returned as linear units, not levels.

    Inputs:
        "input_spsd" - array of squared pressures from an FFT with a frequency
              step size index 1 (rows) are time, index 2 (columns) are freq.
        "fft_bin_size" - the size of the FFT bins in Hz.
        "bin1_center_freq": the freq in Hz of the first element of the FFT
              array - normally this is frequency zero.
        "first_band_idx": the index in 'freq_table' of the first band to compute and
              output
        "last_band_idx": the index in 'freq_table' of the last band to compute and
            output
        "freq_table" - the list of band edges - Nx3 array where column 1 is the
            lowest band frquency, column 2 is the center frequency and 3 is
                  the maximum.

    Outputs:  band squared sound pressure array with the same number of rows as
         input_spsd and one column per band.

    Bruce Martin, JASCO Applied Sciences, Feb 2020.

    Converted and adapted to Python by MHKiT Team, May 28, 2026
    """

    out_spsd = np.zeros((input_spsd.shape[0], last_band_idx - first_band_idx))
    step = fft_bin_size / 2
    n_fft_bins = input_spsd.shape[1]
    start_offset = np.floor(bin1_center_freq / fft_bin_size)

    for j in range(first_band_idx, last_band_idx):
        min_fft_bin = int(
            np.floor((freq_table[j, 0] / fft_bin_size) + step) + 1 - start_offset
        )
        max_fft_bin = int(
            np.floor((freq_table[j, 2] / fft_bin_size) + step) - start_offset
        )
        if max_fft_bin > n_fft_bins:
            max_fft_bin = n_fft_bins
        if min_fft_bin < 0:
            min_fft_bin = 0
        if min_fft_bin == max_fft_bin:
            out_spsd[:, j] = input_spsd[:, min_fft_bin] * (
                (freq_table[j, 2] - freq_table[j, 0]) / fft_bin_size
            )
        else:
            # Add the first partial FFT bin - take the top of the bin and
            # subtract the lower freq to get the amount we will use:
            # the top freq of a bin is bin# * step size - binSize/2 since bin
            # centers are at bin# * step size. We also need to subtract the start
            # offset to get the correct bin number since the first bin may not be at 0 Hz.
            lower_factor = (min_fft_bin + 1 - step) * fft_bin_size - freq_table[j, 0]
            out_spsd[:, j] = input_spsd[:, min_fft_bin] * lower_factor

            # Add the last partial FFT bin.
            upper_factor = (
                freq_table[j, 2] - (max_fft_bin + 1 - 1.5 * fft_bin_size) * fft_bin_size
            )
            out_spsd[:, j] = out_spsd[:, j] + input_spsd[:, max_fft_bin] * upper_factor
            #
            # Add any FFT bins in between min and max.
            if (max_fft_bin - min_fft_bin) > 1:
                out_spsd[:, j] += np.nansum(
                    input_spsd[:, np.arange(min_fft_bin + 1, max_fft_bin)], axis=1
                )

    # Take means
    band_widths = freq_table[:, 2] - freq_table[:, 0]
    out_spsd /= band_widths

    return out_spsd


def _convert_to_band_spectral_density(
    spsd: xr.DataArray, base: int, bands_per_division: int, use_fft_res_at_bottom: bool
) -> xr.DataArray:
    """Convert sound pressure spectral density to banded spectral density based on
    the specified base and bands per division.

    Args:
        spsd (xr.DataArray): DataArray with frequency dimension.
        base (int): Base for the band levels, generally 10 or 2.
        bands_per_division (int): The number of bands to divide the spectrum into
            per increase by a factor of 'base'. A base of 2 and
            "bands_per_division" of 3 results in third octaves base 2. Base 10
            and "bands_per_division" of 1000 results in millidecades.
        use_fft_res_at_bottom (bool): In some cases, like millidecades, we do not want
            to have logarithmically spaced frequency bands across the full
            spectrum, instead we have the option to have bands that are equal
            'fft_bin_size'. The switch to log spacing is made at the band that
                  has a bandwidth greater than 'fft_bin_size' and such that the
            frequency space between band center frequencies is at least
                  'fft_bin_size'.

    Returns:
        xr.DataArray: DataArray with frequency dimension converted to banded spectral density.
    """
    # Get original frequencies
    freq = spsd["freq"].values
    # Get bands
    bands = _get_band_table(
        fft_bin_size=freq[1] - freq[0],
        bin1_center_freq=freq[0],
        fs=spsd.fs,
        base=base,
        bands_per_division=bands_per_division,
        first_output_band_center_freq=freq[0],
        use_fft_res_at_bottom=use_fft_res_at_bottom,
    )
    band_spsd = _band_mean_power_spectral_density(
        spsd.values,
        fft_bin_size=freq[1] - freq[0],
        bin1_center_freq=freq[0],
        first_band_idx=0,
        last_band_idx=bands.shape[0],
        freq_table=bands,
    )
    out = xr.DataArray(
        band_spsd,
        coords={"time": spsd["time"], "freq": bands[:, 1]},
        dims=["time", "freq"],
        attrs=spsd.attrs,
    )

    return out


def convert_to_millidecade(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to millidecade spacing with base 10."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, base=10, bands_per_division=1000, use_fft_res_at_bottom=True
    )
    return out


def convert_to_third_octave(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to third octave spacing with base 2."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, base=2, bands_per_division=3, use_fft_res_at_bottom=False
    )
    return out


def convert_to_decidecade(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to decidecade spacing with base 10."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, base=10, bands_per_division=10, use_fft_res_at_bottom=False
    )
    return out


def convert_to_custom_bands(
    spsd: xr.DataArray,
    base: int,
    bands_per_division: int,
    use_fft_res_at_bottom: bool = False,
) -> xr.DataArray:
    """Convert sound pressure spectral density to custom band spacing based on specified parameters.
    Parameters:
        spsd (xr.DataArray): DataArray with frequency dimension.
        base (int): Base for the band levels, generally 10 or 2.
        bands_per_division (int): The number of bands to divide the spectrum into
            per increase by a factor of 'base'. A base of 2 and
            "bands_per_division" of 3 results in third octaves. Base 10
            and "bands_per_division" of 1000 results in millidecades.
        use_fft_res_at_bottom (bool): In some cases, like millidecades, we do not want
            to have logarithmically spaced frequency bands across the full
            spectrum, instead we have the option to have bands that are equal
            'fft_bin_size'. The switch to log spacing is made at the band that
            has a bandwidth greater than 'fft_bin_size' and such that the
            frequency space between band center frequencies is at least
            'fft_bin_size'.
    Returns:
        xr.DataArray: DataArray with frequency dimension converted to custom banded spectral density.
    """

    out = _convert_to_band_spectral_density(
        spsd=spsd,
        base=base,
        bands_per_division=bands_per_division,
        use_fft_res_at_bottom=use_fft_res_at_bottom,
    )
    return out
