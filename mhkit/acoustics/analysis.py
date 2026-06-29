"""
This module contains key functions for passive acoustics analysis, designed to process
and analyze sound pressure data from .wav files in the frequency and time domains.
The functions herein build on each other, with a structured flow that facilitates the
calculation of sound pressure spectral densities and banded averages based on
input audio data.

The following functionality is provided:

1. **Type Validation**:

   - `_check_numeric`: Validates that a value is of numeric type (int or float).

2. **Frequency Validation and Warning**:

   - `_fmax_warning`: Ensures specified maximum frequency does not exceed the Nyquist frequency,
     adjusting if necessary to avoid aliasing.

3. **Shallow Water Cutoff Frequency**:

   - `minimum_frequency`: Calculates the minimum frequency cutoff based on water depth and the
     speed of sound in water and seabed materials.

4. **Calculation of Frequency Bands**:

    - `create_frequency_bands`: Generates frequency bands based on specified octave divisions,
      minimum and maximum frequency limits, and the chosen base
      (e.g., 2 for octaves, 10 for decades).

5. **Sound Pressure Spectral Density Calculation**:

    - `sound_pressure_spectral_density`: Computes the mean square sound pressure spectral density
      using FFT binning with Hanning windowing and 50% overlap.

6. **Calibration**:

   - `apply_calibration`: Applies calibration adjustments to the spectral density data using
     a sensitivity curve, filling missing values as specified.

7. **Band-Averaged Spectral Density**:

    - `_get_band_table`: Generates a table of frequency bands for logarithmically
      spaced divisions with optional linear spacing at lower frequencies.
    - `_band_power_spectral_density_v3`: Pre-computes bin indices and weights for band averaging.
    - `_band_mean_power_spectral_density_v2`: Computes the mean power spectral density
      within specified bands.
    - `_convert_to_band_spectral_density`: Generic function to convert spectral density
      to custom banded spectral densities.
    - `convert_to_millidecade`, `convert_to_decidecade`, `convert_to_third_octave`:
      Convenience functions to convert spectral density to millidecade, decidecade,
      and third-octave banded spectral densities, respectively.
    - `convert_to_custom_bands`: Convert spectral density to custom band spacing with
      user-specified parameters.

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
        np.log10(fmax),
        step=np.log10(bandwidth),
    )
    band["lower_limit"] = band["center_freq"] / half_bandwidth
    band["upper_limit"] = band["center_freq"] * half_bandwidth
    octave_bins = np.append(band["lower_limit"], band["upper_limit"][-1])

    return octave_bins, band


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
    freq: np.ndarray,
    bands_per_division: int,
    base: int,
    use_fft_res_at_bottom: bool,
):
    """
    Returns a three column array with the start, center, stop frequencies for
    logarithmically spaced frequency bands such as millidecades, decidecades,
    or third octaves base 2. These tables are passed to
    `band_squared_sound_pressure` to convert square spectra to band
    levels. The squared pressure can be converted to power spectral
    density by dividing by the band_widths.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequency values in Hz from the FFT.
    bands_per_division : int
        The number of bands to divide the spectrum into per increase by
        a factor of 'base'. A base of 2 and bands_per_division of 3
        results in third octaves base 2. Base 10 and bands_per_division
        of 1000 results in millidecades.
    base : int
        Base for the band levels, generally 10 or 2.
    use_fft_res_at_bottom : bool
        In some cases, like millidecades, we do not want to have
        logarithmically spaced frequency bands across the full spectrum.
        When True, bands at lower frequencies use linear spacing (equal
        to the FFT bin size), and transition to log spacing at the band
        where the bandwidth exceeds the FFT bin size and the frequency
        space between band center frequencies is at least the FFT bin size.

    Returns
    -------
    bands : np.ndarray
        Three column array where column 1 is the lowest frequency of the
        band, column 2 is the center frequency, and column 3 is the highest
        frequency.

    Originally created by Bruce Martin, JASCO Applied Sciences, Feb 2020:

    Martin, S. Bruce, et al. "Hybrid millidecade spectra: A practical format
        for exchange of long-term ambient sound data." JASA Express Letters
        1.1 (2021).

    Converted to Python and refactored by MHKiT Team, June 2026.
    """

    band_count = 0
    linear_bin_count = 0
    log_bin_count = 0
    max_linear_bin_hz = 0

    fft_bin_size = freq[1] - freq[0]
    bin1_center_freq = freq[0]
    max_freq = freq[-1]

    # Generate the log-spaced bands
    _, band_dict = create_frequency_bands(
        octave=bands_per_division,
        base=base,
        fmin=freq[0],
        fmax=freq[-1],
    )

    # For millidecades and similar
    if use_fft_res_at_bottom:
        # Find the first band where the bandwidth is greater than the FFT bin size
        # and the frequency space between band center frequencies is at least the FFT bin size
        band_count = np.where(np.diff(band_dict["center_freq"]) >= fft_bin_size)[0][1]
        center_freq = band_dict["center_freq"][band_count]

        # Now keep counting until the difference between the log spaced
        # center frequency and new frequency is greater than .025
        max_linear_bin_hz = np.ceil(center_freq / fft_bin_size)
        while (max_linear_bin_hz * fft_bin_size - center_freq) > 0:
            band_count += 1
            max_linear_bin_hz += 1
            center_freq = bin1_center_freq * base ** (band_count / bands_per_division)

        if (fft_bin_size * max_linear_bin_hz) > max_freq:
            max_linear_bin_hz = max_freq / fft_bin_size + 1

        linear_bin_count = int(max_linear_bin_hz / fft_bin_size - 1)

    # Count the log space frequencies
    log_bin_count = len(np.arange(band_count, band_dict["center_freq"].size, 1))

    # Generate the linear frequencies
    bands = np.zeros(((linear_bin_count + log_bin_count), 3))
    if linear_bin_count:
        bands[:linear_bin_count, 1] = (
            np.arange(0, linear_bin_count * fft_bin_size, fft_bin_size)
            + bin1_center_freq
        )
        bands[:linear_bin_count, 0] = (
            np.arange(
                -fft_bin_size / 2,
                linear_bin_count * fft_bin_size - fft_bin_size / 2,
                fft_bin_size,
            )
            + bin1_center_freq
        )
        bands[:linear_bin_count, 2] = (
            np.arange(
                fft_bin_size / 2,
                (linear_bin_count + 1) * fft_bin_size - fft_bin_size / 2,
                fft_bin_size,
            )
            + bin1_center_freq
        )

    # Trim log spaced bands to start the first band after the linear bands if they exist
    idx = np.where(band_dict["center_freq"] > max_linear_bin_hz)[0]
    bands[linear_bin_count:, 1] = band_dict["center_freq"][idx]
    bands[linear_bin_count:, 0] = band_dict["lower_limit"][idx]
    bands[linear_bin_count:, 2] = band_dict["upper_limit"][idx]

    if log_bin_count > 0:
        bands[-1, 2] = max_freq

    return bands


def _band_power_spectral_density_v3(freq_fft, freq_table):
    """
    Sums squared sound pressures to determine in-band totals.
    The band edges are normally obtained from a call to ``get_band_table``.

    Parameters
    ----------
    freq_fft : np.ndarray
        1D array of center frequencies produced by the FFT call [Hz].
    freq_table : np.ndarray
        Nx3 array of band edges where column 0 is the lower limit [Hz],
        column 1 is the center frequency [Hz], and column 2 is the upper
        limit [Hz].

    Returns
    -------
    full_pts : dict
        Dictionary where keys are band indices and values are arrays of FFT bin
        indices that fall fully within the band.
    partial_pts : dict
        Dictionary where keys are band indices and values are arrays of FFT bin
        indices that partially overlap the band.
    weights : dict
        Dictionary where keys are band indices and values are arrays of weights
        for the partial bins, proportional to the fraction of the bin that falls
        within the band.

    Notes
    -----
    Full bins (entirely within a band) are summed
    directly; partial bins (overlapping a band edge) are weighted
    proportionally to the fraction of the bin that falls within the band.
    The result is then divided by each band's bandwidth to return mean PSD.

    Original code by Bruce Martin, JASCO Applied Sciences, Feb 2020:

    Martin, S. Bruce, et al. "Hybrid millidecade spectra: A practical format
        for exchange of long-term ambient sound data." JASA Express Letters
        1.1 (2021).

    Updated to use FFT-generated frequencies as input to handle time-resolved
    spectral processing with a fundamental FFT frequency of 10 Hz by
    Brian Polayge, University of Washington, 2025.

    Converted to Python.
    Split into two functions to pre-compute bin indices and weights in
    ``_band_power_spectral_density_v3``, then compute band averaged PSD
    in ``band_mean_power_spectral_density_v2``.
    MHKiT Team, June 2026.
    """

    fft_bin_size = freq_fft[1] - freq_fft[0]

    # Pre-compute full and partial bin indices for each band.
    # This is band-invariant across time windows, so it is done once.
    full_pts = {}
    partial_pts = {}

    for j in range(freq_table.shape[0]):
        f_lo = freq_table[j, 0]
        f_hi = freq_table[j, 2]

        # FFT bins whose extent falls entirely within the frequency band
        full_mask = (freq_fft - fft_bin_size / 2 >= f_lo) & (
            freq_fft + fft_bin_size / 2 <= f_hi
        )
        full_pts[j] = np.where(full_mask)[0]

        # FFT bins that overlap the frequency band at all (full or partial)
        overlap_mask = ((freq_fft >= f_lo) & (freq_fft <= f_hi)) | (
            (freq_fft + fft_bin_size / 2 > f_lo) & (freq_fft - fft_bin_size / 2 < f_hi)
        )

        # Partial bins = overlapping minus fully-contained
        partial_pts[j] = np.setdiff1d(np.where(overlap_mask)[0], full_pts[j])

    # Compute fractional weights for partial bins
    weights = {}
    for j in range(freq_table.shape[0]):
        weights[j] = np.zeros(partial_pts[j].size)

        for k, idx in enumerate(partial_pts[j]):
            bin_lo = freq_fft[idx] - fft_bin_size / 2
            bin_hi = freq_fft[idx] + fft_bin_size / 2

            if bin_lo < freq_table[j, 0]:
                # Bin extends below the band's lower edge
                weights[j][k] = (
                    fft_bin_size - (freq_table[j, 0] - bin_lo)
                ) / fft_bin_size

            elif bin_hi > freq_table[j, 2]:
                # Bin extends above the band's upper edge
                weights[j][k] = (
                    fft_bin_size - (bin_hi - freq_table[j, 2])
                ) / fft_bin_size

    return full_pts, partial_pts, weights


def _band_mean_power_spectral_density_v2(
    input_spsd, freq_table, full_pts, partial_pts, weights
):
    """
    Sums squared sound pressures to determine in-band totals, then divides by
    the band widths to return mean power spectral density.
    The band edges are normally obtained from a call to ``get_band_table`` and
    the full and partial bin indices and weights are obtained from a call to
    ``band_power_spectral_density_v3``.

    Parameters
    ----------
    input_spsd : np.ndarray
        2D array of sound pressure spectral density values with dimensions (time, freq).
    freq_table : np.ndarray
        Nx3 array of band edges where column 0 is the lower limit [Hz],
        column 1 is the center frequency [Hz], and column 2 is the upper
        limit [Hz].
    full_pts : dict
        Dictionary where keys are band indices and values are arrays of FFT bin
        indices that fall fully within the band.
    partial_pts : dict
        Dictionary where keys are band indices and values are arrays of FFT bin
        indices that partially overlap the band.
    weights : dict
        Dictionary where keys are band indices and values are arrays of weights
        for the partial bins, proportional to the fraction of the bin that falls
        within the band.

    Returns
    -------
    out_spsd : np.ndarray
        2D array of band-averaged sound pressure spectral density values with dimensions
        (time, band), where 'band' corresponds to the center frequencies in 'freq_table'.

    Notes
    -----
    Full bins (entirely within a band) are summed
    directly; partial bins (overlapping a band edge) are weighted
    proportionally to the fraction of the bin that falls within the band.
    The result is then divided by each band's bandwidth to return mean PSD.

    Original code by Bruce Martin, JASCO Applied Sciences, Feb 2020:

    Martin, S. Bruce, et al. "Hybrid millidecade spectra: A practical format
        for exchange of long-term ambient sound data." JASA Express Letters
        1.1 (2021).

    Updated to use FFT-generated frequencies as input to handle time-resolved
    spectral processing with a fundamental FFT frequency of 10 Hz by
    Brian Polayge, University of Washington, 2025.

    Converted to Python.
    Split into two functions to pre-compute bin indices and weights in
    ``_band_power_spectral_density_v3``, then compute band averaged PSD
    in ``band_mean_power_spectral_density_v2``.
    MHKiT Team, June 2026.
    """

    fft_bin_size = input_spsd["freq"].values[1] - input_spsd["freq"].values[0]
    input_spsd = input_spsd.values
    out_spsd = np.zeros((input_spsd.shape[0], freq_table.shape[0]))

    # Accumulate band-squared pressure; vectorised over the time (row) axis
    for j in range(freq_table.shape[0]):
        # Contribution from fully-contained FFT bins
        if len(full_pts[j]) > 0:
            out_spsd[:, j] = np.sum(input_spsd[:, full_pts[j]], axis=1) * fft_bin_size

        # Contribution from partial FFT bins
        if len(partial_pts[j]) > 0:
            out_spsd[:, j] += (
                np.sum(
                    input_spsd[:, partial_pts[j]] * weights[j][np.newaxis, :], axis=1
                )
                * fft_bin_size
            )

    # Take means
    band_widths = freq_table[:, 2] - freq_table[:, 0]
    out_spsd /= band_widths

    return out_spsd


def _convert_to_band_spectral_density(
    spsd: xr.DataArray, bands_per_division: int, base: int, use_fft_res_at_bottom: bool
) -> xr.DataArray:
    """
    Convert sound pressure spectral density to banded spectral density based on
    the specified base and bands per division.

    Parameters
    ----------
    spsd: xr.DataArray
        DataArray with frequency dimension.
    bands_per_division: int
        The number of bands to divide the spectrum into per increase by a factor
        of 'base'. A base of 2 and "bands_per_division" of 3 results in third
        octaves base 2. Base 10 and "bands_per_division" of 1000 results in
        millidecades.
    base: int
        Base for the band levels, generally 10 or 2.
    use_fft_res_at_bottom: bool
        In some cases, like millidecades, we do not want to have logarithmically
        spaced frequency bands across the full spectrum, instead we have the option
        to have bands that are equal 'fft_bin_size'. The switch to log spacing is
        made at the band that has a bandwidth greater than 'fft_bin_size' and such
        that the frequency space between band center frequencies is at least
        'fft_bin_size'.

    Returns
    -------
    xr.DataArray
        DataArray with frequency dimension converted to banded spectral density.
    """

    # Get bands
    bands = _get_band_table(
        freq=spsd["freq"].values,
        bands_per_division=bands_per_division,
        base=base,
        use_fft_res_at_bottom=use_fft_res_at_bottom,
    )
    full_pts, partial_pts, weights = _band_power_spectral_density_v3(
        freq_fft=spsd["freq"].values,
        freq_table=bands,
    )
    band_spsd = _band_mean_power_spectral_density_v2(
        input_spsd=spsd,
        freq_table=bands,
        full_pts=full_pts,
        partial_pts=partial_pts,
        weights=weights,
    )
    out = xr.DataArray(
        band_spsd,
        coords={"time": spsd["time"], "freq": bands[:, 1]},
        dims=["time", "freq"],
        attrs=spsd.attrs,
    )

    return out


def convert_to_millidecade(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to millidecade spacing."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, bands_per_division=1000, base=10, use_fft_res_at_bottom=True
    )
    return out


def convert_to_third_octave(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to third octave spacing."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, bands_per_division=3, base=2, use_fft_res_at_bottom=False
    )
    return out


def convert_to_decidecade(spsd: xr.DataArray) -> xr.DataArray:
    """Convert sound pressure spectral density to decidecade spacing."""

    out = _convert_to_band_spectral_density(
        spsd=spsd, bands_per_division=10, base=10, use_fft_res_at_bottom=False
    )
    return out


def convert_to_custom_bands(
    spsd: xr.DataArray,
    bands_per_division: int,
    base: int,
    use_fft_res_at_bottom: bool = False,
) -> xr.DataArray:
    """
    Convert sound pressure spectral density to custom band spacing based on specified
    parameters.

    Parameters
    ----------
    spsd (xr.DataArray): DataArray with frequency dimension.
    bands_per_division (int): The number of bands to divide the spectrum into
        per increase by a factor of 'base'. A base of 2 and
        "bands_per_division" of 3 results in third octaves. Base 10
        and "bands_per_division" of 1000 results in millidecades.
    base (int): Base for the band levels, generally 10 or 2.
    use_fft_res_at_bottom (bool): In some cases, like millidecades, we do not want
        to have logarithmically spaced frequency bands across the full
        spectrum, instead we have the option to have bands that are equal
        'fft_bin_size'. The switch to log spacing is made at the band that
        has a bandwidth greater than 'fft_bin_size' and such that the
        frequency space between band center frequencies is at least
        'fft_bin_size'.

    Returns
    -------
    xr.DataArray: DataArray with frequency dimension converted to custom banded spectral
        density.
    """

    out = _convert_to_band_spectral_density(
        spsd=spsd,
        bands_per_division=bands_per_division,
        base=base,
        use_fft_res_at_bottom=use_fft_res_at_bottom,
    )
    return out
