"""
This submodule includes the main passive acoustics I/O functions.
`read_hydrophone` is the main function, with a number of wrapper
functions for specific manufacturers. The `export_audio` function
exists to improve audio if one is difficult to listen to.
"""

from typing import BinaryIO, Tuple, Dict, Union, Optional, Any
import io
import struct
import wave
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import wavfile


def _read_wav_metadata(f: BinaryIO) -> int:
    """
    Extracts the bit depth from a WAV file and skips over any metadata blocks
    that might be present (e.g., 'LIST' chunks).

    Parameters
    ----------
    f : BinaryIO
        An open WAV file in binary mode.

    Returns
    -------
    bits_per_sample : int
        The number of bits per sample in the WAV file (commonly 12, 16, 24, or 32).
    """
    if not isinstance(f, io.BufferedIOBase):
        raise TypeError("Expected 'f' to be a binary file object.")

    f.read(4)  # riff_key
    f.read(4)  # file_size "<I"
    f.read(4)  # wave_key
    list_key = f.read(4)

    # Skip metadata if it exists
    if "LIST" in list_key.decode():
        list_size = struct.unpack("<I", f.read(4))[0]
        f.seek(f.tell() + list_size)
    else:
        f.seek(f.tell() - 4)

    f.read(4)  # fmt_key
    fmt_size = struct.unpack("<I", f.read(4))[0]
    f.read(2)  # compression_code "<H"
    f.read(2)  # n_channels "<H"
    f.read(4)  # sample_rate "<I"
    f.read(4)  # bytes_per_sec "<I"
    f.read(2)  # block_align "<H"
    bits_per_sample = struct.unpack("<H", f.read(2))[0]
    f.seek(f.tell() + fmt_size - 16)
    return bits_per_sample


def _calculate_voltage_and_time(
    fs: int,
    raw: np.ndarray,
    bits_per_sample: int,
    peak_voltage: Union[int, float],
    start_time: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex, int]:
    """
    Normalizes the raw data from the WAV file to the appropriate voltage and
    calculates the time array based on the sampling frequency.

    Parameters
    ----------
    fs : int
        Sampling frequency of the audio data in Hertz.
    raw : numpy.ndarray
        Raw audio data extracted from the WAV file.
    bits_per_sample : int
        Number of bits per sample in the WAV file.
    peak_voltage : int or float
        Peak voltage supplied to the analog-to-digital converter (ADC) in volts.
    start_time : str, np.datetime64
        Start time of the recording in ISO 8601 format (e.g., '2024-06-06T00:00:00').

    Returns
    -------
    raw_voltage : numpy.ndarray
        Normalized voltage values corresponding to the raw audio data.
    time : pandas.DatetimeIndex
        Time index for the audio data based on the sample rate and start time.
    max_count : int
        Maximum possible count value for the given bit depth, used for normalization.
    """

    if not isinstance(fs, int):
        raise TypeError("Sampling frequency 'fs' must be an integer.")
    if not isinstance(raw, np.ndarray):
        raise TypeError("Raw audio data 'raw' must be a numpy.ndarray.")
    if not isinstance(bits_per_sample, int):
        raise TypeError("'bits_per_sample' must be an integer.")
    if not isinstance(peak_voltage, (int, float)):
        raise TypeError("'peak_voltage' must be numeric (int or float).")
    if not isinstance(start_time, (str, np.datetime64)):
        raise TypeError("'start_time' must be a string or np.datetime64.")

    length = raw.shape[0] // fs  # length of recording in seconds

    if bits_per_sample in [16, 32]:
        max_count = 2 ** (bits_per_sample - 1)
    elif bits_per_sample == 12:
        max_count = 2 ** (16 - 1) - 2**4  # 12 bit read in as 16 bit
    elif bits_per_sample == 24:
        max_count = 2 ** (32 - 1) - 2**8  # 24 bit read in as 32 bit
    else:
        raise ValueError(
            f"Unknown how to read {bits_per_sample} bit ADC."
            "Please notify MHKiT team."
        )

    # Normalize and then scale to peak voltage
    # Use 64 bit float for decimal accuracy
    raw_voltage = raw.astype(float) / max_count * peak_voltage

    # Get time

    end_time = np.datetime64(start_time) + np.timedelta64(length, "s")
    time = pd.date_range(start_time, end_time, raw.size + 1)
    return raw_voltage, time, max_count


def _process_pressure(
    raw_voltage: np.ndarray,
    peak_voltage: Union[int, float],
    max_count: int,
    sensitivity: Union[int, float],
    gain: Union[int, float],
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Converts the raw voltage data into sound pressure and calculates
    the minimum resolution and saturation levels based on the hydrophone's
    sensitivity and gain.

    Parameters
    ----------
    raw_voltage: numpy.ndarray
        The normalized voltage values corresponding to the raw data from the WAV file.
    peak_voltage: int or float
        The peak voltage supplied to the analog to digital converter (ADC) in volts.
    max_count: int
        The maximum possible count for the given bit depth, used for normalization.
    sensitivity: int or float
        The hydrophone's sensitivity in dB re 1 V/uPa, entered as a negative value.
    gain: int or float
        Amplifier gain in dB. Default is 0.

    Returns
    -------
    processed_pressure : dict
        Dictionary containing:
            - 'pressure': numpy.ndarray
                Calculated sound pressure values in Pascals (Pa).
            - 'min_res': float
                Minimum resolution in micro-Pascals (μPa).
            - 'max_sat': float
                Maximum saturation level in micro-Pascals (μPa).
    """
    if not isinstance(raw_voltage, np.ndarray):
        raise TypeError("'raw_voltage' must be a numpy.ndarray.")
    if not isinstance(peak_voltage, (int, float)):
        raise TypeError("'peak_voltage' must be numeric (int or float).")
    if not isinstance(max_count, int):
        raise TypeError("'max_count' must be an integer.")
    if not isinstance(sensitivity, (int, float)):
        raise TypeError("'sensitivity' must be numeric (int or float).")
    if not isinstance(gain, (int, float)):
        raise TypeError("'gain' must be numeric (int or float).")

    # Subtract gain
    # hydrophone with sensitivity of -177 dB and gain of -3 dB = sensitivity of -174 dB
    if gain:
        sensitivity -= gain
    # Convert calibration from dB rel 1 V/uPa into ratio
    sensitivity = 10 ** (sensitivity / 20)  # V/uPa

    # Sound pressure
    pressure = raw_voltage / sensitivity  # uPa
    pressure = pressure / 1e6  # Pa

    # Min resolution
    min_res = peak_voltage / max_count / sensitivity  # uPa
    # Pressure at which sensor is saturated
    max_sat = peak_voltage / sensitivity  # uPa

    processed_pressure = {"pressure": pressure, "min_res": min_res, "max_sat": max_sat}
    return processed_pressure


def read_hydrophone(
    filename: Union[str, Path],
    peak_voltage: Union[int, float],
    sensitivity: Optional[Union[int, float]] = None,
    gain: Union[int, float] = 0,
    start_time: str = "2024-06-06T00:00:00",
) -> xr.DataArray:
    """
    Read .wav file from a hydrophone. Returns voltage timeseries if sensitivity not
    provided, returns pressure timeseries if it is provided.

    Parameters
    ----------
    filename: str or pathlib.Path
        Input filename
    peak_voltage: int or float
        Peak voltage supplied to the analog to digital converter (ADC) in V.
        (Or 1/2 of the peak to peak voltage).
    sensitivity: int or float
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative. Default: None.
    gain: int or float
        Amplifier gain in dB re 1 V/uPa. Default 0.
    start_time: str
        Start time in the format yyyy-mm-ddTHH:MM:SS

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] or Voltage [V] indexed by time[s]
    """

    if not isinstance(filename, (str, Path)):
        raise TypeError("Filename must be a string or a pathlib.Path object.")
    if not isinstance(peak_voltage, (int, float)):
        raise TypeError("'peak_voltage' must be numeric (int or float).")
    if sensitivity is not None and not isinstance(sensitivity, (int, float)):
        raise TypeError("'sensitivity' must be numeric (int, float) or None.")
    if not isinstance(gain, (int, float)):
        raise TypeError("'gain' must be numeric (int or float).")
    if not isinstance(start_time, (str, np.datetime64)):
        raise TypeError("'start_time' must be a string or np.datetime64")

    if (sensitivity is not None) and (sensitivity > 0):
        raise ValueError(
            "Hydrophone calibrated sensitivity should be entered as a negative number."
        )

    # Read metadata from WAV file
    with open(filename, "rb") as f:
        bits_per_sample = _read_wav_metadata(f)

    # Read data using scipy (will auto drop as int16 or int32)
    fs, raw = wavfile.read(filename)

    # Calculate raw voltage and time array
    raw_voltage, time, max_count = _calculate_voltage_and_time(
        fs, raw, bits_per_sample, peak_voltage, start_time
    )

    # If sensitivity is provided, convert to sound pressure
    if sensitivity is not None:
        processed_pressure = _process_pressure(
            raw_voltage, peak_voltage, max_count, sensitivity, gain
        )

        out = xr.DataArray(
            processed_pressure["pressure"],
            coords={"time": time[:-1]},
            attrs={
                "units": "Pa",
                "sensitivity": sensitivity,
                "resolution": np.round(processed_pressure["min_res"] / 1e6, 9),
                "valid_min": np.round(
                    -processed_pressure["max_sat"] / 1e6,
                    6,
                ),
                "valid_max": np.round(processed_pressure["max_sat"] / 1e6, 6),
                "fs": fs,
                "filename": Path(filename).stem,
            },
        )

    else:
        process_volatage = {}
        # Voltage min resolution
        process_volatage["min_res"] = peak_voltage / max_count  # V
        # Voltage at which sensor is saturated
        process_volatage["max_sat"] = peak_voltage  # V

        out = xr.DataArray(
            raw_voltage,
            coords={"time": time[:-1]},
            attrs={
                "units": "V",
                "resolution": np.round(process_volatage["min_res"], 6),
                "valid_min": -1 * process_volatage["max_sat"],
                "valid_max": process_volatage["max_sat"],
                "fs": fs,
                "filename": Path(filename).stem,
            },
        )

    return out


def read_soundtrap(
    filename: str,
    sensitivity: Optional[Union[int, float]] = None,
    gain: Union[int, float] = 0,
) -> xr.DataArray:
    """
    Read .wav file from an Ocean Instruments SoundTrap hydrophone.
    Returns voltage timeseries if sensitivity not provided, returns pressure
    timeseries if it is provided.

    Parameters
    ----------
    filename : str
        Input filename.
    sensitivity : int or float, optional
        Hydrophone calibration sensitivity in dB re 1 V/μPa.
        Should be negative. Default is None.
    gain : int or float
        Amplifier gain in dB re 1 V/μPa. Default is 0.

    Returns
    -------
    out : xarray.DataArray
        Sound pressure [Pa] or Voltage [V] indexed by time[s].
    """

    if not isinstance(filename, str):
        raise TypeError("'filename' must be a string.")
    if sensitivity is not None and not isinstance(sensitivity, (int, float)):
        raise TypeError("'sensitivity' must be a numeric type (int or float) or None.")
    if not isinstance(gain, (int, float)):
        raise TypeError("'gain' must be a numeric type (int or float).")
    if sensitivity is not None and sensitivity > 0:
        raise ValueError(
            "Hydrophone calibrated sensitivity should be entered \
                          as a negative number."
        )

    # Get time from filename
    st = filename.split(".")[-2]
    start_time = (
        "20"
        + st[:2]
        + "-"
        + st[2:4]
        + "-"
        + st[4:6]
        + "T"
        + st[6:8]
        + ":"
        + st[8:10]
        + ":"
        + st[10:12]
    )

    # Soundtrap uses a peak voltage of 1 V
    out = read_hydrophone(
        filename,
        peak_voltage=1,
        sensitivity=sensitivity,
        gain=gain,
        start_time=start_time,
    )
    out.attrs["make"] = "SoundTrap"

    return out


def _read_iclisten_metadata(f: io.BufferedIOBase) -> Dict[str, Any]:
    """
    Reads the metadata from the icListen .wav file and
    returns the metadata in a dictionary.

    Parameters
    ----------
    f: io.BufferedIOBase
        Opened .wav file for reading metadata.

    Returns
    -------
    metadata: dict
        A dictionary containing metadata such as peak_voltage,
        stored_sensitivity, humidity, temperature, etc.
    """
    if not isinstance(f, io.BufferedIOBase):
        raise TypeError("'f' must be a binary file object opened in read mode.")

    def read_string(f: io.BufferedIOBase) -> str:
        """Reads a string from the file based on its size."""
        if not isinstance(f, io.BufferedIOBase):
            raise TypeError("'f' must be a binary file object opened in read mode.")

        f.read(4)
        size = struct.unpack("<I", f.read(4))[0]
        return f.read(size).decode().rstrip("\x00")

    metadata: Dict[str, Any] = {}

    # Read header keys
    riff_key = f.read(4)
    f.read(4)  # file_size_bytes "<I"
    wave_key = f.read(4)
    list_key = f.read(4)

    # Check if headers are as expected
    if riff_key != b"RIFF" or wave_key != b"WAVE" or list_key != b"LIST":
        raise ValueError("Invalid file format or missing LIST chunk in WAV file.")

    # Read metadata keys
    list_size_bytes = f.read(4)
    if len(list_size_bytes) < 4:
        raise ValueError("Unexpected end of file when reading list size.")

    info_key = f.read(4)
    if info_key != b"INFO":
        raise ValueError("Expected INFO key in metadata but got different key.")

    # Read metadata and store in the dictionary
    metadata["iart"] = read_string(f)  # Hydrophone make and SN
    metadata["iprd"] = read_string(f)  # Hydrophone model
    metadata["icrd"] = read_string(f)  # File creation date
    metadata["isft"] = read_string(f)  # Hydrophone software version
    metadata["inam"] = read_string(f)  # Original filename

    # Additional comments
    icmt_key = f.read(4)
    if icmt_key != b"ICMT":
        raise ValueError("Expected ICMT key in metadata but got different key.")
    icmt_size_bytes = f.read(4)
    if len(icmt_size_bytes) < 4:
        raise ValueError("Unexpected end of file when reading ICMT size.")
    icmt_size = struct.unpack("<I", icmt_size_bytes)[0]
    icmt_bytes = f.read(icmt_size)
    if len(icmt_bytes) < icmt_size:
        raise ValueError("Unexpected end of file when reading ICMT data.")
    icmt = icmt_bytes.decode().rstrip("\x00")

    # Parse the fields from comments and update the metadata dictionary
    fields = icmt.split(",")
    try:
        metadata["peak_voltage"] = float(fields[0].split(" ")[0])
        metadata["stored_sensitivity"] = int(fields[1].strip().split(" ")[0])
        metadata["humidity"] = fields[2].strip()
        metadata["temperature"] = fields[3].strip()
        metadata["accelerometer"] = (
            ",".join(fields[4:7]).strip() if len(fields) > 6 else None
        )
        metadata["magnetometer"] = (
            ",".join(fields[7:10]).strip() if len(fields) > 9 else None
        )
        metadata["count_at_peak_voltage"] = fields[-2].strip()
        metadata["sequence_num"] = fields[-1].strip()
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing metadata comments: {e}") from e

    # Return a dictionary with metadata
    return metadata


def read_iclisten(
    filename: str,
    sensitivity: Optional[Union[int, float]] = None,
    use_metadata: bool = True,
) -> xr.DataArray:
    """
    Read .wav file from an Ocean Sonics icListen "Smart" hydrophone.
    Returns voltage timeseries if sensitivity not provided, returns pressure
    timeseries if it is provided.

    Parameters
    ----------
    filename : str
        Input filename.
    sensitivity : int or float, optional
        Hydrophone calibration sensitivity in dB re 1 V/μPa.
        Should be negative. Default is None.
    use_metadata : bool
        If True and `sensitivity` is None, applies sensitivity value stored
        in the .wav file's LIST block. If False and `sensitivity` is None,
        a sensitivity value isn't applied.

    Returns
    -------
    out : xarray.DataArray
        Sound pressure [Pa] or Voltage [V] indexed by time[s].
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a string.")
    if sensitivity is not None and not isinstance(sensitivity, (int, float)):
        raise TypeError("'sensitivity' must be a numeric type (int or float) or None.")
    if not isinstance(use_metadata, bool):
        raise TypeError("'use_metadata' must be a boolean value.")
    if sensitivity is not None and sensitivity > 0:
        raise ValueError(
            "Hydrophone calibrated sensitivity should be entered \
            as a negative number."
        )

    # Read icListen metadata from file header
    with open(filename, "rb") as f:
        metadata = _read_iclisten_metadata(f)

    # Use stored sensitivity
    if use_metadata and sensitivity is None:
        sensitivity = metadata["stored_sensitivity"]
        if sensitivity is None:
            raise ValueError("Stored sensitivity not found in metadata.")

    # Convert metadata creation date to datetime64
    try:
        start_time = np.datetime64(metadata["icrd"])
    except ValueError as e:
        raise ValueError(f"Invalid creation date format in metadata: {e}") from e

    out = read_hydrophone(
        filename,
        peak_voltage=metadata["peak_voltage"],
        sensitivity=sensitivity,
        gain=0,
        start_time=start_time,
    )

    # Update attributes with metadata
    out.attrs.update(
        {
            "serial_num": metadata["iart"],
            "model": metadata["iprd"],
            "software_ver": metadata["isft"],
            "filename": metadata["inam"] + ".wav",
            "peak_voltage": metadata["peak_voltage"],
            "sensitivity": sensitivity,
            "humidity": metadata["humidity"],
            "temperature": metadata["temperature"],
            "accelerometer": metadata["accelerometer"],
            "magnetometer": metadata["magnetometer"],
            "count_at_peak_voltage": metadata["count_at_peak_voltage"],
            "sequence_num": metadata["sequence_num"],
        }
    )

    return out


def export_audio(
    filename: str, pressure: xr.DataArray, gain: Union[int, float] = 1
) -> None:
    """
    Creates human-scaled audio file from underwater recording.

    Parameters
    ----------
    filename : str
        Output filename for the WAV file (without extension).
    pressure : xarray.DataArray
        Sound pressure data with attributes:
            - 'values' (numpy.ndarray): Pressure data array.
            - 'sensitivity' (int or float): Sensitivity of the hydrophone in dB.
            - 'fs' (int or float): Sampling frequency in Hz.
    gain : int or float, optional
        Gain to multiply the original time series by. Default is 1.

    Returns
    -------
    None
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a string.")

    if not isinstance(pressure, xr.DataArray):
        raise TypeError("'pressure' must be an xarray.DataArray.")

    if not hasattr(pressure, "values") or not isinstance(pressure.values, np.ndarray):
        raise TypeError("'pressure.values' must be a numpy.ndarray.")

    if not hasattr(pressure, "sensitivity") or not isinstance(
        pressure.sensitivity, (int, float)
    ):
        raise TypeError("'pressure.sensitivity' must be a numeric type (int or float).")

    if not hasattr(pressure, "fs") or not isinstance(pressure.fs, (int, float)):
        raise TypeError("'pressure.fs' must be a numeric type (int or float).")

    if not isinstance(gain, (int, float)):
        raise TypeError("'gain' must be a numeric type (int or float).")

    # Convert from Pascals to UPa
    upa = pressure.values.T * 1e6
    # Change to voltage waveform
    v = upa * 10 ** (pressure.sensitivity / 20)  # in V
    # Normalize
    v = v / max(abs(v)) * gain
    # Convert to (little-endian) 16 bit integers.
    audio = (v * (2**16 - 1)).astype("<h")

    # pylint incorrectly thinks this is opening in read mode
    with wave.open(f"{filename}.wav", mode="w") as f:
        f.setnchannels(1)  # pylint: disable=no-member
        f.setsampwidth(2)  # pylint: disable=no-member
        f.setframerate(pressure.fs)  # pylint: disable=no-member
        f.writeframes(audio.tobytes())  # pylint: disable=no-member
