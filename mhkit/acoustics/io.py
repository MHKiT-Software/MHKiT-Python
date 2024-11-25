"""
This submodule provides input/output functions for passive acoustics data,
focusing on hydrophone recordings stored in WAV files. The main functionality
includes reading and processing hydrophone data from various manufacturers 
and exporting audio files for easy playback and analysis.

Supported Hydrophone Models
---------------------------
- **SoundTrap** (Ocean Instruments)
- **icListen** (Ocean Sonics)

Functions Overview
------------------

1. **Data Reading**:

   - `read_hydrophone`: Main function to read a WAV file from a hydrophone and 
     convert it to either a voltage or pressure time series, depending on the 
     availability of sensitivity data.

   - `read_soundtrap`: Wrapper for reading Ocean Instruments SoundTrap hydrophone 
     files, automatically using appropriate metadata.

   - `read_iclisten`: Wrapper for reading Ocean Sonics icListen hydrophone files, 
     including metadata processing to apply hydrophone sensitivity for direct 
     sound pressure calculation.

2. **Audio Export**:

   - `export_audio`: Converts processed sound pressure data back into a WAV file 
     format, with optional gain adjustment to improve playback quality.

3. **Data Extraction**:

   - `_read_wav_metadata`: Extracts metadata from a WAV file, including bit depth 
     and other header information.

   - `_calculate_voltage_and_time`: Converts raw WAV data into voltage values and 
     generates a time index based on the sampling frequency.
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


def _read_wav_metadata(f: BinaryIO) -> dict:
    """
    Extracts the bit depth from a WAV file and skips over any metadata blocks
    that might be present (e.g., 'LIST' chunks).

    Parameters
    ----------
    f : BinaryIO
        An open WAV file in binary mode.

    Returns
    -------
    header : dict
        Dictionary containing .wav file's header data
    """

    header = {}
    f.read(4)  # riff_key
    header["filesize"] = struct.unpack("<I", f.read(4))[0]
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
    header["compression_code"] = struct.unpack("<H", f.read(2))[0]
    header["n_channels"] = struct.unpack("<H", f.read(2))[0]
    header["sample_rate"] = struct.unpack("<I", f.read(4))[0]
    header["bytes_per_sec"] = struct.unpack("<I", f.read(4))[0]
    header["block_align"] = struct.unpack("<H", f.read(2))[0]
    header["bits_per_sample"] = struct.unpack("<H", f.read(2))[0]
    f.seek(f.tell() + fmt_size - 16)

    return header


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
        raise IOError(
            f"Unknown how to read {bits_per_sample} bit ADC."
            "Please notify MHKiT team."
        )

    # Normalize and then scale to peak voltage
    # Use 64 bit float for decimal accuracy
    raw_voltage = raw.astype(float) / max_count * peak_voltage

    # Get time
    end_time = np.datetime64(start_time) + np.timedelta64(length * 1000, "ms")
    time = pd.date_range(start_time, end_time, raw.size + 1)

    return raw_voltage, time, max_count


def read_hydrophone(
    filename: Union[str, Path],
    peak_voltage: Union[int, float],
    sensitivity: Optional[Union[int, float]] = None,
    gain: Union[int, float] = 0,
    start_time: str = "2024-01-01T00:00:00",
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
        header = _read_wav_metadata(f)

    # Read data using scipy (will auto drop as int16 or int32)
    fs, raw = wavfile.read(filename)

    # Calculate raw voltage and time array
    raw_voltage, time, max_count = _calculate_voltage_and_time(
        fs, raw, header["bits_per_sample"], peak_voltage, start_time
    )

    # If sensitivity is provided, convert to sound pressure
    if sensitivity is not None:
        # Subtract gain
        # Hydrophone with sensitivity of -177 dB and gain of -3 dB = sensitivity of -174 dB
        if gain:
            sensitivity -= gain
        # Convert calibration from dB rel 1 V/uPa into ratio
        sensitivity = 10 ** (sensitivity / 20)  # V/uPa

        # Sound pressure
        pressure = raw_voltage / sensitivity  # uPa
        pressure = pressure / 1e6  # Pa

        out = xr.DataArray(
            pressure,
            coords={"time": time[:-1]},
            attrs={
                "units": "Pa",
                "sensitivity": np.round(sensitivity, 12),
                # Pressure min resolution
                "resolution": np.round(peak_voltage / max_count / sensitivity / 1e6, 9),
                # Minimum pressure sensor can read
                "valid_min": np.round(-peak_voltage / sensitivity / 1e6, 6),
                # Pressure at which sensor is saturated
                "valid_max": np.round(peak_voltage / sensitivity / 1e6, 6),
                "fs": fs,
                "filename": Path(filename).stem,
            },
        )
    else:
        out = xr.DataArray(
            raw_voltage,
            coords={"time": time[:-1]},
            attrs={
                "units": "V",
                # Voltage min resolution
                "resolution": np.round(peak_voltage / max_count, 6),
                # Minimum voltage sensor can read
                "valid_min": -peak_voltage,
                # Voltage at which sensor is saturated
                "valid_max": peak_voltage,
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

    def read_string(f: io.BufferedIOBase) -> dict:
        """Reads a string from the file based on its size."""
        key = f.read(4).decode().lower()  # skip 4 bytes to bypass key name
        item = struct.unpack("<I", f.read(4))[0]
        return {key: f.read(item).decode().rstrip("\x00")}

    metadata: Dict[str, Any] = {}

    # Read header keys
    riff_key = f.read(4)
    metadata["filesize"] = struct.unpack("<I", f.read(4))[0]
    wave_key = f.read(4)
    # Check if headers are as expected
    if riff_key != b"RIFF" or wave_key != b"WAVE":
        raise IOError("Invalid file format or file corrupted.")

    # Read metadata keys
    list_key = f.read(4)
    if list_key != b"LIST":
        raise KeyError("Missing LIST chunk in WAV file.")
    list_size_bytes = f.read(4)
    if len(list_size_bytes) < 4:
        raise EOFError("Unexpected end of file when reading list size.")
    info_key = f.read(4)
    if info_key != b"INFO":
        raise KeyError("Expected INFO key in metadata but got different key.")

    # Read metadata and store in the dictionary
    metadata.update(read_string(f))  # Hydrophone make and SN
    metadata.update(read_string(f))  # Hydrophone model
    metadata.update(read_string(f))  # File creation date
    metadata.update(read_string(f))  # Hydrophone software version
    metadata.update(read_string(f))  # Original filename

    # Additional comments
    icmt_key = f.read(4)
    if icmt_key != b"ICMT":
        raise KeyError("Expected ICMT key in metadata but got different key.")
    icmt_size_bytes = f.read(4)
    if len(icmt_size_bytes) < 4:
        raise EOFError("Unexpected end of file when reading ICMT size.")
    icmt_size = struct.unpack("<I", icmt_size_bytes)[0]
    icmt_bytes = f.read(icmt_size)
    if len(icmt_bytes) < icmt_size:
        raise EOFError("Unexpected end of file when reading ICMT data.")
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

    if not isinstance(use_metadata, bool):
        raise TypeError("'use_metadata' must be a boolean value.")

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
