"""
This submodule includes the main passive acoustics I/O functions.
`read_hydrophone` is the main function, with a number of wrapper
functions for specific manufacturers. The `export_audio` function
exists to improve audio if one is difficult to listen to.
"""

import struct
import wave
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import wavfile


# Helper function to read WAV file and extract metadata
def read_wav_metadata(f):
    """
    Extracts WAV file metadata, such as bit depth, and skips any metadata blocks
    that might be present in the file.

    Parameters
    ----------
    f: file object
        Opened WAV file in binary mode.

    Returns
    -------
    bits_per_sample: int
        The number of bits per sample in the WAV file. Typical values are 12, 16, 24, or 32.
    """
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


# Helper function to calculate normalization and time range
def calculate_voltage_and_time(fs, raw, bits_per_sample, peak_voltage, start_time):
    """
    Normalizes the raw data from the WAV file to the appropriate voltage and
    calculates the time array based on the sampling frequency.

    Parameters
    ----------
    raw: numpy.ndarray
        The raw data extracted from the WAV file, typically int16 or int32.
    bits_per_sample: int
        The number of bits per sample in the WAV file (e.g., 16, 24, 32 bits).
    peak_voltage: float
        The peak voltage supplied to the analog to digital converter (ADC) in volts.
    length: int
        The duration of the recording in seconds, calculated as the number of samples
        divided by the sample rate.

    Returns
    -------
    raw_voltage: numpy.ndarray
        The normalized voltage values corresponding to the raw data from the WAV file.
    time: pandas.DatetimeIndex
        A time series index for the WAV data, based on the sample rate and start time.
    max_count: int
        The maximum possible count for the given bit depth, used for normalization.
    """

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


# Helper function to process sensitivity and pressure
def process_pressure(raw_voltage, peak_voltage, max_count, sensitivity, gain):
    """
    Converts the raw voltage data into sound pressure and calculates
    the minimum resolution and saturation levels based on the hydrophone's
    sensitivity and gain.

    Parameters
    ----------
    raw_voltage: numpy.ndarray
        The normalized voltage values corresponding to the raw data from the WAV file.
    peak_voltage: float
        The peak voltage supplied to the analog to digital converter (ADC) in volts.
    max_count: int
        The maximum possible count for the given bit depth, used for normalization.
    sensitivity: float
        The hydrophone's sensitivity in dB re 1 V/uPa, entered as a negative value.
    gain: float
        Amplifier gain in dB. Default is 0.

    Returns
    -------
    pressure: numpy.ndarray
        The calculated sound pressure values in Pascals (Pa).
    min_res: float
        The minimum resolution in micro-Pascals (uPa) for the hydrophone.
    max_sat: float
        The maximum saturation level in micro-Pascals (uPa) for the hydrophone.
    """
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
    filename,
    peak_voltage=None,
    sensitivity=None,
    gain=0,
    start_time="2024-06-06T00:00:00",
):
    """
    Read .wav file from a hydrophone. Returns voltage timeseries if sensitivity not
    provided, returns pressure timeseries if it is provided.

    Parameters
    ----------
    filename: str or pathlib.Path
        Input filename
    peak_voltage: numeric
        Peak voltage supplied to the analog to digital converter (ADC) in V.
        (Or 1/2 of the peak to peak voltage).
    sensitivity: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative. Default: None.
    gain: numeric
        Amplifier gain in dB re 1 V/uPa. Default 0.
    start_time: str
        Start time in the format yyyy-mm-ddTHH:MM:SS

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] or Voltage [V] indexed by time[s]
    """

    if (not isinstance(filename, str)) and (not isinstance(filename.as_posix(), str)):
        raise TypeError("Filename should be a string.")
    if peak_voltage is None:
        raise ValueError(
            "Please provide the peak voltage of the hydrophone's ADC `peak_voltage`."
        )
    if (sensitivity is not None) and (sensitivity > 0):
        raise ValueError(
            "Hydrophone calibrated sensitivity should be entered as a negative number."
        )

    # Read metadata from WAV file
    with open(filename, "rb") as f:
        bits_per_sample = read_wav_metadata(f)

    # Read data using scipy (will auto drop as int16 or int32)
    fs, raw = wavfile.read(filename)

    # Calculate raw voltage and time array
    raw_voltage, time, max_count = calculate_voltage_and_time(
        fs, raw, bits_per_sample, peak_voltage, start_time
    )

    # If sensitivity is provided, convert to sound pressure
    if sensitivity is not None:
        processed_pressure = process_pressure(
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


def read_soundtrap(filename, sensitivity=None, gain=0):
    """
    Read .wav file from an Ocean Instruments SoundTrap hydrophone.
    Returns voltage timeseries if sensitivity not provided, returns pressure
    timeseries if it is provided.

    Parameters
    ----------
    filename: string
        Input filename
    sensitivity: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative.
    gain: numeric
        Amplifier gain in dB re 1 V/uPa. Default 0.

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] or Voltage [V] indexed by time[s]
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


def read_iclisten(filename, sensitivity=None, use_metadata=True):
    """
    Read .wav file from an Ocean Sonics icListen "Smart" hydrophone.
    Returns voltage timeseries if sensitivity not provided, returns pressure
    timeseries if it is provided.

    Parameters
    ----------
    filename: string
        Input filename
    sensitivity: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative. Default: None.
    use_metadata: bool
        If True and `sensitivity` = None, applies sensitivity value stored in .wav file LIST block.
        If False and `sensitivity` = None, a sensitivity value isn't applied

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] or [V] indexed by time[s]
    """

    def read_iclisten_metadata(f):
        """
        Reads the metadata from the icListen .wav file LIST block and
        returns the metadata in a dictionary.

        Parameters
        ----------
        f: file object
            Opened .wav file for reading metadata.

        Returns
        -------
        metadata: dict
            A dictionary containing metadata such as peak_voltage,
            stored_sensitivity, humidity, temperature, etc.
        """

        def read_string(f):
            """Reads a string from the file based on its size."""
            f.read(4)
            size = struct.unpack("<I", f.read(4))[0]
            return f.read(size).decode().rstrip("\x00")

        metadata = {}

        # Read header keys
        f.read(4)  # riff_key
        f.read(4)  # file_size "<I"
        f.read(4)  # wave_key
        f.read(4)  # list_key

        # Read metadata keys
        f.read(4)  # list_size "<I"
        f.read(4)  # info_key

        # Read metadata and store in the dictionary
        metadata["iart"] = read_string(f)  # Hydrophone make and SN
        metadata["iprd"] = read_string(f)  # Hydrophone model
        metadata["icrd"] = read_string(f)  # File creation date
        metadata["isft"] = read_string(f)  # Hydrophone software version
        metadata["inam"] = read_string(f)  # Original filename

        # Additional comments
        f.read(4)  # icmt_key
        icmt_size = struct.unpack("<I", f.read(4))[0]
        icmt = f.read(icmt_size).decode().rstrip("\x00")

        # Parse the fields from comments and update the metadata dictionary
        fields = icmt.split(",")
        metadata["peak_voltage"] = float(fields[0].split(" ")[0])
        metadata["stored_sensitivity"] = int(fields[1].lstrip().split(" ")[0])
        metadata["humidity"] = fields[2].lstrip()
        metadata["temperature"] = fields[3].lstrip()
        metadata["accelerometer"] = (
            ",".join(fields[4:7]).lstrip() if len(fields) > 6 else []
        )
        metadata["magnetometer"] = (
            ",".join(fields[7:10]).lstrip() if len(fields) > 6 else []
        )
        metadata["count_at_peak_voltage"] = fields[-2].lstrip()
        metadata["sequence_num"] = fields[-1].lstrip()

        # Return a dictionary with metadata
        return metadata

    # Read icListen metadata from file header
    with open(filename, "rb") as f:
        metadata = read_iclisten_metadata(f)

    # Use stored sensitivity
    if use_metadata and (sensitivity is None):
        sensitivity = metadata["stored_sensitivity"]

    # Call the read_hydrophone function with appropriate parameters
    out = read_hydrophone(
        filename,
        peak_voltage=metadata["peak_voltage"],
        sensitivity=sensitivity,
        gain=0,
        start_time=np.datetime64(metadata["icrd"]),
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


def export_audio(filename, pressure, gain=1):
    """
    Creates human-scaled audio file from underwater recording.

    Parameters
    ----------
    filename: string
        Output filename for the WAV file (without extension).
    pressure: object
        Sound pressure object with attributes 'values' (numpy array of pressure data),
        'sensitivity' (sensitivity of the hydrophone in dB), and 'fs' (sampling frequency in Hz).
    gain: numeric, optional
        Gain to multiply original time series by. Default is 1.
    """

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
