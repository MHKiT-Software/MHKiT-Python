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

    with open(filename, "rb") as f:
        f.read(4)  # riff_key
        f.read(4)  # file_size "<I"
        f.read(4)  # wave_key
        list_key = f.read(4)

        # Skip metadata if it exits (don't know how to parse)
        if "LIST" in list_key.decode():
            list_size = struct.unpack("<I", f.read(4))[0]
            f.seek(f.tell() + list_size)
        else:
            f.seek(f.tell() - 4)

        # Read this to get the bits/sample
        f.read(4)  # fmt_key
        fmt_size = struct.unpack("<I", f.read(4))[0]
        f.read(2)  # compression_code "<H"
        f.read(2)  # n_channels "<H"
        f.read(4)  # sample_rate "<I"
        f.read(4)  # bytes_per_sec "<I"
        f.read(2)  # block_align "<H"
        bits_per_sample = struct.unpack("<H", f.read(2))[0]
        f.seek(f.tell() + fmt_size - 16)

    # Read data using scipy cause that's easier (will auto drop as int16 or int32)
    fs, raw = wavfile.read(filename)
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

    if sensitivity is not None:
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

        out = xr.DataArray(
            pressure,
            coords={"time": time[:-1]},
            attrs={
                "units": "Pa",
                "sensitivity": sensitivity,
                "resolution": np.round(min_res / 1e6, 9),
                "valid_min": np.round(
                    -max_sat / 1e6,
                    6,
                ),
                "valid_max": np.round(max_sat / 1e6, 6),
                "fs": fs,
                "filename": Path(filename).stem,
            },
        )

    else:
        # Voltage min resolution
        min_res = peak_voltage / max_count  # V
        # Voltage at which sensor is saturated
        max_sat = peak_voltage  # V

        out = xr.DataArray(
            raw_voltage,
            coords={"time": time[:-1]},
            attrs={
                "units": "V",
                "resolution": np.round(min_res, 6),
                "valid_min": -max_sat,
                "valid_max": max_sat,
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

    # Read icListen metadata from file header
    with open(filename, "rb") as f:
        # Read header keys
        f.read(4)  # riff_key
        f.read(4)  # file_size "<I"
        f.read(4)  # wave_key
        f.read(4)  # list_key

        # Read metadata keys
        f.read(4)  # list_size "<I"
        f.read(4)  # info_key

        # Hydrophone make and SN
        f.read(4)  # iart_key
        iart_size = struct.unpack("<I", f.read(4))[0]
        iart = f.read(iart_size).decode().rstrip("\x00")

        # Hydrophone model
        f.read(4)  # iprd_key
        iprd_size = struct.unpack("<I", f.read(4))[0]
        iprd = f.read(iprd_size).decode().rstrip("\x00")

        # File creation date
        f.read(4)  # icrd_key
        icrd_size = struct.unpack("<I", f.read(4))[0]
        icrd = f.read(icrd_size).decode().rstrip("\x00")

        # Hydrophone make and software version
        f.read(4)  # isft_key
        isft_size = struct.unpack("<I", f.read(4))[0]
        isft = f.read(isft_size).decode().rstrip("\x00")

        # Original filename
        f.read(4)  # inam_key
        inam_size = struct.unpack("<I", f.read(4))[0]
        inam = f.read(inam_size).decode().rstrip("\x00")

        # Additional comments
        f.read(4)  # icmt_key
        icmt_size = struct.unpack("<I", f.read(4))[0]
        icmt = f.read(icmt_size).decode().rstrip("\x00")

        fields = icmt.split(",")
        peak_voltage = fields[0]
        stored_sensitivity = fields[1].lstrip()
        humidity = fields[2].lstrip()
        temp = fields[3].lstrip()
        if len(fields) > 6:
            accel = ",".join(fields[4:7]).lstrip()
            mag = ",".join(fields[7:10]).lstrip()
        else:
            accel = []
            mag = []
        count_at_peak_voltage = fields[-2].lstrip()
        n_sequence = fields[-1].lstrip()

    peak_voltage = float(peak_voltage.split(" ")[0])
    stored_sensitivity = int(stored_sensitivity.split(" ")[0])

    # Use stored sensitivity
    if use_metadata and (sensitivity is None):
        sensitivity = stored_sensitivity

    out = read_hydrophone(
        filename,
        peak_voltage=peak_voltage,
        sensitivity=sensitivity,
        gain=0,
        start_time=np.datetime64(icrd),
    )

    out.attrs.update(
        {
            "serial_num": iart,
            "model": iprd,
            "software_ver": isft,
            "filename": inam + ".wav",
            "peak_voltage": peak_voltage,
            "sensitivity": sensitivity,
            "humidity": humidity,
            "temperature": temp,
            "accelerometer": accel,
            "magnetometer": mag,
            "count_at_peak_voltage": count_at_peak_voltage,
            "sequence_num": n_sequence,
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

    with wave.open(f"{filename}.wav", "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(pressure.fs)
        f.writeframes(audio.tobytes())
