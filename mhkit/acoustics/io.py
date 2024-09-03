import struct
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import wave
from scipy.io import wavfile


def read_hydrophone(
    filename, peak_V=None, sensitivity=None, gain=0, start_time="2024-06-06T00:00:00"
):
    """
    Read .wav file from a hydrophone. Returns voltage timeseries if sensitivity not
    provided, returns pressure timeseries if it is provided.

    Parameters
    ----------
    filename: string
        Input filename
    peak_V: numeric
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
    if not isinstance(filename, str):
        raise TypeError("Filename should be a string.")
    if peak_V is None:
        raise ValueError(
            "Please provide the peak voltage of the hydrophone's ADC `peak_V`."
        )
    if (sensitivity is not None) and (sensitivity > 0):
        raise ValueError(
            "Hydrophone calibrated sensitivity should be entered as a negative number."
        )

    with open(filename, "rb") as f:
        riff_key = f.read(4)
        file_size = struct.unpack("<I", f.read(4))[0]
        wave_key = f.read(4)
        list_key = f.read(4)

        # Skip metadata if it exits (don't know how to parse)
        if "LIST" in list_key.decode():
            list_size = struct.unpack("<I", f.read(4))[0]
            f.seek(f.tell() + list_size)
        else:
            f.seek(f.tell() - 4)

        # Read this to get the bits/sample
        fmt_key = f.read(4)
        fmt_size = struct.unpack("<I", f.read(4))[0]
        compression_code = struct.unpack("<H", f.read(2))[0]
        n_channels = struct.unpack("<H", f.read(2))[0]
        sample_rate = struct.unpack("<I", f.read(4))[0]
        bytes_per_sec = struct.unpack("<I", f.read(4))[0]
        block_align = struct.unpack("<H", f.read(2))[0]
        bits_per_sample = struct.unpack("<H", f.read(2))[0]
        f.seek(f.tell() + fmt_size - 16)

    # Read data using scipy cause that's easier (will auto drop as int16 or int32)
    fs, raw = wavfile.read(filename)
    length = raw.shape[0] // fs  # length of recording in seconds

    if bits_per_sample == 16:
        max_count = 2 ** (16 - 1)  # 16 bit
    elif bits_per_sample == 24:
        max_count = 2 ** (32 - 1) - 2**8  # 24 bit read in as 32 bit
    elif bits_per_sample <= 32:
        max_count = 2 ** (32 - 1)  # 32 bit

    # Normalize and then scale to peak voltage
    # Use 64 bit float for decimal accuracy
    raw_V = raw.astype(float) / max_count * peak_V

    # Get time
    end_time = np.datetime64(start_time) + np.timedelta64(length, "s")
    time = pd.date_range(start_time, end_time, raw.size + 1)

    if sensitivity is not None:
        # Subtract gain
        # hydrophone with sensitivity of -177 dB and gain of -3 dB = sensitivity of -174 dB
        if gain:
            sensitivity -= gain
        # Convert calibration from dB rel 1 V/uPa into ratio
        Sf = 10 ** (sensitivity / 20)  # V/uPa

        # Sound pressure
        pressure = raw_V / Sf  # uPa
        pressure = pressure / 1e6  # Pa

        # Min resolution
        min_res = peak_V / max_count / Sf  # uPa
        # Pressure at which sensor is saturated
        max_sat = peak_V / Sf  # uPa

        out = xr.DataArray(
            pressure,
            coords={"time": time[:-1]},
            attrs={
                "units": "Pa",
                "sensitivity": Sf,
                "resolution": np.round(min_res / 1e6, 9),
                "valid_min": np.round(
                    -max_sat / 1e6,
                    6,
                ),
                "valid_max": np.round(max_sat / 1e6, 6),
                "fs": fs,
                "filename": filename.replace("\\", "/").split("/")[-1],
            },
        )

    else:
        # Voltage min resolution
        min_res = peak_V / max_count  # V
        # Voltage at which sensor is saturated
        max_sat = peak_V  # V

        out = xr.DataArray(
            raw_V,
            coords={"time": time[:-1]},
            attrs={
                "units": "V",
                "resolution": np.round(min_res, 6),
                "valid_min": -max_sat,
                "valid_max": max_sat,
                "fs": fs,
                "filename": str(filename).replace("\\", "/").split("/")[-1],
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
        filename, peak_V=1, sensitivity=sensitivity, gain=gain, start_time=start_time
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
        riff_key = f.read(4)
        file_size = struct.unpack("<I", f.read(4))[0]
        wave_key = f.read(4)
        list_key = f.read(4)

        # Read metadata if available
        if "LIST" in list_key.decode():
            list_size = struct.unpack("<I", f.read(4))[0]
            info_key = f.read(4)

            # Hydrophone make and SN
            iart_key = f.read(4)
            iart_size = struct.unpack("<I", f.read(4))[0]
            iart = f.read(iart_size).decode().rstrip("\x00")

            # Hydrophone model
            iprd_key = f.read(4)
            iprd_size = struct.unpack("<I", f.read(4))[0]
            iprd = f.read(iprd_size).decode().rstrip("\x00")

            # File creation date
            icrd_key = f.read(4)
            icrd_size = struct.unpack("<I", f.read(4))[0]
            icrd = f.read(icrd_size).decode().rstrip("\x00")

            # Hydrophone make and software version
            isft_key = f.read(4)
            isft_size = struct.unpack("<I", f.read(4))[0]
            isft = f.read(isft_size).decode().rstrip("\x00")

            # Original filename
            inam_key = f.read(4)
            inam_size = struct.unpack("<I", f.read(4))[0]
            inam = f.read(inam_size).decode().rstrip("\x00")

            # Additional comments
            icmt_key = f.read(4)
            icmt_size = struct.unpack("<I", f.read(4))[0]
            icmt = f.read(icmt_size).decode().rstrip("\x00")

            fields = icmt.split(",")
            peak_voltage = fields[0]
            hphone_sensitivity = fields[1].lstrip()
            humidity = fields[2].lstrip()
            temp = fields[3].lstrip()
            if len(fields) > 6:
                accel = ",".join(fields[4:7]).lstrip()
                mag = ",".join(fields[7:10]).lstrip()
            else:
                accel = []
                mag = []
            count_at_peak_V = fields[-2].lstrip()
            n_sequence = fields[-1].lstrip()
        else:
            f.seek(f.tell() - 4)

    peak_V = float(peak_voltage.split(" ")[0])
    Sf = int(hphone_sensitivity.split(" ")[0])

    # Use stored sensitivity
    if use_metadata and (sensitivity is None):
        sensitivity = Sf

    out = read_hydrophone(
        filename,
        peak_V=peak_V,
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
            "count_at_peak_voltage": count_at_peak_V,
            "sequence_num": n_sequence,
        }
    )

    return out


def export_audio(filename, P, gain=1):
    """Creates human-scaled audio file from underwater recording."""
    # Convert from Pascals to UPa
    uPa = P.values.T * 1e6
    # Change to voltage waveform
    V = uPa * 10 ** (P.sensitivity / 20)  # in V
    # Normalize
    V = V / max(abs(V)) * gain
    # Convert to (little-endian) 16 bit integers.
    audio = (V * (2**16 - 1)).astype("<h")

    with wave.open(f"{filename}.wav", "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(P.fs)
        f.writeframes(audio.tobytes())
