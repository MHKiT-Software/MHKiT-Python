import struct
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import wavfile


def read_hydrophone(
    filename, peak_V=None, Sf=None, gain=0, start_time="2024-06-06T00:00:00"
):
    """
    Read .wav file from a hydrophone.

    Parameters
    ----------
    filename: string
        Input filename
    peak_V: numeric
        Peak voltage supplied to the analog to digital converter (ADC) in V.
        (Or 1/2 of the peak to peak voltage).
    Sf: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative.
    gain: numeric
        Amplifier gain in dB re 1 V/uPa. Default 0.

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] indexed by time[s]
    """
    if peak_V is None:
        raise ValueError(
            "Please provide the peak voltage of the hydrophone's ADC `peak_V`."
        )
    if Sf is None:
        raise ValueError("Please provide the hydrophone's calibrated sensitivity `Sf`.")
    elif Sf > 0:
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

    # Need to cross-check with wavfile.read datatype?
    if bits_per_sample == 16:
        max_count = 2 ** (16 - 1)  # 16 bit
    elif bits_per_sample <= 32:
        max_count = 2 ** (32 - 1)  # 24 (computer doesn't read 24 bit numbers) or 32 bit

    # Subtract gain
    # hydrophone with sensitivity of -177 dB and gain of -3 dB = sensitivity of 174 dB
    if gain:
        Sf -= gain
    # Convert calibration from dB rel 1 V/uPa into ratio
    Sf = 10 ** (Sf / 20)  # V/uPa

    # Use 64 bit float for decimal accuracy
    raw_V = raw.astype(float) / max_count * peak_V

    # Sound pressure
    pressure = raw_V / Sf  # uPa
    pressure = pressure / 1e6  # Pa

    # Min resolution
    min_res = peak_V / max_count / Sf  # uPa
    # Pressure at which sensor is saturated
    max_sat = peak_V / Sf  # uPa

    # Get time
    end_time = np.datetime64(start_time) + np.timedelta64(length, "s")
    time = pd.date_range(start_time, end_time, raw.size + 1)

    out = xr.DataArray(
        pressure,
        coords={"time": time[:-1]},
        attrs={
            "units": "Pa",
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

    return out


def read_soundtrap(filename, Sf=None, gain=0):
    """
    Read .wav file from an Ocean Instruments SoundTrap hydrophone.

    Parameters
    ----------
    filename: string
        Input filename
    Sf: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative.
    gain: numeric
        Amplifier gain in dB re 1 V/uPa. Default 0.

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] indexed by time[s]
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
        filename, peak_V=1, Sf=Sf, gain=gain, start_time=start_time
    )
    out.attrs["make"] = "SoundTrap"

    return out


def read_iclisten(filename):
    """
    Read .wav file from an Ocean Sonics icListen "Smart" hydrophone.

    Parameters
    ----------
    filename: string
        Input filename
    Sf: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa.
        Should be negative.
    gain: numeric
        Amplifier gain in dB re 1 V/uPa. Default 0.

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] indexed by time[s]
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
            accel = ",".join(fields[4:7]).lstrip()
            mag = ",".join(fields[7:10]).lstrip()
            count_at_peak_V = fields[-2].lstrip()
            n_sequence = fields[-1].lstrip()
        else:
            f.seek(f.tell() - 4)

    Sf = int(hphone_sensitivity.split(" ")[0])
    peak_V = float(peak_voltage.split(" ")[0])
    out = read_hydrophone(
        filename, peak_V=peak_V, Sf=Sf, gain=0, start_time=np.datetime64(icrd)
    )

    out.attrs.update(
        {
            "serial_num": iart,
            "model": iprd,
            "software_ver": isft,
            "filename": inam + ".wav",
            "peak_voltage": peak_voltage,
            "sensitivity": hphone_sensitivity,
            "humidity": humidity,
            "temperature": temp,
            "accelerometer": accel,
            "magnetometer": mag,
            "count_at_peak_voltage": count_at_peak_V,
            "sequence_num": n_sequence,
        }
    )

    return out
