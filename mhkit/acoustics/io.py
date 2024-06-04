import struct
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import wavfile
from datetime import datetime


def read_iclisten(filename, Sf=None, gain=0):
    """
    Read .wav file from an Ocean Sonics icListen hydrophone.

    Parameters
    ----------
    filename: string
        Input filename
    Sf: numeric
        Hydrophone calibration sensitivity in dB re 1 V/uPa
    gain: numeric
        Amplifier gain. Default 0.

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] indexed by time[s]
    """

    # icListen hydrophone
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

        data_key = f.read(4)
        data_size = struct.unpack("<I", f.read(4))[0]

    if Sf is None:
        Sf = int(hphone_sensitivity.split(" ")[0])
    else:
        warnings.warn(f"Overriding recorded hydrophone sensitivity value with {Sf}.")
    if Sf > 0:
        raise ValueError("Hydrophone sensitivity should be negative.")
    if gain:
        warnings.warn(
            "Hydrophone gain not currently taken into consideration for icListens."
        )

    # convert calibration from dB rel 1 V/uPa into ratio
    Sf = 10 ** (Sf / 20)  # V/uPa

    # Read data using scipy cause that's easier
    fs, raw = wavfile.read(filename)
    length = raw.shape[0] // fs  # length of recording in seconds

    # Normalize raw data and scale to peak voltage
    peak_V = float(peak_voltage.split(" ")[0])
    if bits_per_sample == 16:
        mx_count = 2 ** (16 - 1)  # 16 bit
    elif bits_per_sample <= 32:
        mx_count = 2 ** (32 - 1)  # 24 (can't read 24 bit numbers) or 32 bit
    raw_V = raw.astype(np.float32) / mx_count * peak_V

    # Min resolution
    min_res = peak_V / mx_count / Sf  # uPa
    # Pressure at which sensor is saturated
    max_sat = peak_V / Sf  # uPa

    # Sound pressure
    pressure = raw_V / Sf  # uPa
    pressure = pressure / 1e6  # Pa

    # Sanity check - check total average sound pressure level
    rms_V = np.sqrt(np.mean(raw_V**2))
    spl_avg = 20 * np.log10(rms_V / Sf)

    # Get time
    start_time = np.datetime64(icrd)
    end_time = start_time + np.timedelta64(length, "s")
    time = pd.date_range(start_time, end_time, raw.size + 1)

    out = xr.DataArray(
        pressure,
        coords={"time": time[:-1]},
        attrs={
            "units": "Pa",
            "resolution": np.round(min_res / 1e6, 9),
            "valid_max": np.round(max_sat / 1e6, 6),
            "fs": fs,
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
        },
    )

    return out
