import numpy as np

century = 2000
adcp_type = {
    4: "Broadband",
    5: "Broadband",
    6: "Navigator",
    10: "Rio Grande",
    11: "H-ADCP",
    14: "Ocean Surveyor",
    16: "Workhorse",
    19: "Navigator",
    23: "Ocean Surveyor",
    28: "ChannelMaster",
    31: "StreamPro",
    34: "Explorer",
    37: "Navigator",
    41: "DVS",
    43: "Workhorse",
    44: "RiverRay",
    47: "SentinelV",
    50: "Workhorse",
    51: "Workhorse",
    52: "Workhorse",
    53: "Navigator",
    55: "DVS",
    56: "RiverPro",
    59: "Meridian",
    61: "Pinnacle",
    66: "SentinelV",
    67: "Pathfinder",
    73: "Pioneer",
    74: "Tasman",
    76: "WayFinder",
    77: "Workhorse",
    78: "Workhorse",
}

data_defs = {
    "number": (
        [],
        "data_vars",
        "uint32",
        "1",
        "Ensemble Number",
        "number_of_observations",
    ),
    "rtc": ([7], "sys", "uint16", "1", "Real Time Clock", ""),
    "builtin_test_fail": ([], "data_vars", "bool", "1", "Built-In Test Failures", ""),
    "c_sound": (
        [],
        "data_vars",
        "float32",
        "m s-1",
        "Speed of Sound",
        "speed_of_sound_in_sea_water",
    ),
    "depth": ([], "data_vars", "float32", "m", "Depth", "depth"),
    "pitch": ([], "data_vars", "float32", "degree", "Pitch", "platform_pitch"),
    "roll": ([], "data_vars", "float32", "degree", "Roll", "platform_roll"),
    "heading": (
        [],
        "data_vars",
        "float32",
        "degree",
        "Heading",
        "platform_orientation",
    ),
    "temp": (
        [],
        "data_vars",
        "float32",
        "degree_C",
        "Temperature",
        "sea_water_temperature",
    ),
    "salinity": ([], "data_vars", "float32", "psu", "Salinity", "sea_water_salinity"),
    "min_preping_wait": (
        [],
        "data_vars",
        "float32",
        "s",
        "Minimum Pre-Ping Wait Time Between Measurements",
        "",
    ),
    "heading_std": (
        [],
        "data_vars",
        "float32",
        "degree",
        "Heading Standard Deviation",
        "",
    ),
    "pitch_std": ([], "data_vars", "float32", "degree", "Pitch Standard Deviation", ""),
    "roll_std": ([], "data_vars", "float32", "degree", "Roll Standard Deviation", ""),
    "adc": ([8], "sys", "uint8", "1", "Analog-Digital Converter Output", ""),
    "error_status": ([], "attrs", "float32", "1", "Error Status", ""),
    "pressure": ([], "data_vars", "float32", "dbar", "Pressure", "sea_water_pressure"),
    "pressure_std": (
        [],
        "data_vars",
        "float32",
        "dbar",
        "Pressure Standard Deviation",
        "",
    ),
    "vel": (["nc", 4], "data_vars", "float32", "m s-1", "Water Velocity", ""),
    "amp": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "1",
        "Acoustic Signal Amplitude",
        "signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
    ),
    "corr": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "1",
        "Acoustic Signal Correlation",
        "beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water",
    ),
    "prcnt_gd": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "%",
        "Percent Good",
        "proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water",
    ),
    "status": (["nc", 4], "data_vars", "float32", "1", "Status", ""),
    "dist_bt": ([4], "data_vars", "float32", "m", "Bottom Track Measured Depth", ""),
    "vel_bt": (
        [4],
        "data_vars",
        "float32",
        "m s-1",
        "Platform Velocity from Bottom Track",
        "",
    ),
    "corr_bt": (
        [4],
        "data_vars",
        "uint8",
        "1",
        "Bottom Track Acoustic Signal Correlation",
        "",
    ),
    "amp_bt": (
        [4],
        "data_vars",
        "uint8",
        "1",
        "Bottom Track Acoustic Signal Amplitude",
        "",
    ),
    "prcnt_gd_bt": ([4], "data_vars", "uint8", "%", "Bottom Track Percent Good", ""),
    "time": (
        [],
        "coords",
        "float64",
        "seconds since 1970-01-01 00:00:00",
        "Time",
        "time",
    ),
    "alt_dist": ([], "data_vars", "float32", "m", "Altimeter Range", "altimeter_range"),
    "alt_rssi": (
        [],
        "data_vars",
        "uint8",
        "dB",
        "Altimeter Recieved Signal Strength Indicator",
        "",
    ),
    "alt_eval": ([], "data_vars", "uint8", "dB", "Altimeter Evaluation Amplitude", ""),
    "alt_status": ([], "data_vars", "uint8", "bit", "Altimeter Status", ""),
    "time_gps": (
        [],
        "coords",
        "float64",
        "seconds since 1970-01-01 00:00:00",
        "GPS Time",
        "time",
    ),
    "clock_offset_UTC_gps": (
        [],
        "data_vars",
        "float64",
        "s",
        "Instrument Clock Offset from UTC",
        "",
    ),
    "latitude_gps": (
        [],
        "data_vars",
        "float32",
        "degrees_north",
        "Latitude",
        "latitude",
    ),
    "longitude_gps": (
        [],
        "data_vars",
        "float32",
        "degrees_east",
        "Longitude",
        "longitude",
    ),
    "avg_speed_gps": (
        [],
        "data_vars",
        "float32",
        "m s-1",
        "Average Platform Speed",
        "platform_speed_wrt_ground",
    ),
    "avg_dir_gps": (
        [],
        "data_vars",
        "float32",
        "degree",
        "Average Platform Direction",
        "platform_course",
    ),
    "speed_made_good_gps": (
        [],
        "data_vars",
        "float32",
        "m s-1",
        "Platform Speed Made Good",
        "platform_speed_wrt_ground",
    ),
    "dir_made_good_gps": (
        [],
        "data_vars",
        "float32",
        "degree",
        "Platform Direction Made Good",
        "platform_course",
    ),
    "flags_gps": ([], "data_vars", "float32", "bits", "GPS Flags", ""),
    "fix_gps": ([], "data_vars", "int8", "1", "GPS Fix", ""),
    "n_sat_gps": ([], "data_vars", "int8", "count", "Number of Satellites", ""),
    "hdop_gps": (
        [],
        "data_vars",
        "float32",
        "1",
        "Horizontal Dilution of Precision",
        "",
    ),
    "elevation_gps": ([], "data_vars", "float32", "m", "Elevation above MLLW", ""),
    "rtk_age_gps": (
        [],
        "data_vars",
        "float32",
        "s",
        "Age of Received Real Time Kinetic Signal",
        "",
    ),
    "speed_over_grnd_gps": (
        [],
        "data_vars",
        "float32",
        "m s-1",
        "Platform Speed over Ground",
        "platform_speed_wrt_ground",
    ),
    "dir_over_grnd_gps": (
        [],
        "data_vars",
        "float32",
        "degree",
        "Platform Direction over Ground",
        "platform_course",
    ),
    "heading_gps": (
        [],
        "data_vars",
        "float32",
        "degree",
        "GPS Heading",
        "platform_orientation",
    ),
    "pitch_gps": ([], "data_vars", "float32", "degree", "GPS Pitch", "platform_pitch"),
    "roll_gps": ([], "data_vars", "float32", "degree", "GPS Roll", "platform_roll"),
    "dist_nmea": ([], "data_vars", "float32", "m", "Depth Sounder Range", ""),
    "vel_sl": (
        ["nc", 4],
        "data_vars",
        "float32",
        "m s-1",
        "Surface Layer Water Velocity",
        "",
    ),
    "corr_sl": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "1",
        "Surface Layer Acoustic Signal Correlation",
        "beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water",
    ),
    "amp_sl": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "1",
        "Surface Layer Acoustic Signal Amplitude",
        "signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
    ),
    "prcnt_gd_sl": (
        ["nc", 4],
        "data_vars",
        "uint8",
        "%",
        "Surface Layer Percent Good",
        "proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water",
    ),
    "status_sl": (["nc", 4], "data_vars", "float32", "1", "Surface Layer Status", ""),
}


def _get(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        return dat[nm]
    else:
        return dat[grp][nm]


def _in_group(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        return nm in dat
    else:
        return nm in dat[grp]


def _pop(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        dat.pop(nm)
    else:
        dat[grp].pop(nm)


def _setd(dat, nm, val):
    grp = data_defs[nm][1]
    if grp is None:
        dat[nm] = val
    else:
        dat[grp][nm] = val


def _idata(dat, nm, sz):
    group = data_defs[nm][1]
    dtype = data_defs[nm][2]
    units = data_defs[nm][3]
    long_name = data_defs[nm][4]
    standard_name = data_defs[nm][5]
    arr = np.empty(sz, dtype=dtype)
    if dtype.startswith("float"):
        arr[:] = np.NaN
    dat[group][nm] = arr
    dat["units"][nm] = units
    dat["long_name"][nm] = long_name
    if standard_name:
        dat["standard_name"][nm] = standard_name
    return dat


def _get_size(name, n=None, ncell=0):
    sz = list(data_defs[name][0])  # create a copy!
    if "nc" in sz:
        sz.insert(sz.index("nc"), ncell)
        sz.remove("nc")
    if n is None:
        return tuple(sz)
    return tuple(sz + [n])


class _variable_setlist(set):
    def __iadd__(self, vals):
        if vals[0] not in self:
            self |= set(vals)
        return self


class _ensemble:
    n_avg = 1
    k = -1  # This is the counter for filling the ensemble object

    def __getitem__(self, nm):
        return getattr(self, nm)

    def __init__(self, navg, n_cells):
        if navg is None or navg == 0:
            navg = 1
        self.n_avg = navg
        self.n_cells = n_cells
        for nm in data_defs:
            setattr(
                self,
                nm,
                np.zeros(_get_size(nm, n=navg, ncell=n_cells), dtype=data_defs[nm][2]),
            )

    def clean_data(self):
        self["vel"][self["vel"] == -32.768] = np.NaN
