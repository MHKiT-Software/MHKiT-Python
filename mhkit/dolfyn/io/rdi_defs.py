import numpy as np
import logging

from . import rdi_lib as lib
from .. import time as tmlib


century = np.uint16(2000)
adcp_type = {
    4: "Broadband",
    5: "Broadband",
    6: "Navigator",
    8: "Workhorse",
    9: "Navigator",
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
        "degree_north",
        "Latitude",
        "latitude",
    ),
    "longitude_gps": (
        [],
        "data_vars",
        "float32",
        "degree_east",
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


def skip_Ncol(rdr, n_skip=1):
    """Skip specified number of columns. For profile measurements."""
    rdr.f.seek(n_skip * rdr.cfg["n_cells"], 1)
    rdr._nbyte = 2 + n_skip * rdr.cfg["n_cells"]


def skip_Nbyte(rdr, n_skip):
    """Skip specified number of bytes. For non-profile measurements."""
    rdr.f.seek(n_skip, 1)
    rdr._nbyte = 2 + n_skip


def switch_profile(rdr, bb):
    """Switch between bb, nb and sl profiles"""
    if bb == 1:
        ens = rdr.ensembleBB
        cfg = rdr.cfgbb
        # Placeholder for dual profile mode
        # Solution for vmdas profile in bb spot (vs nb)
        tag = ""
    elif bb == 2:
        ens = rdr.ensemble
        cfg = rdr.cfg
        tag = "_sl"
    else:
        ens = rdr.ensemble
        cfg = rdr.cfg
        tag = ""

    return ens, cfg, tag


def read_cfgseg(rdr, bb=False):
    """Read ensemble configuration header"""
    cfgstart = rdr.f.tell()

    if bb:
        cfg = rdr.cfgbb
    else:
        cfg = rdr.cfg
    fd = rdr.f
    tmp = fd.read_ui8(5)
    prog_ver0 = tmp[0]
    cfg["prog_ver"] = float(tmp[0] + tmp[1] * 0.01)
    cfg["inst_model"] = adcp_type.get(tmp[0], "unrecognized instrument")
    config = tmp[2:4]
    cfg["beam_angle"] = [15, 20, 30, [0, 25][int(tmp[0] in [11, 47, 66])]][
        (config[1] & 3)
    ]
    beam5 = [0, 1][int((config[1] & 16) == 16)]
    # Carrier frequency
    if tmp[0] in [47, 66]:  # new freqs for Sentinel Vs
        cfg["freq"] = [38.4, 76.8, 153.6, 307.2, 491.52, 983.04, 2457.6][
            (config[0] & 7)
        ]
    elif tmp[0] == 31:
        cfg["freq"] = 2000
    elif tmp[0] == 61:
        cfg["freq"] = 44
    else:
        cfg["freq"] = [75, 150, 300, 600, 1200, 2400, 38][(config[0] & 7)]
    cfg["beam_pattern"] = ["concave", "convex"][int((config[0] & 8) == 8)]
    cfg["orientation"] = ["down", "up"][int((config[0] & 128) == 128)]
    simflag = ["real", "simulated"][tmp[4]]
    fd.seek(1, 1)
    cfg["n_beams"] = fd.read_ui8(1) + beam5
    # Check if number of cells has changed
    n_cells = fd.read_ui8(1)
    if ("n_cells" not in cfg) or (n_cells != cfg["n_cells"]):
        cfg["n_cells"] = n_cells
        if rdr._debug_level > 0:
            logging.info(f"Number of cells set to {cfg['n_cells']}")
    cfg["pings_per_ensemble"] = fd.read_ui16(1)
    # Check if cell size has changed
    cs = float(fd.read_ui16(1) * 0.01)
    if ("cell_size" not in cfg) or (cs != cfg["cell_size"]):
        rdr.cs_diff = cs if "cell_size" not in cfg else (cs - cfg["cell_size"])
        cfg["cell_size"] = cs
        if rdr._debug_level > 0:
            logging.info(f"Cell size set to {cfg['cell_size']}")
    cfg["blank_dist"] = round(float(fd.read_ui16(1) * 0.01), 2)
    cfg["profiling_mode"] = fd.read_ui8(1)
    cfg["min_corr_threshold"] = fd.read_ui8(1)
    cfg["n_code_reps"] = fd.read_ui8(1)
    cfg["min_prcnt_gd"] = fd.read_ui8(1)
    cfg["max_error_vel"] = float(fd.read_ui16(1) * 0.001)
    cfg["sec_between_ping_groups"] = round(
        float(np.sum(np.array(fd.read_ui8(3)) * [60.0, 1.0, 0.01])), 3
    )
    coord_sys = fd.read_ui8(1)
    cfg["coord_sys"] = ["beam", "inst", "ship", "earth"][((coord_sys >> 3) & 3)]
    cfg["use_pitchroll"] = ["no", "yes"][(coord_sys & 4) == 4]
    cfg["use_3beam"] = ["no", "yes"][(coord_sys & 2) == 2]
    cfg["bin_mapping"] = ["no", "yes"][(coord_sys & 1) == 1]
    cfg["heading_misalign_deg"] = float(fd.read_i16(1) * 0.01)
    cfg["magnetic_var_deg"] = float(fd.read_i16(1) * 0.01)
    cfg["sensors_src"] = np.binary_repr(fd.read_ui8(1), 8)
    cfg["sensors_avail"] = np.binary_repr(fd.read_ui8(1), 8)
    cfg["bin1_dist_m"] = round(float(fd.read_ui16(1) * 0.01), 4)
    cfg["transmit_pulse_m"] = round(float(fd.read_ui16(1) * 0.01), 2)
    cfg["water_ref_cells"] = list(fd.read_ui8(2).astype(list))  # list for attrs
    cfg["false_target_threshold"] = fd.read_ui8(1)
    fd.seek(1, 1)
    cfg["transmit_lag_m"] = float(fd.read_ui16(1) * 0.01)
    rdr._nbyte = 40

    if cfg["prog_ver"] >= 8.14:
        cpu_serialnum = fd.read_ui8(8)
        rdr._nbyte += 8
    if cfg["prog_ver"] >= 8.24:
        cfg["bandwidth"] = fd.read_ui16(1)
        rdr._nbyte += 2
    if cfg["prog_ver"] >= 9.68:
        cfg["power_level"] = fd.read_ui8(1)
        # cfg['navigator_basefreqindex'] = fd.read_ui8(1)
        fd.seek(1, 1)
        cfg["serialnum"] = fd.read_ui32(1)
        ba = fd.read_ui8(1)
        if not cfg["beam_angle"]:
            cfg["beam_angle"] = ba
        rdr._nbyte += 7

    rdr.configsize = rdr.f.tell() - cfgstart
    if rdr._debug_level > -1:
        logging.info("Read Config")


def read_fixed(rdr, bb=False):
    """Read fixed header"""
    read_cfgseg(rdr, bb=bb)
    rdr._nbyte += 2
    if rdr._debug_level > -1:
        logging.info("Read Fixed")

    # Check if n_cells has increased (for winriver transect files)
    if hasattr(rdr, "ensemble"):
        rdr.n_cells_diff = rdr.cfg["n_cells"] - rdr.ensemble["n_cells"]
        # Increase n_cells if greater than 0
        if rdr.n_cells_diff > 0:
            rdr.ensemble = lib._ensemble(rdr.n_avg, rdr.cfg["n_cells"])
            if rdr._debug_level > 0:
                logging.warning(
                    f"Maximum number of cells increased to {rdr.cfg['n_cells']}"
                )


def read_fixed_sl(rdr):
    """Read surface layer fixed header"""
    cfg = rdr.cfg
    cfg["surface_layer"] = 1
    n_cells = rdr.f.read_ui8(1)
    # Check if n_cells is greater than what was used in prior profiles
    if n_cells > rdr.n_cells_sl:
        rdr.n_cells_sl = n_cells
        if rdr._debug_level > 0:
            logging.warning(
                f"Maximum number of surface layer cells increased to {n_cells}"
            )
    cfg["n_cells_sl"] = n_cells
    # Assuming surface layer profile cell size never changes
    cfg["cell_size_sl"] = float(rdr.f.read_ui16(1) * 0.01)
    cfg["bin1_dist_m_sl"] = round(float(rdr.f.read_ui16(1) * 0.01), 4)

    if rdr._debug_level > -1:
        logging.info("Read Surface Layer Config")
    rdr._nbyte = 2 + 5


def read_var(rdr, bb=False):
    """Read variable header"""
    fd = rdr.f
    if bb:
        ens = rdr.ensembleBB
    else:
        ens = rdr.ensemble
    ens.k += 1
    ens = rdr.ensemble
    k = ens.k
    rdr.vars_read += [
        "number",
        "rtc",
        "number",
        "builtin_test_fail",
        "c_sound",
        "depth",
        "heading",
        "pitch",
        "roll",
        "salinity",
        "temp",
        "min_preping_wait",
        "heading_std",
        "pitch_std",
        "roll_std",
        "adc",
    ]
    ens.number[k] = fd.read_ui16(1)
    ens.rtc[:, k] = fd.read_ui8(7)
    ens.number[k] += 65535 * fd.read_ui8(1)
    ens.builtin_test_fail[k] = fd.read_ui16(1)
    ens.c_sound[k] = fd.read_ui16(1)
    ens.depth[k] = fd.read_ui16(1) * 0.1
    ens.heading[k] = fd.read_ui16(1) * 0.01
    ens.pitch[k] = fd.read_i16(1) * 0.01
    ens.roll[k] = fd.read_i16(1) * 0.01
    ens.salinity[k] = fd.read_i16(1)
    ens.temp[k] = fd.read_i16(1) * 0.01
    ens.min_preping_wait[k] = (fd.read_ui8(3) * np.array([60, 1, 0.01])).sum()
    ens.heading_std[k] = fd.read_ui8(1)
    ens.pitch_std[k] = fd.read_ui8(1) * 0.1
    ens.roll_std[k] = fd.read_ui8(1) * 0.1
    ens.adc[:, k] = fd.read_i8(8)
    rdr._nbyte = 2 + 40

    cfg = rdr.cfg
    if cfg["inst_model"].lower() == "broadband":
        if cfg["prog_ver"] >= 5.55:
            fd.seek(15, 1)
            cent = fd.read_ui8(1)
            ens.rtc[:, k] = fd.read_ui8(7)
            ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
            rdr._nbyte += 23
    elif cfg["inst_model"].lower() == "ocean surveyor":
        fd.seek(16, 1)  # 30 bytes all set to zero, 14 read above
        rdr._nbyte += 16
        if cfg["prog_ver"] > 23:
            fd.seek(2, 1)
            rdr._nbyte += 2
    else:
        ens.error_status[k] = np.binary_repr(fd.read_ui32(1), 32)
        rdr.vars_read += ["pressure", "pressure_std"]
        rdr._nbyte += 4
        if cfg["prog_ver"] >= 8.13:
            # Added pressure sensor stuff in 8.13
            fd.seek(2, 1)
            ens.pressure[k] = fd.read_ui32(1) * 0.001  # dPa to dbar
            ens.pressure_std[k] = fd.read_ui32(1) * 0.001
            rdr._nbyte += 10
        if cfg["prog_ver"] >= 8.24:
            # Spare byte added 8.24
            fd.seek(1, 1)
            rdr._nbyte += 1
        if cfg["prog_ver"] >= 16.05:
            # Added more fields with century in clock
            cent = fd.read_ui8(1)
            ens.rtc[:, k] = fd.read_ui8(7)
            ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
            rdr._nbyte += 8
        if cfg["prog_ver"] >= 56:
            fd.seek(1)  # lag near bottom flag
            rdr._nbyte += 1

    if rdr._debug_level > -1:
        logging.info("Read Var")


def read_vel(rdr, bb=0):
    """Read water velocity block"""
    ens, cfg, tg = switch_profile(rdr, bb)
    rdr.vars_read += ["vel" + tg]
    n_cells = cfg["n_cells" + tg]

    k = ens.k
    vel = np.array(rdr.f.read_i16(4 * n_cells)).reshape((n_cells, 4)) * 0.001
    ens["vel" + tg][:n_cells, :, k] = vel
    rdr._nbyte = 2 + 4 * n_cells * 2
    if rdr._debug_level > -1:
        logging.info("Read Vel")


def read_corr(rdr, bb=0):
    """Read acoustic signal correlation block"""
    ens, cfg, tg = switch_profile(rdr, bb)
    rdr.vars_read += ["corr" + tg]
    n_cells = cfg["n_cells" + tg]

    k = ens.k
    ens["corr" + tg][:n_cells, :, k] = np.array(rdr.f.read_ui8(4 * n_cells)).reshape(
        (n_cells, 4)
    )
    rdr._nbyte = 2 + 4 * n_cells
    if rdr._debug_level > -1:
        logging.info("Read Corr")


def read_amp(rdr, bb=0):
    """Read acoustic signal amplitude block"""
    ens, cfg, tg = switch_profile(rdr, bb)
    rdr.vars_read += ["amp" + tg]
    n_cells = cfg["n_cells" + tg]

    k = ens.k
    ens["amp" + tg][:n_cells, :, k] = np.array(rdr.f.read_ui8(4 * n_cells)).reshape(
        (n_cells, 4)
    )
    rdr._nbyte = 2 + 4 * n_cells
    if rdr._debug_level > -1:
        logging.info("Read Amp")


def read_prcnt_gd(rdr, bb=0):
    """Read acoustic signal 'percent good' block"""
    ens, cfg, tg = switch_profile(rdr, bb)
    rdr.vars_read += ["prcnt_gd" + tg]
    n_cells = cfg["n_cells" + tg]

    ens["prcnt_gd" + tg][:n_cells, :, ens.k] = np.array(
        rdr.f.read_ui8(4 * n_cells)
    ).reshape((n_cells, 4))
    rdr._nbyte = 2 + 4 * n_cells
    if rdr._debug_level > -1:
        logging.info("Read PG")


def read_status(rdr, bb=0):
    """Read ADCP status block"""
    ens, cfg, tg = switch_profile(rdr, bb)
    rdr.vars_read += ["status" + tg]
    n_cells = cfg["n_cells" + tg]

    ens["status" + tg][:n_cells, :, ens.k] = np.array(
        rdr.f.read_ui8(4 * n_cells)
    ).reshape((n_cells, 4))
    rdr._nbyte = 2 + 4 * n_cells
    if rdr._debug_level > -1:
        logging.info("Read Status")


def read_bottom(rdr):
    """Read bottom track block"""
    rdr.vars_read += ["dist_bt", "vel_bt", "corr_bt", "amp_bt", "prcnt_gd_bt"]
    fd = rdr.f
    ens = rdr.ensemble
    k = ens.k
    cfg = rdr.cfg
    if rdr._vm_source == 2:
        rdr.vars_read += ["latitude_gps", "longitude_gps"]
        fd.seek(2, 1)
        long1 = fd.read_ui16(1)
        fd.seek(6, 1)
        ens.latitude_gps[k] = fd.read_i32(1) * rdr._cfac32
        if ens.latitude_gps[k] == 0:
            ens.latitude_gps[k] = np.nan
    else:
        fd.seek(14, 1)
    ens.dist_bt[:, k] = fd.read_ui16(4) * 0.01
    ens.vel_bt[:, k] = fd.read_i16(4) * 0.001
    ens.corr_bt[:, k] = fd.read_ui8(4)
    ens.amp_bt[:, k] = fd.read_ui8(4)
    ens.prcnt_gd_bt[:, k] = fd.read_ui8(4)
    if rdr._vm_source == 2:
        fd.seek(2, 1)
        ens.longitude_gps[k] = (long1 + 65536 * fd.read_ui16(1)) * rdr._cfac32
        if ens.longitude_gps[k] > 180:
            ens.longitude_gps[k] = ens.longitude_gps[k] - 360
        if ens.longitude_gps[k] == 0:
            ens.longitude_gps[k] = np.nan
        fd.seek(16, 1)
        qual = fd.read_ui8(1)
        if qual == 0:
            if rdr._debug_level > 0:
                logging.info(
                    "  qual==%d,%f %f"
                    % (qual, ens.latitude_gps[k], ens.longitude_gps[k])
                )
            ens.latitude_gps[k] = np.nan
            ens.longitude_gps[k] = np.nan
        fd.seek(71 - 45 - 16 - 17, 1)
        rdr._nbyte = 2 + 68
    else:
        # Skip reference layer data
        fd.seek(26, 1)
        rdr._nbyte = 2 + 68
    if cfg["prog_ver"] >= 5.3:
        fd.seek(7, 1)  # skip to rangeMsb bytes
        ens.dist_bt[:, k] = ens.dist_bt[:, k] + fd.read_ui8(4) * 655.36
        rdr._nbyte += 11
    if cfg["prog_ver"] >= 16.2 and (cfg.get("sourceprog", "").lower() != "winriver"):
        fd.seek(4, 1)  # not documented
        rdr._nbyte += 4
    if cfg["prog_ver"] >= 56.1:
        fd.seek(4, 1)  # not documented
        rdr._nbyte += 4

    if rdr._debug_level > -1:
        logging.info("Read Bottom Track")


def read_alt(rdr):
    """Read altimeter (range of vertical beam) block"""
    fd = rdr.f
    ens = rdr.ensemble
    k = ens.k
    rdr.vars_read += ["alt_dist", "alt_rssi", "alt_eval", "alt_status"]
    ens.alt_eval[k] = fd.read_ui8(1)  # evaluation amplitude
    ens.alt_rssi[k] = fd.read_ui8(1)  # RSSI amplitude
    ens.alt_dist[k] = fd.read_ui32(1) * 0.001  # range to surface/seafloor
    ens.alt_status[k] = fd.read_ui8(1)  # status bit flags
    rdr._nbyte = 7 + 2
    if rdr._debug_level > -1:
        logging.info("Read Altimeter")


def read_winriver(rdr):
    """Skip WinRiver1 Navigation block (outdated)"""
    rdr._winrivprob = True
    rdr.cfg["sourceprog"] = "WINRIVER"
    if rdr._vm_source not in [2, 3]:
        if rdr._debug_level > -1:
            logging.warning(
                "\n***** Apparently a WinRiver1 file - "
                "NMEA data handler for WinRiver1 not implemented\n"
            )
        rdr._vm_source = 2
    startpos = rdr.f.tell()
    sz = rdr.f.read_ui16(1)
    tmp = rdr.f.reads(sz - 2)
    rdr._nbyte = rdr.f.tell() - startpos + 2


def read_winriver2(rdr):
    """Read WinRiver2 Navigation block"""
    startpos = rdr.f.tell()
    rdr._winrivprob = True
    rdr.cfg["sourceprog"] = "WinRiver2"
    ens = rdr.ensemble
    k = ens.k
    if rdr._debug_level > -1:
        logging.info("Read WinRiver2")
    rdr._vm_source = 3

    spid = rdr.f.read_ui16(1)  # NMEA specific IDs
    if spid in [4, 104]:  # GGA
        sz = rdr.f.read_ui16(1)
        dtime = rdr.f.read_f64(1)
        if sz <= 43:  # If no sentence, data is still stored in nmea format
            empty_gps = rdr.f.reads(sz - 2)
            rdr.f.seek(2, 1)
        else:  # TRDI rewrites the nmea string into their format if one is found
            start_string = rdr.f.reads(6)
            if not isinstance(start_string, str):
                if rdr._debug_level > 0:
                    logging.warning(
                        f"Invalid GGA string found in ensemble {k}," " skipping..."
                    )
                return "FAIL"
            rdr.f.seek(1, 1)
            gga_time = rdr.f.reads(9)
            time = tmlib.timedelta(
                hours=int(gga_time[0:2]),
                minutes=int(gga_time[2:4]),
                seconds=int(gga_time[4:6]),
                milliseconds=int(float(gga_time[6:]) * float(1000)),
            )
            clock = rdr.ensemble.rtc[:, :]
            if clock[0, 0] < 100:
                clock[0, :] += century
            date = tmlib.datetime(*clock[:3, 0]) + time
            ens.time_gps[k] = tmlib.date2epoch(date)[0]
            rdr.f.seek(1, 1)
            ens.latitude_gps[k] = rdr.f.read_f64(1)
            tcNS = rdr.f.reads(1)  # 'N' or 'S'
            if tcNS == "S":
                ens.latitude_gps[k] *= -1
            ens.longitude_gps[k] = rdr.f.read_f64(1)
            tcEW = rdr.f.reads(1)  # 'E' or 'W'
            if tcEW == "W":
                ens.longitude_gps[k] *= -1
            ens.fix_gps[k] = rdr.f.read_ui8(1)  # gps fix type/quality
            ens.n_sat_gps[k] = rdr.f.read_ui8(1)  # of satellites
            # horizontal dilution of precision
            ens.hdop_gps[k] = rdr.f.read_f32(1)
            ens.elevation_gps[k] = rdr.f.read_f32(1)  # altitude
            m = rdr.f.reads(1)  # altitude unit, 'm'
            h_geoid = rdr.f.read_f32(1)  # height of geoid
            m2 = rdr.f.reads(1)  # geoid unit, 'm'
            ens.rtk_age_gps[k] = rdr.f.read_f32(1)
            station_id = rdr.f.read_ui16(1)
        rdr.vars_read += [
            "time_gps",
            "longitude_gps",
            "latitude_gps",
            "fix_gps",
            "n_sat_gps",
            "hdop_gps",
            "elevation_gps",
            "rtk_age_gps",
        ]
        rdr._nbyte = rdr.f.tell() - startpos + 2

    elif spid in [5, 105]:  # VTG
        sz = rdr.f.read_ui16(1)
        dtime = rdr.f.read_f64(1)
        if sz <= 22:  # if no data
            empty_gps = rdr.f.reads(sz - 2)
            rdr.f.seek(2, 1)
        else:
            start_string = rdr.f.reads(6)
            if not isinstance(start_string, str):
                if rdr._debug_level > 0:
                    logging.warning(
                        f"Invalid VTG string found in ensemble {k}," " skipping..."
                    )
                return "FAIL"
            rdr.f.seek(1, 1)
            true_track = rdr.f.read_f32(1)
            t = rdr.f.reads(1)  # 'T'
            magn_track = rdr.f.read_f32(1)
            m = rdr.f.reads(1)  # 'M'
            speed_knot = rdr.f.read_f32(1)
            kts = rdr.f.reads(1)  # 'N'
            speed_kph = rdr.f.read_f32(1)
            kph = rdr.f.reads(1)  # 'K'
            mode = rdr.f.reads(1)
            # knots -> m/s
            ens.speed_over_grnd_gps[k] = speed_knot / 1.944
            ens.dir_over_grnd_gps[k] = true_track
        rdr.vars_read += ["speed_over_grnd_gps", "dir_over_grnd_gps"]
        rdr._nbyte = rdr.f.tell() - startpos + 2

    elif spid in [6, 106]:  # 'DBT' depth sounder
        sz = rdr.f.read_ui16(1)
        dtime = rdr.f.read_f64(1)
        if sz <= 20:
            empty_gps = rdr.f.reads(sz - 2)
            rdr.f.seek(2, 1)
        else:
            start_string = rdr.f.reads(6)
            if not isinstance(start_string, str):
                if rdr._debug_level > 0:
                    logging.warning(
                        f"Invalid DBT string found in ensemble {k}," " skipping..."
                    )
                return "FAIL"
            rdr.f.seek(1, 1)
            depth_ft = rdr.f.read_f32(1)
            ft = rdr.f.reads(1)  # 'f'
            depth_m = rdr.f.read_f32(1)
            m = rdr.f.reads(1)  # 'm'
            depth_fathom = rdr.f.read_f32(1)
            f = rdr.f.reads(1)  # 'F'
            ens.dist_nmea[k] = depth_m
        rdr.vars_read += ["dist_nmea"]
        rdr._nbyte = rdr.f.tell() - startpos + 2

    elif spid in [7, 107]:  # 'HDT'
        sz = rdr.f.read_ui16(1)
        dtime = rdr.f.read_f64(1)
        if sz <= 14:
            empty_gps = rdr.f.reads(sz - 2)
            rdr.f.seek(2, 1)
        else:
            start_string = rdr.f.reads(6)
            if not isinstance(start_string, str):
                if rdr._debug_level > 0:
                    logging.warning(
                        f"Invalid HDT string found in ensemble {k}," " skipping..."
                    )
                return "FAIL"
            rdr.f.seek(1, 1)
            ens.heading_gps[k] = rdr.f.read_f64(1)
            tt = rdr.f.reads(1)
        rdr.vars_read += ["heading_gps"]
        rdr._nbyte = rdr.f.tell() - startpos + 2


def read_vmdas(rdr):
    """Read VMDAS Navigation block"""
    fd = rdr.f
    rdr.cfg["sourceprog"] = "VMDAS"
    ens = rdr.ensemble
    k = ens.k
    if rdr._vm_source != 1 and rdr._debug_level > -1:
        logging.info("  \n***** Apparently a VMDAS file \n\n")
    rdr._vm_source = 1
    rdr.vars_read += [
        "time_gps",
        "clock_offset_UTC_gps",
        "latitude_gps",
        "longitude_gps",
        "avg_speed_gps",
        "avg_dir_gps",
        "speed_made_good_gps",
        "dir_made_good_gps",
        "flags_gps",
        "pitch_gps",
        "roll_gps",
        "heading_gps",
    ]
    # UTC date time
    utim = fd.read_ui8(4)
    date_utc = tmlib.datetime(utim[2] + utim[3] * 256, utim[1], utim[0])

    # 1st lat/lon position after previous ADCP ping
    # This byte is in hundredths of seconds (10s of milliseconds):
    utc_time_first_fix = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) * 0.1)))
    ens.clock_offset_UTC_gps[k] = (
        fd.read_i32(1) * 0.001
    )  # "PC clock offset from UTC" in ms
    latitude_first_gps = fd.read_i32(1) * rdr._cfac32
    longitude_first_gps = fd.read_i32(1) * rdr._cfac32

    # Last lat/lon position prior to current ADCP ping
    utc_time_fix = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) * 0.1)))
    ens.time_gps[k] = tmlib.date2epoch(date_utc + utc_time_fix)[0]
    ens.latitude_gps[k] = fd.read_i32(1) * rdr._cfac32
    ens.longitude_gps[k] = fd.read_i32(1) * rdr._cfac32

    ens.avg_speed_gps[k] = fd.read_ui16(1) * 0.001
    ens.avg_dir_gps[k] = fd.read_ui16(1) * rdr._cfac16  # avg true track
    fd.seek(2, 1)  # avg magnetic track
    ens.speed_made_good_gps[k] = fd.read_ui16(1) * 0.001
    ens.dir_made_good_gps[k] = fd.read_ui16(1) * rdr._cfac16
    fd.seek(2, 1)  # reserved
    ens.flags_gps[k] = int(np.binary_repr(fd.read_ui16(1)))
    fd.seek(6, 1)  # reserved, ADCP ensemble #

    # ADCP date time
    utim = fd.read_ui8(4)
    date_adcp = tmlib.datetime(utim[0] + utim[1] * 256, utim[3], utim[2])
    time_adcp = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) * 0.1)))

    ens.pitch_gps[k] = fd.read_ui16(1) * rdr._cfac16
    ens.roll_gps[k] = fd.read_ui16(1) * rdr._cfac16
    ens.heading_gps[k] = fd.read_ui16(1) * rdr._cfac16

    fd.seek(10, 1)
    rdr._nbyte = 2 + 76

    if rdr._debug_level > -1:
        logging.info("Read VMDAS")
    rdr._read_vmdas = True
