import numpy as np

century = 2000
adcp_type = {4: 'Broadband',
             5: 'Broadband',
             6: 'Navigator',
             10: 'Rio Grande',
             11: 'H-ADCP',
             14: 'Ocean Surveyor',
             16: 'Workhorse',
             19: 'Navigator',
             23: 'Ocean Surveyor',
             28: 'ChannelMaster',
             31: 'StreamPro',
             34: 'Explorer',
             37: 'Navigator',
             41: 'DVS',
             43: 'Workhorse',
             44: 'RiverRay',
             47: 'SentinelV',
             50: 'Workhorse',
             51: 'Workhorse',
             52: 'Workhorse',
             53: 'Navigator',
             55: 'DVS',
             56: 'RiverPro',
             59: 'Meridian',
             61: 'Pinnacle',
             66: 'SentinelV',
             67: 'Pathfinder',
             73: 'Pioneer',
             74: 'Tasman',
             76: 'WayFinder',
             77: 'Workhorse',
             78: 'Workhorse',
             }

data_defs = {'number': ([], 'data_vars', 'uint32', '1', 'Ensemble Number', 'number_of_observations'),
             'rtc': ([7], 'sys', 'uint16', '1', '', ''),
             'builtin_test_fail': ([], 'data_vars', 'bool', '1', 'Built-In Test Failures', 'built_in_test'),
             'c_sound': ([], 'data_vars', 'float32', 'm s-1', 'Speed of Sound', 'speed_of_sound_in_sea_water'),
             'depth': ([], 'data_vars', 'float32', 'm', 'Depth', 'depth_below_platform'),
             'pitch': ([], 'data_vars', 'float32', 'degree', 'Pitch', 'platform_pitch'),
             'roll': ([], 'data_vars', 'float32', 'degree', 'Roll', 'platform_roll'),
             'heading': ([], 'data_vars', 'float32', 'degree', 'Heading', 'platform_orientation'),
             'temp': ([], 'data_vars', 'float32', 'degree_C', 'Temperature', 'sea_water_temperature'),
             'salinity': ([], 'data_vars', 'float32', 'psu', 'Salinity', 'sea_water_salinity'),
             'min_preping_wait': ([], 'data_vars', 'float32', 's', 'Minimum Pre-Ping Wait Time', 'time_between_measurements'),
             'heading_std': ([], 'data_vars', 'float32', 'degree', 'Heading Std', 'platform_course_standard_devation'),
             'pitch_std': ([], 'data_vars', 'float32', 'degree', 'Pitch Std', 'platform_pitch_standard_devation'),
             'roll_std': ([], 'data_vars', 'float32', 'degree', 'Roll Std', 'platform_roll_standard_devation'),
             'adc': ([8], 'sys', 'uint8', '1', '', ''),
             'error_status': ([], 'attrs', 'float32', '1', '', ''),
             'pressure': ([], 'data_vars', 'float32', 'dbar', 'Pressure', 'sea_water_pressure'),
             'pressure_std': ([], 'data_vars', 'float32', 'dbar', 'Pressure Std', 'sea_water_pressure_standard_devation'),
             'vel': (['nc', 4], 'data_vars', 'float32', 'm s-1', 'Water Velocity',
                     'velocity_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'amp': (['nc', 4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Amplitude',
                     'signal_intensity_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'corr': (['nc', 4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Correlation',
                      'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'prcnt_gd': (['nc', 4], 'data_vars', 'uint8', '%', 'Percent Good',
                          'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'),
             'status': (['nc', 4], 'data_vars', 'float32', '1', '', ''),
             'dist_bt': ([4], 'data_vars', 'float32', 'm', 'Bottom Track Depth', 'depth_below_platform'),
             'vel_bt': ([4], 'data_vars', 'float32', 'm s-1', 'Bottom Track Velocity',
                        'platform_velocity_from_bottom_track'),
             'corr_bt': ([4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Correlation',
                         'beam_consistency_indicator_from_bottom_track'),
             'amp_bt': ([4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Amplitude',
                        'signal_intensity_from_bottom_track'),
             'prcnt_gd_bt': ([4], 'data_vars', 'uint8', '%', 'Percent Good',
                             'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'),
             'time': ([], 'coords', 'float64', 'seconds since 1970-01-01 00:00:00', 'Time', 'time'),
             'alt_dist': ([], 'data_vars', 'float32', 'm', 'Altimeter Range', 'altimeter_range'),
             'alt_rssi': ([], 'data_vars', 'uint8', 'dB', 'Altimeter RSSI', 'altimeter_recieved_signal_strength_indicator'),
             'alt_eval': ([], 'data_vars', 'uint8', 'dB', 'Altimeter Evaluation Amplitude', 'altimeter_signal_intensity'),
             'alt_status': ([], 'data_vars', 'uint8', 'bit', 'Altimeter Status', 'altimeter_status'),
             'time_gps': ([], 'coords', 'float64', 'seconds since 1970-01-01 00:00:00', 'Time', 'time'),
             'clock_offset_UTC_gps': ([], 'data_vars', 'float64', 's', 'Instrument Clock Offset from UTC', 'clock_offset_from_utc'),
             'latitude_gps': ([], 'data_vars', 'float32', 'deg N', 'Latitude', 'latitude'),
             'longitude_gps': ([], 'data_vars', 'float32', 'deg E', 'Longitude', 'longitude'),
             'avg_speed_gps': ([], 'data_vars', 'float32', 'm s-1', 'Average Platform Speed', 'average_platform_speed_wrt_ground'),
             'avg_dir_gps': ([], 'data_vars', 'float32', 'degree', 'Average Platform Direction', 'average_platform_course'),
             'speed_made_good_gps': ([], 'data_vars', 'float32', 'm s-1', 'Platform Speed Made Good', 'platform_speed_made_good'),
             'dir_made_good_gps': ([], 'data_vars', 'float32', 'degree', 'Platform Direction Made Good', 'platform_course_made_good'),
             'flags_gps': ([], 'data_vars', 'float32', 'bits', 'GPS Flags', 'gps_flags'),
             'fix_gps': ([], 'data_vars', 'int8', '1', 'GPS Fix', 'gps_fix_type'),
             'n_sat_gps': ([], 'data_vars', 'int8', 'count', 'Number of Satellites', 'number_of_satellites'),
             'hdop_gps': ([], 'data_vars', 'float32', '1', 'HDOP', 'horizontal_dilution_of_precision'),
             'elevation_gps': ([], 'data_vars', 'float32', 'm', 'Elevation', 'elevation_above_MLLW'),
             'rtk_age_gps': ([], 'data_vars', 'float32', 's', 'RTK Age', 'age_of_received_real_time_kinetic_signal'),
             'speed_over_grnd_gps': ([], 'data_vars', 'float32', 'm s-1', 'Platform Speed', 'platform_speed_wrt_ground'),
             'dir_over_grnd_gps': ([], 'data_vars', 'float32', 'degree', 'Platform Direction', 'platform_course'),
             'heading_gps': ([], 'data_vars', 'float64', 'degree', 'GPS Heading', 'platform_course'),
             'dist_nmea': ([], 'data_vars', 'float32', 'm', 'Depth', 'depth_sounder_range'),
             'vel_sl': (['nc', 4], 'data_vars', 'float32', 'm s-1', 'Water Velocity',
                        'velocity_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'corr_sl': (['nc', 4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Correlation',
                         'beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'amp_sl': (['nc', 4], 'data_vars', 'uint8', 'counts', 'Acoustic Signal Amplitude',
                        'signal_intensity_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water'),
             'prcnt_gd_sl': (['nc', 4], 'data_vars', 'uint8', '%', 'Percent Good',
                             'proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water'),
             'status_sl': (['nc', 4], 'data_vars', 'float32', '1', '', ''),
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
    if dtype.startswith('float'):
        arr[:] = np.NaN
    dat[group][nm] = arr
    dat['units'][nm] = units
    dat['long_name'][nm] = long_name
    dat['standard_name'][nm] = standard_name
    return dat


def _get_size(name, n=None, ncell=0):
    sz = list(data_defs[name][0])  # create a copy!
    if 'nc' in sz:
        sz.insert(sz.index('nc'), ncell)
        sz.remove('nc')
    if n is None:
        return tuple(sz)
    return tuple(sz + [n])


class _variable_setlist(set):
    def __iadd__(self, vals):
        if vals[0] not in self:
            self |= set(vals)
        return self


class _ensemble():
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
            setattr(self, nm,
                    np.zeros(_get_size(nm, n=navg, ncell=n_cells),
                             dtype=data_defs[nm][2]))

    def clean_data(self,):
        self['vel'][self['vel'] == -32.768] = np.NaN
