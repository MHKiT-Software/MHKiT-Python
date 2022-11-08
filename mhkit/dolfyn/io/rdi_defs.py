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

data_defs = {'number': ([], 'data_vars', 'uint32', ''),
             'rtc': ([7], 'sys', 'uint16', ''),
             'builtin_test_fail': ([], 'data_vars', 'bool', ''),
             'c_sound': ([], 'data_vars', 'float32', 'm/s'),
             'depth': ([], 'data_vars', 'float32', 'm'),
             'pitch': ([], 'data_vars', 'float32', 'deg'),
             'roll': ([], 'data_vars', 'float32', 'deg'),
             'heading': ([], 'data_vars', 'float32', 'deg'),
             'temp': ([], 'data_vars', 'float32', 'C'),
             'salinity': ([], 'data_vars', 'float32', 'psu'),
             'min_preping_wait': ([], 'data_vars', 'float32', 's'),
             'heading_std': ([], 'data_vars', 'float32', 'deg'),
             'pitch_std': ([], 'data_vars', 'float32', 'deg'),
             'roll_std': ([], 'data_vars', 'float32', 'deg'),
             'adc': ([8], 'sys', 'uint8', ''),
             'error_status': ([], 'attrs', 'float32', ''),
             'pressure': ([], 'data_vars', 'float32', 'dbar'),
             'pressure_std': ([], 'data_vars', 'float32', 'dbar'),
             'vel': (['nc', 4], 'data_vars', 'float32', 'm/s'),
             'amp': (['nc', 4], 'data_vars', 'uint8', 'counts'),
             'corr': (['nc', 4], 'data_vars', 'uint8', 'counts'),
             'prcnt_gd': (['nc', 4], 'data_vars', 'uint8', '%'),
             'status': (['nc', 4], 'data_vars', 'float32', ''),
             'dist_bt': ([4], 'data_vars', 'float32', 'm'),
             'vel_bt': ([4], 'data_vars', 'float32', 'm/s'),
             'corr_bt': ([4], 'data_vars', 'uint8', 'counts'),
             'amp_bt': ([4], 'data_vars', 'uint8', 'counts'),
             'prcnt_gd_bt': ([4], 'data_vars', 'uint8', '%'),
             'time': ([], 'coords', 'float64', ''),
             'alt_dist': ([], 'data_vars', 'float32', 'm'),
             'alt_rssi': ([], 'data_vars', 'uint8', 'dB'),
             'alt_eval': ([], 'data_vars', 'uint8', 'dB'),
             'alt_status': ([], 'data_vars', 'uint8', 'bit'),
             'time_gps': ([], 'coords', 'float64', ''),
             'clock_offset_UTC_gps': ([], 'data_vars', 'float64', 's'),
             'latitude_gps': ([], 'data_vars', 'float32', 'deg'),
             'longitude_gps': ([], 'data_vars', 'float32', 'deg'),
             'avg_speed_gps': ([], 'data_vars', 'float32', 'm/s'),
             'avg_dir_gps': ([], 'data_vars', 'float32', 'deg'),
             'speed_made_good_gps': ([], 'data_vars', 'float32', 'm/s'),
             'dir_made_good_gps': ([], 'data_vars', 'float32', 'deg'),
             'flags_gps': ([], 'data_vars', 'float32', 'bits'),
             'fix_gps': ([], 'data_vars', 'int8', '1'),
             'n_sat_gps': ([], 'data_vars', 'int8', 'count'),
             'hdop_gps': ([], 'data_vars', 'float32', '1'),
             'elevation_gps': ([], 'data_vars', 'float32', 'm'),
             'rtk_age_gps': ([], 'data_vars', 'float32', 's'),
             'speed_over_grnd_gps': ([], 'data_vars', 'float32', 'm/s'),
             'dir_over_grnd_gps': ([], 'data_vars', 'float32', 'deg'),
             'heading_gps': ([], 'data_vars', 'float64', 'deg'),
             'dist_nmea': ([], 'data_vars', 'float32', 'm'),
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
    arr = np.empty(sz, dtype=dtype)
    if dtype.startswith('float'):
        arr[:] = np.NaN
    dat[group][nm] = arr
    dat['units'][nm] = units
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
