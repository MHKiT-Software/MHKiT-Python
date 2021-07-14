from __future__ import print_function
import numpy as np
import xarray as xr
import datetime
import warnings
from os.path import getsize

from ._read_bin import eofException, bin_reader
from .base import WrongFileType, read_userdata, create_dataset
from ..rotate.rdi import calc_beam_orientmat as _calc_bmat, calc_orientmat as _calc_omat
from ..rotate.base import _set_coords
from ..rotate.api import set_declination


def read_rdi(fname, userdata=None, nens=None, debug=0):
    """
    Read a TRDI binary data file.

    Parameters
    ----------
    filename : string
               Filename of Nortek file to read.

    userdata : True, False, or string of userdata.json filename
               (default ``True``) Whether to read the
               '<base-filename>.userdata.json' file.

    nens : None (default: read entire file), int, or
           2-element tuple (start, stop)
              Number of pings to read from the file

    Returns
    -------
    adp_data : :class:`ADPdata <dolfyn.adp.base.ADPdata>`

    """
    # Reads into a dictionary of dictionaries using netcdf naming conventions
    # Should be easier to debug
    with adcp_loader(fname, debug_level=debug) as ldr:
        dat = ldr.load_data(nens=nens)
        
    # Read in userdata    
    userdata = read_userdata(fname, userdata)
    for nm in userdata:
        dat['attrs'][nm] = userdata[nm]

    # If GPS data is sampling at a different rate for WinRiver or VMDAS
    if 'time_gps' in dat['coords']:
        dat['coords']['time_gps'], idx = np.unique(dat['coords']['time_gps'],
                                                   return_index=True)
        nan = np.zeros(dat['coords']['time'].shape, dtype=bool)
        if any(np.isnan(dat['coords']['time_gps'])):
            nan = np.isnan(dat['coords']['time_gps'])
            dat['coords']['time_gps'] = dat['coords']['time_gps'][~nan]
            
        for key in dat['data_vars']:
            if 'gps' in key:
                dat['data_vars'][key] = dat['data_vars'][key][idx]
                if sum(nan) > 0:
                    dat['data_vars'][key] = dat['data_vars'][key][~nan]
    
    # Create xarray dataset from upper level dictionary
    ds = create_dataset(dat)
    ds = _set_coords(ds, ref_frame=ds.coord_sys)
    
    # Create orientation matrices
    ds['beam2inst_orientmat'] = xr.DataArray(_calc_bmat(
                            ds.beam_angle,
                            ds.beam_pattern=='convex'),
                                             coords={'x':[1,2,3,4],
                                                     'x*':[1,2,3,4]},
                                             dims=['x','x*'])
                                 
    ds['orientmat'] = xr.DataArray(_calc_omat(ds),
                                   coords={'inst': ['X','Y','Z'],
                                           'earth': ['E','N','U'], 
                                           'time': ds['time']},
                                   dims=['inst','earth','time'])
    
    # Check magnetic declination if provided via software and/or userdata
    ds = _set_rdi_declination(ds, fname)
    
    return ds


def _set_rdi_declination(dat, fname='????'):
    # NEED TO CONFIRM: If magnetic_var_deg is set, this means
    # that the declination is already included in the heading,
    # and in the velocity data.
    # I'm assuming this is the case for now...
    ## Can confirm - jrm

    declin = dat.attrs.pop('declination', None) # userdata declination

    if dat.attrs['magnetic_var_deg'] != 0: # from TRDI software if set
        dat.attrs['declination'] = dat.attrs['magnetic_var_deg']
        dat.attrs['declination_in_orientmat'] = 1 # logical

    if dat.attrs['magnetic_var_deg'] != 0 and declin is not None:
        warnings.warn(
            "'magnetic_var_deg' is set to {:.2f} degrees in the binary "
            "file '{}', AND 'declination' is set in the 'userdata.json' "
            "file. DOLfYN WILL USE THE VALUE of {:.2f} degrees in "
            "userdata.json. If you want to use the value in "
            "'magnetic_var_deg', delete the value from userdata.json and "
            "re-read the file."
            .format(dat.attrs['magnetic_var_deg'], fname, declin))
        dat.attrs['declination'] = declin

    if declin is not None:
        dat = set_declination(dat, declin)
        
    return dat


# Four pound symbols ("####"), indicate a duplication of a comment from
# Rich Pawlawicz' rdadcp routines.

#century=1900
century = 2000

# changed group to xarray info type (data_var, coord, dim, attr)
# added units to end
data_defs = {'number': ([], 'sys', 'uint32', ''),
             'rtc': ([7], 'sys', 'uint16', ''),
             'bit': ([], 'sys', 'bool', ''),
             'ssp': ([], 'sys', 'uint16', ''),
             'depth': ([], 'data_vars', 'float32', 'm'),
             'pitch': ([], 'data_vars', 'float32', 'deg'),
             'roll': ([], 'data_vars', 'float32', 'deg'),
             'heading': ([], 'data_vars', 'float32', 'deg'),
             'temp': ([], 'data_vars', 'float32', 'C'),
             'salinity': ([], 'data_vars', 'float32', 'psu'),
             'mpt_sec': ([], 'sys', 'float32', 's'),
             'heading_std': ([], 'sys', 'float32', 'deg'),
             'pitch_std': ([], 'sys', 'float32', 'deg'),
             'roll_std': ([], 'sys', 'float32', 'deg'),
             'adc': ([8], 'sys', 'uint16', ''),
             'error_status_wd': ([], 'attrs', 'float32', ''),
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
             'etime_gps': ([], 'coords', 'float64', ''),
             'elatitude_gps': ([], 'data_vars', 'float64', 'deg'),
             'elongitude_gps': ([], 'data_vars', 'float64', 'deg'),
             'time_gps': ([], 'coords', 'float64', ''),
             'latitude_gps': ([], 'data_vars', 'float64', 'deg'),
             'longitude_gps': ([], 'data_vars', 'float64', 'deg'),
             'ntime': ([], 'coords', 'float64', ''),
             'flags': ([], 'data_vars', 'float32', ''),
             }


# These are shortcut functions (so that I don't have to do this `if
# grp` over and over)
def get(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        return dat[nm]
    else:
        return dat[grp][nm]


def in_group(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        return nm in dat
    else:
        return nm in dat[grp]


def pop(dat, nm):
    grp = data_defs[nm][1]
    if grp is None:
        dat.pop(nm)
    else:
        dat[grp].pop(nm)


def setd(dat, nm, val):
    grp = data_defs[nm][1]
    if grp is None:
        dat[nm] = val
    else:
        dat[grp][nm] = val


def idata(dat, nm, sz):
    # Complete
    group = data_defs[nm][1]
    dtype = data_defs[nm][2]
    units = data_defs[nm][3]
    arr = np.empty(sz, dtype=dtype)
    if dtype.startswith('float'):
        arr[:] = np.NaN
    #if group is None:
    #    dat[nm] = arr
    #else:
        # if group not in dat:
        #     dat[group] = data()
    dat[group][nm] = arr
    dat['units'][nm] = units
    return dat


class variable_setlist(set):
    def __iadd__(self, vals):
        if vals[0] not in self:
            self |= set(vals)
        return self


def get_size(name, n=None, ncell=0):
    sz = list(data_defs[name][0])  # create a copy!
    if 'nc' in sz:
        sz.insert(sz.index('nc'), ncell)
        sz.remove('nc')
    if n is None:
        return tuple(sz)
    return tuple(sz + [n])


class ensemble(object):
    n_avg = 1
    k = -1  # This is the counter for filling the ensemble object

    def __getitem__(self, nm):
        return getattr(self, nm)

    def __init__(self, navg, n_cells):
        if navg is None or navg == 0:
            navg = 1
        self.n_avg = navg
        for nm in data_defs:
            setattr(self, nm,
                    np.zeros(get_size(nm, n=navg, ncell=n_cells),
                             dtype=data_defs[nm][2]))

    def clean_data(self,):
        self['vel'][self['vel'] == -32.768] = np.NaN


class adcp_loader(object):
    _n_beams = 4  # Placeholder for 5-beam adcp, not currently used.
    _pos = 0
    progress = 0
    _cfgnames = dict.fromkeys([4, 5], 'bb-adcp')
    _cfgnames.update(dict.fromkeys([8, 9, 16], 'wh-adcp'))
    _cfgnames.update(dict.fromkeys([14, 23], 'os-adcp'))
    _cfac = 180 / 2 ** 31
    _source = 0
    _fixoffset = 0
    _nbyte = 0
    _winrivprob = False
    _search_num = 30000  # Maximum distance? to search
    _debug7f79 = None
    extrabytes = 0

    # def _debug_print(self, lvl, msg):
    #     if self._debug_level > lvl:
    #         print(msg)
    
    def __exit__(self, type, value, traceback):
        self.f.close()

    def __enter__(self,):
        return self

    def __init__(self, fname, navg=1, avg_func='mean', debug_level=0):
        self.fname = fname
        print('\nReading file {} ...'.format(fname))
        self._debug_level = debug_level
        self.cfg = {} #config(_type='ADCP')
        self.cfg['name'] = 'wh-adcp'
        self.cfg['sourceprog'] = 'instrument'
        self.cfg['prog_ver'] = 0
        self.hdr = {} #config(_type='RDI-HEADER')
        #self.f=io.npfile(fname,'r','l')
        self.f = bin_reader(fname)
        self.read_hdr()
        self.read_cfg()
        # Seek back to the beginning of the file:
        self.f.seek(self._pos, 0)
        self.n_avg = navg
        self.ensemble = ensemble(self.n_avg, self.cfg['n_cells'])
        self._filesize = getsize(fname)
        self._npings = int(self._filesize / (self.hdr['nbyte'] + 2 +
                                             self.extrabytes))
        self.vars_read = variable_setlist(['time'])
        
        if self._debug_level > 0:
            print('  %d pings estimated in this file' % self._npings)
        self.avg_func = getattr(self, avg_func)

    def mean(self, dat):
        if self.n_avg == 1:
            return dat[..., 0]
        return np.nanmean(dat, axis=-1)

    def print_progress(self,):
        self.progress = self.f.tell()
        if self._debug_level > 1:
            print('  pos %0.0fmb/%0.0fmb\r' %
                  (self.f.tell() / 1048576., self._filesize / 1048576.))
        if (self.f.tell() - self.progress) < 1048576:
            return

    def print_pos(self, byte_offset=-1):
        """
        Print the position in the file, used for debugging.
        """
        if self._debug_level > 3:
            if hasattr(self, 'ensemble'):
                k = self.ensemble.k
            else:
                k = 0
            print('  pos: %d, pos_: %d, nbyte: %d, k: %d, byte_offset: %d' %
                  (self.f.tell(), self._pos, self._nbyte, k, byte_offset))

    def read_dat(self, id):
        function_map = {0: (self.read_fixed, []),   # 0000
                        128: (self.read_var, []),     # 0080
                        256: (self.read_vel, []),     # 0100
                        512: (self.read_corr, []),    # 0200
                        768: (self.read_amp, []),    # 0300
                        1024: (self.read_prcnt_gd, []),  # 0400
                        1280: (self.read_status, []),  # 0500
                        1536: (self.read_bottom, []),  # 0600
                        8192: (self.read_vmdas, []),   # 2000
                        8226: (self.read_winriver2, []),  # 2022
                        8448: (self.read_winriver, [38]),  # 2100
                        8449: (self.read_winriver, [97]),  # 2101
                        8450: (self.read_winriver, [45]),  # 2102
                        8451: (self.read_winriver, [60]),  # 2103
                        8452: (self.read_winriver, [38]),  # 2104
                        # Loading of these data is currently not implemented:
                        1793: (self.skip_Ncol, [4]),  # 0701 number of pings
                        1794: (self.skip_Ncol, [4]),  # 0702 sum of squared vel
                        1795: (self.skip_Ncol, [4]),  # 0703 sum of velocities
                        2560: (self.skip_Ncol, []),  # 0A00 Beam 5 velocity
                        # 0301 Beam 5 Number of good pings
                        769: (self.skip_Ncol, []),
                        # 0302 Beam 5 Sum of squared velocities
                        770: (self.skip_Ncol, []),
                        # 0303 Beam 5 Sum of velocities
                        771: (self.skip_Ncol, []),
                        # 020C Ambient sound profile
                        524: (self.skip_Nbyte, [4]),
                        12288: (self.skip_Nbyte, [32]),
                        # 3000 Fixed attitude data format for OS-ADCPs
                        # #### This is pretty idiotic - for OS-ADCPs
                        # (phase 2) they suddenly decided to code the
                        # number of bytes into the header ID word. And
                        # then they don't really document what they
                        # did! So, this is cruft of a high order, and
                        # although it works on the one example I have
                        # - caveat emptor....
                        }
        # Call the correct function:
        if id in function_map:
            if self._debug_level >= 2:
                print('  Reading code {}...'.format(hex(id)), end='')
            retval = function_map.get(id)[0](*function_map[id][1])
            if retval:
                return retval
            if self._debug_level >= 2:
                print('    success!')
        else:
            self.read_nocode(id)

    def read_nocode(self, id):
        # #### There appear to be codes 0340-03FC to deal with. I am not
        # #### going to decode them, but I am going to try to figure out
        # #### how many bytes to skip.
        # # Does he mean "3040-30FC", or is the code wrong?  Hopefully the
        # # former.
        # hxid = hex(id)
        # if hxid[2:4] == '30':
        #     raise Exception("")
        #     # I want to count the number of 1s in the middle 4 bits
        #     # of the 2nd two bytes.
        #     # 60 is a 0b00111100 mask
        #     nflds = (bin(int(hxid[3]) & 60).count('1') +
        #              bin(int(hxid[4]) & 60).count('1'))
        #     # I want to count the number of 1s in the highest
        #     # 2 bits of byte 3
        #     # 3 is a 0b00000011 mask:
        #     dfac = bin(int(hxid[3], 0) & 3).count('1')
        #     self.skip_Nbyte(12 * nflds * dfac)
        # else:
        print('  Unrecognized ID code: %0.4X\n' % id)

    def read_fixed(self,):
        if hasattr(self, 'configsize'):  # and False:
            # Skipping the cfgseg was something I added,
            # because it seemed unnecessary to read it every
            # time, so I just skip it after the first read.
            # If this causes problems I may need to remove this.
            # The other option may be that if a problem is encountered, I could
            # come back here and re-read the header.
            self.f.seek(self.configsize, 1)
            self._nbyte = self.configsize
        else:
            self.read_cfgseg()
        if self._debug_level >= 1:
            print(self._pos)
        self._nbyte += 2

    def read_var(self,):
        """ Read variable leader """
        fd = self.f
        self.ensemble.k += 1  # Increment k.
        ens = self.ensemble
        k = ens.k
        self.vars_read += ['number',
                           'rtc',
                           'number',
                           'bit',
                           'ssp',
                           'depth',
                           'heading',
                           'pitch',
                           'roll',
                           'salinity',
                           'temp',
                           'mpt_sec',
                           'heading_std',
                           'pitch_std',
                           'roll_std',
                           'adc']
        ens.number[k] = fd.read_ui16(1)
        ens.rtc[:, k] = fd.read_ui8(7)
        ens.number[k] += 65535 * fd.read_ui8(1)
        ens.bit[k] = fd.read_ui16(1)
        ens.ssp[k] = fd.read_ui16(1)
        ens.depth[k] = fd.read_ui16(1) * 0.1
        ens.heading[k] = fd.read_ui16(1) * 0.01
        ens.pitch[k] = fd.read_i16(1) * 0.01
        ens.roll[k] = fd.read_i16(1) * 0.01
        ens.salinity[k] = fd.read_i16(1)
        ens.temp[k] = fd.read_i16(1) * 0.01
        ens.mpt_sec[k] = (fd.read_ui8(3) * np.array([60, 1, .01])).sum()
        ens.heading_std[k] = fd.read_ui8(1)
        ens.pitch_std[k] = fd.read_i8(1) * 0.1
        ens.roll_std[k] = fd.read_i8(1) * 0.1
        ens.adc[:, k] = fd.read_ui8(8)
        self._nbyte = 2 + 40
        
        #!!! TRDI changed pressure to depth?
        #cfg = self.cfg
        # if cfg['name'] == 'bb-adcp':
        #     if cfg['prog_ver'] >= 5.55:
        #         fd.seek(15, 1)
        #         cent = fd.read_ui8(1)
        #         ens.rtc[:, k] = fd.read_ui8(7)
        #         ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
        #         self._nbyte += 23
        # elif cfg['name'] == 'wh-adcp':
        #     ens.error_status_wd[k] = fd.read_ui32(1)
        #     self.vars_read += ['error_status_wd', 'pressure', 'pressure_std', ]
        #     self._nbyte += 4
        #     if (np.fix(cfg['prog_ver']) == [8, 16]).any():
        #         if cfg['prog_ver'] >= 8.13:
        #             # Added pressure sensor stuff in 8.13
        #             fd.seek(2, 1)
        #             ens.pressure[k] = fd.read_ui32(1)
        #             ens.pressure_std[k] = fd.read_ui32(1)
        #             self._nbyte += 10
        #         if cfg['prog_ver'] >= 8.24:
        #             # Spare byte added 8.24
        #             fd.seek(1, 1)
        #             self._nbyte += 1
        #         if cfg['prog_ver'] >= 16.05:
        #             # Added more fields with century in clock
        #             cent = fd.read_ui8(1)
        #             ens.rtc[:, k] = fd.read_ui8(7)
        #             ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
        #             self._nbyte += 8
        #     elif np.fix(cfg['prog_ver']) == 9:
        #         fd.seek(2, 1)
        #         ens.pressure[k] = fd.read_ui32(1)
        #         ens.pressure_std[k] = fd.read_ui32(1)
        #         self._nbyte += 10
        #         if cfg['prog_ver'] >= 9.10:  # Spare byte added...
        #             fd.seek(1, 1)
        #             self._nbyte += 1
        # elif cfg['name'] == 'os-adcp':
        #     fd.seek(16, 1)  # 30 bytes all set to zero, 14 read above
        #     self._nbyte += 16
        #     if cfg['prog_ver'] > 23:
        #         fd.seek(2, 1)
        #         self._nbyte += 2

    def read_vel(self,):
        ens = self.ensemble
        # if self.cfg['coord_sys'] == 'beam':
        #     var_nms = ['beam1vel', 'beam2vel', 'beam3vel', 'beam4vel']
        # else:
        #     var_nms = ['u', 'v', 'w', 'err_vel']
        self.vars_read += ['vel']
        k = ens.k
        ens['vel'][:, :, k] = np.array(
            self.f.read_i16(4 * self.cfg['n_cells'])
        ).reshape((self.cfg['n_cells'], 4)) * .001
        self._nbyte = 2 + 4 * self.cfg['n_cells'] * 2

    def read_corr(self,):
        k = self.ensemble.k
        self.vars_read += ['corr']
        self.ensemble.corr[:, :, k] = np.array(
            self.f.read_ui8(4 * self.cfg['n_cells'])
        ).reshape((self.cfg['n_cells'], 4))
        self._nbyte = 2 + 4 * self.cfg['n_cells']

    def read_amp(self,):
        k = self.ensemble.k
        self.vars_read += ['amp']
        self.ensemble.amp[:, :, k] = np.array(
            self.f.read_ui8(4 * self.cfg['n_cells'])
        ).reshape((self.cfg['n_cells'], 4))
        self._nbyte = 2 + 4 * self.cfg['n_cells']

    def read_prcnt_gd(self,):
        self.vars_read += ['prcnt_gd']
        self.ensemble.prcnt_gd[:, :, self.ensemble.k] = np.array(
            self.f.read_ui8(4 * self.cfg['n_cells'])
        ).reshape((self.cfg['n_cells'], 4))
        self._nbyte = 2 + 4 * self.cfg['n_cells']

    def read_status(self,):
        self.vars_read += ['status']
        self.ensemble.status[:, :, self.ensemble.k] = np.array(
            self.f.read_ui8(4 * self.cfg['n_cells'])
        ).reshape((self.cfg['n_cells'], 4))
        self._nbyte = 2 + 4 * self.cfg['n_cells']

    def read_bottom(self,):
        self.vars_read += ['dist_bt', 'vel_bt', 'corr_bt', 'amp_bt',
                           'prcnt_gd_bt']
        fd = self.f
        ens = self.ensemble
        k = ens.k
        cfg = self.cfg
        if self._source == 2:
            print('Warning: lat/lon bottom reading passed')
            # self.vars_read += ['latitude_gps', 'longitude_gps']
            # fd.seek(2, 1)
            # long1 = fd.read_ui16(1)
            # fd.seek(6, 1)
            # ens.latitude_gps[k] = fd.read_i32(1) * self._cfac
            # if ens.latitude_gps[k] == 0:
            #     ens.latitude_gps[k] = np.NaN
        else:
            fd.seek(14, 1)
        ens.dist_bt[:, k] = fd.read_ui16(4) * 0.01
        ens.vel_bt[:, k] = fd.read_i16(4) * 0.001
        ens.corr_bt[:, k] = fd.read_ui8(4)
        ens.amp_bt[:, k] = fd.read_ui8(4)
        ens.prcnt_gd_bt[:, k] = fd.read_ui8(4)
        if self._source == 2:
            print('Warning: lat/lon bottom reading passed')
            # fd.seek(2, 1)
            # ens.longitude_gps[k] = (long1 + 65536 * fd.read_ui16(1)) * self._cfac
            # if ens.longitude_gps[k] > 180:
            #     ens.longitude_gps[k] = ens.longitude_gps[k] - 360
            # if ens.longitude_gps[k] == 0:
            #     ens.longitude_gps[k] = np.NaN
            # fd.seek(16, 1)
            # qual = fd.read_ui8(1)
            # if qual == 0:
            #     print('  qual==%d,%f %f' % (qual,
            #                                 ens.latitude_gps[k],
            #                                 ens.longitude_gps[k]))
            #     ens.latitude_gps[k] = np.NaN
            #     ens.longitude_gps[k] = np.NaN
            # fd.seek(71 - 45 - 16 - 17, 1)
            # self._nbyte = 2 + 68
        else:
            fd.seek(71 - 45, 1)
            self._nbyte = 2 + 68
        if cfg['prog_ver'] >= 5.3:
            fd.seek(78 - 71, 1)
            ens.dist_bt[:, k] = ens.dist_bt[:, k] + fd.read_ui8(4) * 655.36
            self._nbyte += 11
            #!!! Assumption: never running bottom-tracking without WinRiver or VMDAS
            # if cfg['name'] == 'wh-adcp':
            #     if cfg['prog_ver'] >= 16.20:
            #         # RDI documentation claims these extra bytes were added in
            #         # v8.17, but they don't appear in my 8.33 data -
            #         # conversation with Egil suggests they were added in 16.20
            #         fd.seek(4, 1)
            #         self._nbyte += 4

    def read_vmdas(self,):
        """ Read something from VMDAS """
        fd = self.f
        # The raw files produced by VMDAS contain a binary navigation data
        # block.
        self.cfg['sourceprog'] = 'VMDAS'
        ens = self.ensemble
        k = ens.k
        if self._source != 1 and self._debug_level >= 1:
            print('  \n***** Apparently a VMDAS file \n\n')
        self._source = 1
        self.vars_read += ['time_gps',
                           'latitude_gps',
                           'longitude_gps',
                           'etime_gps',
                           'elatitude_gps',
                           'elongitude_gps',
                           'flags',
                           'ntime', ]
        utim = fd.read_ui8(4)
        date = datetime.datetime(utim[2] + utim[3] * 256, utim[1], utim[0])
        # This byte is in hundredths of seconds (10s of milliseconds):
        time = datetime.timedelta(milliseconds=(int(fd.read_ui32(1) / 10)))
        fd.seek(4, 1)  # "PC clock offset from UTC" - clock drift in ms?
        ens.time_gps[k] = (date + time).timestamp()
        ens.latitude_gps[k] = fd.read_i32(1) * self._cfac
        ens.longitude_gps[k] = fd.read_i32(1) * self._cfac
        ens.etime_gps[k] = (date + datetime.timedelta(
            milliseconds=int(fd.read_ui32(1) * 10))).timestamp()
        ens.elatitude_gps[k] = fd.read_i32(1) * self._cfac
        ens.elongitude_gps[k] = fd.read_i32(1) * self._cfac
        fd.seek(12, 1)
        ens.flags[k] = fd.read_ui16(1)
        fd.seek(6, 1)
        utim = fd.read_ui8(4)
        date = datetime.datetime(utim[0] + utim[1] * 256, utim[3], utim[2])
        ens.ntime[k] = (date + datetime.timedelta(
            milliseconds=int(fd.read_ui32(1) / 10))).timestamp()
        fd.seek(16, 1)
        self._nbyte = 2 + 76

    def read_winriver2(self, ):
        startpos = self.f.tell()
        self._winrivprob = True
        self.cfg['sourceprog'] = 'WINRIVER'
        ens = self.ensemble
        k = ens.k
        if self._source != 3 and self._debug_level >= 1:
            print('  \n***** Apparently a WINRIVER2 file\n'
                  '*****    WARNING: Raw NMEA data '
                  'handler not yet fully implemented\n\n')
        self._source = 3
        spid = self.f.read_ui16(1)
        if spid == 104:
            sz = self.f.read_ui16(1)
            dtime = self.f.read_f64(1)
            start_string = self.f.reads(6)
            _ = self.f.reads(1)
            if start_string != '$GPGGA':
                if self._debug_level > 1:
                    print(
                        '  WARNING: Invalid GPGGA string found in ensemble {},'
                        ' skipping...'.format(k))
                return 'FAIL'
            gga_time = str(self.f.reads(9))
            time = datetime.timedelta(hours=int(gga_time[0:2]), 
                                      minutes=int(gga_time[2:4]),
                                      seconds=int(gga_time[4:6]),
                                      milliseconds=int(gga_time[7:])*100)
            clock = self.ensemble.rtc[:, :]
            if clock[0, 0] < 100:
                clock[0, :] += century
            ens.time_gps[k] = (datetime.datetime(*clock[:3, 0]) + time).timestamp()
            self.f.seek(1, 1)
            ens.latitude_gps[k] = self.f.read_f64(1)
            tcNS = self.f.reads(1)
            if tcNS == 'S':
                ens.latitude_gps[k] *= -1
            elif tcNS != 'N':
                if self._debug_level > 1:
                    print(
                        '  WARNING: Invalid GPGGA string found in ensemble {},'
                        ' skipping...'.format(k))
                return 'FAIL'
            ens.longitude_gps[k] = self.f.read_f64(1)
            tcEW = self.f.reads(1)
            if tcEW == 'W':
                ens.longitude_gps[k] *= -1
            elif tcEW != 'E':
                if self._debug_level > 1:
                    print(
                        '  WARNING: Invalid GPGGA string found in ensemble {},'
                        ' skipping...'.format(k))
                return 'FAIL'
            ucqual, n_sat = self.f.read_ui8(2)
            tmp = self.f.read_float(2)
            ens.hdop, ens.altitude = tmp
            if self.f.reads(1) != 'M':
                if self._debug_level > 1:
                    print(
                        '  WARNING: Invalid GPGGA string found in ensemble {},'
                        ' skipping...'.format(k))
                return 'FAIL'
            ggeoid_sep = self.f.read_float(1)
            if self.f.reads(1) != 'M':
                if self._debug_level > 1:
                    print(
                        '  WARNING: Invalid GPGGA string found in ensemble {},'
                        ' skipping...'.format(k))
                return 'FAIL'
            gage = self.f.read_float(1)
            gstation_id = self.f.read_ui16(1)
            # ## For some reason this was only on the first block that
            # ## I was looking at.
            # I believe these like the following:
            # 4 unknown bytes (2 reserved+2 checksum?)
            # 78 bytes for GPGGA string (including \r\n)
            # 2 reserved + 2 checksum
            self.vars_read += ['longitude_gps', 'latitude_gps', 'time_gps']
            self._nbyte = self.f.tell() - startpos + 2
            if self._debug_level >= 5:
                print('')
                print(sz, ens.longitude_gps[k])

    def read_winriver(self, nbt):
        self._winrivprob = True
        self.cfg['sourceprog'] = 'WINRIVER'
        if self._source not in [2, 3]:
            if self._debug_level >= 1:
                print('\n  ***** Apparently a WINRIVER file - '
                      'Raw NMEA data handler not yet implemented\n\n')
            self._source = 2
        startpos = self.f.tell()
        sz = self.f.read_ui16(1)
        tmp = self.f.reads(sz)
        self._nbyte = self.f.tell() - startpos + 2

    def skip_Ncol(self, n_skip=1):
        self.f.seek(n_skip * self.cfg['n_cells'], 1)
        self._nbyte = 2 + n_skip * self.cfg['n_cells']

    def skip_Nbyte(self, n_skip):
        self.f.seek(n_skip, 1)
        self._nbyte = self._nbyte = 2 + n_skip

    def read_hdr(self,):
        fd = self.f
        cfgid = list(fd.read_ui8(2))
        nread = 0
        if self._debug_level > 2:
            print(self.f.pos)
            print('  cfgid0: [{:x}, {:x}]'.format(*cfgid))
            # print(cfgid[0] not in [127] or cfgid[1] not in [127])
            # print(not self.checkheader())
        while (cfgid[0] != 127 or cfgid[1] != 127) or not self.checkheader():
            nextbyte = fd.read_ui8(1)
            pos = fd.tell()
            nread += 1
            # print(pos)
            # print(nextbyte)
            cfgid[1] = cfgid[0]
            cfgid[0] = nextbyte
            if not pos % 1000:
                print('  Still looking for valid cfgid at file '
                      'position %d ...' % pos)
        self._pos = self.f.tell() - 2
        # if nread > 0:
        #     print('  Junk found at BOF... skipping %d bytes until\n'
        #           '  cfgid: (%x,%x) at file pos %d.'
        #           % (self._pos, cfgid[0], cfgid[1], nread))
        if self._debug_level > 0:
            print(fd.tell())
        self.read_hdrseg()

    def read_cfg(self,):
        cfgid = self.f.read_ui16(1)
        ### SKIPPED SOME DEBUGGING STUFF HERE ###
        self.read_cfgseg()

    def read_cfgseg(self,):
        cfgstart = self.f.tell()
        cfg = self.cfg
        fd = self.f
        tmp = fd.read_ui8(5)
        prog_ver0 = tmp[0]
        cfg['prog_ver'] = tmp[0] + tmp[1] / 100.
        cfg['name'] = self._cfgnames.get(tmp[0],
                                         'unrecognized firmware version')
        config = tmp[2:4]
        #cfg['config'] = (np.binary_repr(config[1], 8) + '-' +
        #                 np.binary_repr(config[0], 8))
        cfg['beam_angle'] = [15, 20, 30][(config[1] & 3)]
        #cfg['numbeams'] = [4, 5][int((config[1] & 16) == 16)]
        cfg['freq'] = ([75, 150, 300, 600, 1200, 2400, 38][(config[0] & 7)])
        cfg['beam_pattern'] = (['concave',
                                'convex'][int((config[0] & 8) == 8)])
        cfg['orientation'] = ['down', 'up'][int((config[0] & 128) == 128)]
        #cfg['simflag'] = ['real', 'simulated'][tmp[4]]
        fd.seek(1, 1)
        cfg['n_beams'] = fd.read_ui8(1)
        cfg['n_cells'] = fd.read_ui8(1)
        cfg['pings_per_ensemble'] = fd.read_ui16(1)
        cfg['cell_size'] = fd.read_ui16(1) * .01
        cfg['blank'] = fd.read_ui16(1) * .01
        cfg['prof_mode'] = fd.read_ui8(1)
        cfg['corr_threshold'] = fd.read_ui8(1)
        cfg['prof_codereps'] = fd.read_ui8(1)
        cfg['min_pgood'] = fd.read_ui8(1)
        cfg['evel_threshold'] = fd.read_ui16(1)
        cfg['sec_between_ping_groups'] = (
            np.sum(np.array(fd.read_ui8(3)) *
                   np.array([60., 1., .01])))
        coord_sys = fd.read_ui8(1)
        #cfg['coord'] = np.binary_repr(coord_sys, 8)
        cfg['coord_sys'] = (['beam', 'inst',
                             'ship', 'earth'][((coord_sys >> 3) & 3)])
        cfg['use_pitchroll'] = ['no', 'yes'][(coord_sys & 4) == 4]
        cfg['use_3beam'] = ['no', 'yes'][(coord_sys & 2) == 2]
        cfg['bin_mapping'] = ['no', 'yes'][(coord_sys & 1) == 1]
        cfg['xducer_misalign_deg'] = fd.read_i16(1) * .01
        cfg['magnetic_var_deg'] = fd.read_i16(1) * .01
        cfg['sensors_src'] = np.binary_repr(fd.read_ui8(1), 8)
        cfg['sensors_avail'] = np.binary_repr(fd.read_ui8(1), 8)
        cfg['bin1_dist_m'] = fd.read_ui16(1) * .01
        cfg['xmit_pulse'] = fd.read_ui16(1) * .01
        cfg['water_ref_cells'] = fd.read_ui8(2)
        cfg['fls_target_threshold'] = fd.read_ui8(1)
        fd.seek(1, 1)
        cfg['xmit_lag_m'] = fd.read_ui16(1) * .01
        self._nbyte = 40

        #!!! Legacy (no test data for this?)
        # if prog_ver0 in [8, 16]:
        #     if cfg['prog_ver'] >= 8.14:
        #         cfg['serialnum'] = fd.read_ui8(8)
        #         self._nbyte += 8
        #     if cfg['prog_ver'] >= 8.24:
        #         cfg['sysbandwidth'] = fd.read_ui8(2)
        #         self._nbyte += 2
        #     if cfg['prog_ver'] >= 16.05:
        #         cfg['syspower'] = fd.read_ui8(1)
        #         self._nbyte += 1
        #     if cfg['prog_ver'] >= 16.27:
        #         cfg['navigator_basefreqindex'] = fd.read_ui8(1)
        #         cfg['remus_serialnum'] = fd.reaadcpd('uint8', 4)
        #         cfg['h_adcp_beam_angle'] = fd.read_ui8(1)
        #         self._nbyte += 6
        # elif prog_ver0 == 9:
        #     if cfg['prog_ver'] >= 9.10:
        #         cfg['serialnum'] = fd.read_ui8(8)
        #         cfg['sysbandwidth'] = fd.read_ui8(2)
        #         self._nbyte += 10
        # elif prog_ver0 in [14, 23]:
        #     cfg['serialnum'] = fd.read_ui8(8)
        #     self._nbyte += 8
        self.configsize = self.f.tell() - cfgstart

    def read_hdrseg(self,):
        fd = self.f
        hdr = self.hdr
        hdr['nbyte'] = fd.read_i16(1)
        if self._debug_level > 2:
            print(fd.tell())
        fd.seek(1, 1)
        ndat = fd.read_i8(1)
        hdr['dat_offsets'] = fd.read_i16(ndat)
        self._nbyte = 4 + ndat * 2

    def init_data(self,):
        outd = {'data_vars':{},'coords':{},'attrs':{},'units':{},'sys':{}}
        outd['attrs']['inst_make'] = 'TRDI'
        outd['attrs']['inst_model'] = '<WORKHORSE?>'
        outd['attrs']['inst_type'] = 'ADCP'
        outd['attrs']['rotate_vars'] = ['vel', ]
        # Currently RDI doesn't use IMUs
        outd['attrs']['has_imu'] = 0
        for nm in data_defs:
            outd = idata(outd, nm,
                         sz=get_size(nm, self._nens, self.cfg['n_cells']))
        self.outd = outd

    def load_data(self, nens=None):
        if nens is None:
            self._nens = int(self._npings / self.n_avg)
            self._ens_range = (0, self._nens)
        elif (nens.__class__ is tuple or nens.__class__ is list) and \
                len(nens) == 2:
            nens = list(nens)
            if nens[1] == -1:
                nens[1] = self._npings
            self._nens = int((nens[1] - nens[0]) / self.n_avg)
            self._ens_range = nens
            self.f.seek((self.hdr['nbyte'] + 2 + self.extrabytes) *
                        self._ens_range[0], 1)
        else:
            self._nens = nens
            self._ens_range = (0, nens)
        if self._debug_level > 0:
            print('  taking data from pings %d - %d' % tuple(self._ens_range))
            print('  %d ensembles will be produced.' % self._nens)
        self.init_data()
        dat = self.outd
        dat['coords']['range'] = (self.cfg['bin1_dist_m'] +
                                  np.arange(self.cfg['n_cells']) *
                                  self.cfg['cell_size'])
        for nm in self.cfg:
            dat['attrs'][nm] = self.cfg[nm]
        # if self.cfg['orientation'] == 1: # keeping range positive always
        #     dat['coords']['range'] *= -1
        for iens in range(self._nens):
            try:
                self.read_buffer()
            except eofException:
                self.remove_end(iens)
                break
            self.ensemble.clean_data()
            # Fix the 'real-time-clock' century
            clock = self.ensemble.rtc[:, :]
            if clock[0, 0] < 100:
                clock[0, :] += century
            # Copy the ensemble to the dataset.
            for nm in self.vars_read:
                get(dat, nm)[..., iens] = self.avg_func(self.ensemble[nm])
            try:
                dats = datetime.datetime(*clock[:6, 0],
                                         microsecond=clock[6, 0] * 10000).timestamp()
            except ValueError:
                warnings.warn("Invalid time stamp in ping {}.".format(
                    int(self.ensemble.number[0])))
                dat['coords']['time'][iens] = np.NaN
            else:
                dat['coords']['time'][iens] = np.median(dats)
        self.finalize()
        if 'vel_bt' in dat['data_vars']:
            #dat['props']['rotate_vars'].update({'vel_bt', })
            dat['attrs']['rotate_vars'].append('vel_bt')
        return dat

    def finalize(self, ):
        """
        Remove the attributes from the data that were never loaded.

        """
        dat = self.outd
        for nm in set(data_defs.keys()) - self.vars_read:
            pop(dat, nm)
        for nm in self.cfg:
            dat['attrs'][nm] = self.cfg[nm]
        dat['attrs']['fs'] = (dat['attrs']['sec_between_ping_groups'] *
                              dat['attrs']['pings_per_ensemble']) **(-1)
        #dat.props['coord_sys'] = dat.config.coord_sys
        for nm in data_defs:
            shp = data_defs[nm][0]
            if len(shp) and shp[0] == 'nc' and in_group(dat, nm):
                setd(dat, nm, np.swapaxes(get(dat, nm), 0, 1))

    def remove_end(self, iens):
        dat = self.outd
        #if iens < dat.shape[-1]:
        print('  Encountered end of file.  Cleaning up data.')
        for nm in self.vars_read:
            setd(dat, nm, get(dat, nm)[..., :iens])

    def search_buffer(self):
        """
        Check to see if the next bytes indicate the beginning of a
        data block.  If not, search for the next data block, up to
        _search_num times.
        """
        id1 = list(self.f.read_ui8(2))
        search_cnt = 0
        fd = self.f
        if self._debug_level > 3:
            print('  -->In search_buffer...')
        while (search_cnt < self._search_num and
               ((id1[0] != 127 or id1[1] != 127) or
                not self.checkheader())):
            search_cnt += 1
            nextbyte = fd.read_ui8(1)
            id1[1] = id1[0]
            id1[0] = nextbyte
        if search_cnt == self._search_num:
            raise WrongFileType(
                'Searched {} entries... Bad data encountered. -> {}'
                .format(search_cnt, id1))
        elif search_cnt > 0:
            if self._debug_level > 0:
                print('  WARNING: Searched {} bytes to find next '
                      'valid ensemble start [{:x}, {:x}]'.format(search_cnt,
                                                                 *id1))

    def read_buffer(self,):
        fd = self.f
        self.ensemble.k = -1  # so that k+=1 gives 0 on the first loop.
        self.print_progress()
        hdr = self.hdr
        while self.ensemble.k < self.ensemble.n_avg - 1:
            self.search_buffer()
            startpos = fd.tell() - 2
            self.read_hdrseg()
            byte_offset = self._nbyte + 2
            for n in range(len(hdr['dat_offsets'])):
                id = fd.read_ui16(1)
                self._winrivprob = False
                #print( "%0.4X" % id )
                self.print_pos()
                ##### Read the data for header "id"g
                retval = self.read_dat(id)
                if retval == 'FAIL':
                    break
                byte_offset += self._nbyte
                if n < (len(hdr['dat_offsets']) - 1):
                    oset = hdr['dat_offsets'][n + 1] - byte_offset
                    if oset != 0:
                        if self._debug_level > 0:
                            print('  %s: Adjust location by %d\n' % (id, oset))
                        fd.seek(oset, 1)
                    byte_offset = hdr['dat_offsets'][n + 1]
                else:
                    if hdr['nbyte'] - 2 != byte_offset:
                        if not self._winrivprob:
                            if self._debug_level > 0:
                                print('  {:d}: Adjust location by {:d}\n'
                                      .format(id, hdr['nbyte'] - 2 - byte_offset))
                            self.f.seek(hdr['nbyte'] - 2 - byte_offset, 1)
                    byte_offset = hdr['nbyte'] - 2
            readbytes = fd.tell() - startpos
            #### The 2 is for the checksum:
            offset = hdr['nbyte'] + 2 - byte_offset
            self.check_offset(offset, readbytes)
            self.print_pos(byte_offset=byte_offset)

    def check_offset(self, offset, readbytes):
        fd = self.f
        if offset != 4 and self._fixoffset == 0:
            if self._debug_level >= 1:
                print('\n  ********************************************\n')
                if fd.tell() == self._filesize:
                    print(' EOF reached unexpectedly - discarding this last ensemble\n')
                else:
                    print("  Adjust location by {:d} (readbytes={:d},hdr['nbyte']={:d}\n"
                          .format(offset, readbytes, self.hdr['nbyte']))
                    print("""
                    NOTE - If this appears at the beginning of the file, it may be
                           a dolfyn problem. Please report this message, with details here:
                           https://github.com/lkilcher/dolfyn/issues/8

                         - If this appears at the end of the file it means
                           The file is corrupted and only a partial record
                           has been read\n
                    """)
                print('\n  ********************************************\n')
            self._fixoffset = offset - 4
        fd.seek(4 + self._fixoffset, 1)

    def checkheader(self,):
        if self._debug_level > 1:
            print("  ###In checkheader!")
        fd = self.f
        valid = 0
        #print(self.f.pos)
        numbytes = fd.read_i16(1)
        #print('numbytes={}'.format(numbytes))
        #print(self.f.pos)
        if numbytes > 0:
            #print(self.f.pos)
            fd.seek(numbytes - 2, 1)
            #print(self.f.pos)
            cfgid = fd.read_ui8(2)
            #print(self.f.pos)
            #print('cfgid: [{:x}, {:x}]'.format(*cfgid))
            #### sloppy code:
            if len(cfgid) == 2:
                fd.seek(-numbytes - 2, 1)
                #print(self.f.pos)
                if cfgid[0] == 127 and cfgid[1] in [127, 121]:
                    if cfgid[1] == 121 and self._debug7f79 is None:
                        self._debug7f79 = True
                        warnings.warn(
                            "This ADCP file has an undocumented "
                            "sync-code.  If possible, please notify the "
                            "DOLfYN developers that you are recieving "
                            "this warning by posting the hardware and "
                            "software details on how you acquired this file "
                            "to "
                            "http://github.com/lkilcher/dolfyn/issues/7")
                    valid = 1
        else:
            fd.seek(-2, 1)
        #print(self.f.pos)
        if self._debug_level > 1:
            print("  ###Leaving checkheader.")
        return valid
