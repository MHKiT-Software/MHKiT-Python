import numpy as np
import xarray as xr
from struct import unpack, calcsize
import warnings
from . import nortek2_defs as defs
from . import nortek2_lib as lib
from .base import _find_userdata, _create_dataset
from ..rotate.vector import _euler2orient
from ..rotate.base import _set_coords
from ..rotate.api import set_declination


def read_signature(filename, userdata=True, nens=None):
    """Read a Nortek Signature (.ad2cp) datafile

    Parameters
    ----------
    filename : string
        The filename of the file to load.
    userdata : bool
        To search for and use a .userdata.json or not
    nens : int, or tuple of 2 ints
        The number of ensembles to read, if int (starting at the
        beginning); or the range of ensembles to read, if tuple.

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from the binary instrument data
    """
    if nens is None:
        nens = [0, None]
    else:
        try:
            n = len(nens)
        except TypeError:
            nens = [0, nens]
        else:
            # passes: it's a list/tuple/array
            if n != 2:
                raise TypeError('nens must be: None (), int, or len 2')

    userdata = _find_userdata(filename, userdata)

    rdr = _Ad2cpReader(filename)
    d = rdr.readfile(nens[0], nens[1])
    rdr.sci_data(d)
    out = _reorg(d)
    _reduce(out)
    
    declin = None
    for nm in userdata:
        if 'dec' in nm:
            declin = userdata[nm]
        else:
            out['attrs'][nm] = userdata[nm]
    
    # Create xarray dataset from upper level dictionary
    ds = _create_dataset(out)
    ds = _set_coords(ds, ref_frame=ds.coord_sys)

    if 'orientmat' not in ds:
        omat = _euler2orient(ds['heading'], ds['pitch'], ds['roll'])
        ds['orientmat'] = xr.DataArray(omat,
                                       coords={'inst': ['X','Y','Z'],
                                               'earth': ['E','N','U'], 
                                               'time': ds['time']},
                                       dims=['inst','earth','time'])
    if declin is not None:
        ds = set_declination(ds, declin)

    return ds


class _Ad2cpReader():
    def __init__(self, fname, endian=None, bufsize=None, rebuild_index=False):
        self.fname = fname
        self._check_nortek(endian)
        self._index = lib._get_index(fname,
                                    reload=rebuild_index)
        self._reopen(bufsize)
        self.filehead_config = self._read_filehead_config_string()
        self._ens_pos = lib._index2ens_pos(self._index)
        self._config = lib._calc_config(self._index)
        self._init_burst_readers()
        self.unknown_ID_count = {}
    
    def _check_nortek(self, endian):
        self._reopen(10)
        byts = self.f.read(2)
        if endian is None:
            if unpack('<' + 'BB', byts) == (165, 10):
                endian = '<'
            elif unpack('>' + 'BB', byts) == (165, 10):
                endian = '>'
            else:
                raise Exception(
                    "I/O error: could not determine the 'endianness' "
                    "of the file.  Are you sure this is a Nortek "
                    "AD2CP file?")
        self.endian = endian

    def _reopen(self, bufsize=None):
        if bufsize is None:
            bufsize = 1000000
        try:
            self.f.close()
        except AttributeError:
            pass
        self.f = open(self.fname, 'rb', bufsize)

    def _read_filehead_config_string(self, ):
        hdr = self._read_hdr()
        out = {}
        s_id, string = self._read_str(hdr['sz'])
        string = string.decode('utf-8')
        for ln in string.splitlines():
            ky, val = ln.split(',', 1)
            if ky in out:
                # There are more than one of this key
                if not isinstance(out[ky], list):
                    tmp = out[ky]
                    out[ky] = []
                    out[ky].append(tmp)
                out[ky].append(val)
            else:
                out[ky] = val
        out2 = {}
        for ky in out:
            if ky.startswith('GET'):
                dat = out[ky]
                d = out2[ky.lstrip('GET')] = dict()
                for itm in dat.split(','):
                    k, val = itm.split('=')
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    d[k] = val
            else:
                out2[ky] = out[ky]
        return out2

    def _init_burst_readers(self, ):
        self._burst_readers = {}
        for rdr_id, cfg in self._config.items():
            if rdr_id == 28:
                self._burst_readers[rdr_id] = defs._calc_echo_struct(
                    cfg['_config'], cfg['ncells'])
            elif rdr_id == 23:
                self._burst_readers[rdr_id] = defs._calc_bt_struct(
                    cfg['_config'], cfg['nbeams'])
            else:
                self._burst_readers[rdr_id] = defs._calc_burst_struct(
                    cfg['_config'], cfg['nbeams'], cfg['ncells'])
                
    def init_data(self, ens_start, ens_stop):
        outdat = {}
        nens = int(ens_stop - ens_start)
        n26 = ((self._index['ID'] == 26) &
               (self._index['ens'] >= ens_start) &
               (self._index['ens'] < ens_stop)).sum()
        for ky in self._burst_readers:
            if ky == 26:
                n = n26
                ens = np.zeros(n, dtype='uint32')
            else:
                ens = np.arange(ens_start,
                                ens_stop).astype('uint32')
                n = nens
            outdat[ky] = self._burst_readers[ky].init_data(n)
            outdat[ky]['ensemble'] = ens
            outdat[ky]['units'] = self._burst_readers[ky].data_units()
        return outdat

    def _read_hdr(self, do_cs=False):
        res = defs.header.read2dict(self.f, cs=do_cs)
        if res['sync'] != 165:
            raise Exception("Out of sync!")
        return res
    
    def _read_str(self, size):
        string = self.f.read(size)
        id = string[0]
        #end = string[-1]
        string = string[1:-1]
        return id, string
    
    def _read_burst(self, id, dat, c, echo=False):
        rdr = self._burst_readers[id]
        rdr.read_into(self.f, dat, c)


    def readfile(self, ens_start=0, ens_stop=None):
        nens_total = len(self._ens_pos)
        if ens_stop is None or ens_stop > nens_total:
            ens_stop = nens_total - 1
        ens_start = int(ens_start)
        ens_stop = int(ens_stop)
        nens = ens_stop - ens_start
        outdat = self.init_data(ens_start, ens_stop)
        outdat['filehead_config'] = self.filehead_config
        print('Reading file %s ...' % self.fname)
        retval = None
        c = 0
        c26 = 0
        self.f.seek(self._ens_pos[ens_start], 0)
        while not retval:
            try:
                hdr = self._read_hdr()
            except IOError:
                return outdat
            id = hdr['id']
            if id in [21, 23, 24, 28]: # vel, bt, vel_b5, echo
                self._read_burst(id, outdat[id], c)
            elif id in [26]:  # alt_raw (altimeter burst)
                rdr = self._burst_readers[26]
                if not hasattr(rdr, '_nsamp_index'):
                    first_pass = True
                    tmp_idx = rdr._nsamp_index = rdr._names.index('altraw_nsamp')  # noqa
                    shift = rdr._nsamp_shift = calcsize(
                        defs._format(rdr._format[:tmp_idx],
                                     rdr._N[:tmp_idx]))
                else:
                    first_pass = False
                    tmp_idx = rdr._nsamp_index
                    shift = rdr._nsamp_shift
                tmp_idx = tmp_idx + 2  # Don't add in-place
                self.f.seek(shift, 1)
                # Now read the num_samples
                sz = unpack('<I', self.f.read(4))[0]
                self.f.seek(-shift - 4, 1)
                if first_pass:
                    # Fix the reader
                    rdr._shape[tmp_idx].append(sz)
                    rdr._N[tmp_idx] = sz
                    rdr._struct = defs.Struct('<' + rdr.format)
                    rdr.nbyte = calcsize(rdr.format)
                    rdr._cs_struct = defs.Struct('<' + '{}H'.format(int(rdr.nbyte // 2)))
                    # Initialize the array
                    outdat[26]['altraw_samp'] = defs._nans(
                        [rdr._N[tmp_idx],
                         len(outdat[26]['altraw_samp'])],
                        dtype=np.uint16)
                else:
                    if sz != rdr._N[tmp_idx]:
                        raise Exception(
                            "The number of samples in this 'Altimeter Raw' "
                            "burst is different from prior bursts.")
                self._read_burst(id, outdat[id], c26)
                outdat[id]['ensemble'][c26] = c
                c26 += 1

            elif id in [22, 27, 29, 30, 31, 35, 36]: # avg record, bt record, 
            # DVL, alt record, avg alt_raw record, raw echo, raw echo transmit
                warnings.warn(
                    "Unhandled ID: 0x{:02X} ({:02d})\n"
                    "    This ID is not yet handled by DOLfYN.\n".format(id, id))
                self.f.seek(hdr['sz'], 1)
            elif id == 160:
                # 0xa0 (i.e., 160) is a 'string data record'
                if id not in outdat:
                    outdat[id] = dict()
                s_id, s = self._read_str(hdr['sz'], )
                outdat[id][(c, s_id)] = s
            else:
                if id not in self.unknown_ID_count:
                    self.unknown_ID_count[id] = 1
                    print('Unknown ID: 0x{:02X}!'.format(id))
                else:
                    self.unknown_ID_count[id] += 1
                self.f.seek(hdr['sz'], 1)
            # Count checking
            if c + ens_start + 1 >= nens_total:
                # Make sure we're not at the end of the count list.
                continue
            while (self.f.tell() >= self._ens_pos[c + ens_start + 1]):
                c += 1
                if c + ens_start + 1 >= nens_total:
                    # Again check end of count list
                    break
            if c >= nens:
                return outdat

    def sci_data(self, dat):
        for id in dat:
            dnow = dat[id]
            if id not in self._burst_readers:
                continue
            rdr = self._burst_readers[id]
            rdr.sci_data(dnow)
            if 'vel' in dnow and 'vel_scale' in dnow:
                dnow['vel'] = (dnow['vel'] *
                               10.0 ** dnow['vel_scale']).astype('float32')

    def __exit__(self, type, value, trace,):
        self.f.close()

    def __enter__(self,):
        return self


def _reorg(dat):
    """This function grabs the data from the dictionary of data types
    (organized by ID), and combines them into the
    :class:`dolfyn.ADPdata` object.
    """
    outdat = {'data_vars':{},'coords':{},'attrs':{},
              'units':{},'sys':{},'altraw':{}}
    cfg = outdat['attrs']
    cfh = cfg['filehead_config'] = dat['filehead_config']
    cfg['inst_model'] = (cfh['ID'].split(',')[0][5:-1])
    cfg['inst_make'] = 'Nortek'
    cfg['inst_type'] = 'ADCP'
    cfg['rotate_vars'] = ['vel',]

    for id, tag in [(21, ''), (23, '_bt'), (24, '_b5'), (26, '_ar'), (28, '_echo')]:
        if id in [24, 26]:
            collapse_exclude = [0]
        else:
            collapse_exclude = []
        if id not in dat:
            continue
        dnow = dat[id]
        outdat['units'].update(dnow['units'])
        cfg['burst_config' + tag] = lib._headconfig_int2dict(
            lib._collapse(dnow['config'], exclude=collapse_exclude,
                          name='config'))
        outdat['coords']['time' + tag] = lib._calc_time(
            dnow['year'] + 1900,
            dnow['month'],
            dnow['day'],
            dnow['hour'],
            dnow['minute'],
            dnow['second'],
            dnow['usec100'].astype('uint32') * 100)
        tmp = lib._beams_cy_int2dict(
            lib._collapse(dnow['beam_config'], exclude=collapse_exclude,
                          name='beam_config'), 21)
        cfg['ncells' + tag] = tmp['ncells']
        cfg['coord_sys_axes' + tag] = tmp['cy']
        cfg['nbeams' + tag] = tmp['nbeams']
        cfg['xmit_energy' + tag] = np.median(dnow['xmit_energy'])
        cfg['ambig_vel' + tag] = np.median(dnow['ambig_vel'])
        
        for ky in ['SerialNum', 'cell_size', 'blank_dist']:
            # These ones should 'collapse'
            # (i.e., all values should be the same)
            # So we only need that one value.
            cfg[ky + tag] = lib._collapse(dnow[ky], 
                                          exclude=collapse_exclude,
                                          name=ky)
        for ky in ['nom_corr', 'data_desc',
                   'vel_scale', 'power_level']:
            # These ones should 'collapse'
            # (i.e., all values should be the same)
            # So we only need that one value.
            cfg['burst_config' + tag][ky + tag] = lib._collapse(dnow[ky], 
                                         exclude=collapse_exclude,
                                         name=ky)
            
        for ky in ['c_sound', 'temp', 'pressure',
                   'heading', 'pitch', 'roll',
                   'mag', 'accel',
                   ]:
            # No if statement here
            outdat['data_vars'][ky + tag] = dnow[ky]
            
        for ky in ['batt_V', 'temp_mag', 'temp_clock',
                   'error', 'status',
                   '_ensemble', 'ensemble',
                   ]:
            outdat['sys'][ky + tag] = dnow[ky]
            
        for ky in ['vel', 'amp', 'corr', 'prcnt_gd',
                   'echo', 'dist', 
                   'orientmat', 'angrt', 'quaternion',
                   ]:
            if ky in dnow:
                outdat['data_vars'][ky + tag] = dnow[ky]
        
        for ky in ['alt_dist', 'alt_quality', 'alt_status',
                   'ast_dist', 'ast_quality', 'ast_offset_time',
                   'ast_pressure', 
                   'altraw_nsamp', 'altraw_dsamp', 'altraw_samp',
                   ]:
            if ky in dnow:
                outdat['data_vars'][ky + tag] = dnow[ky]
                
        for ky in ['status0', 'fom',
                   'temp_press', 'std_press'
                   'std_pitch', 'std_roll', 'std_heading',
                   ]:
            if ky in dnow:
                outdat['sys'][ky + tag] = dnow[ky]  

    # Move 'altimeter raw' data to it's own down-sampled structure
    if 26 in dat:
        ard = outdat['altraw']
        for ky in list(outdat['data_vars']):
            if ky.endswith('_ar'):
                grp = ky.split('.')[0]
                if '.' in ky and grp not in ard:
                    ard[grp] = {}
                ard[ky.rstrip('_ar')] = outdat['data_vars'].pop(ky)
        for ky in list(outdat['sys']):
            if ky.endswith('_ar'):
                grp = ky.split('.')[0]
                if '.' in ky and grp not in ard:
                    ard[grp] = {}            
                ard[ky.rstrip('_ar')] = outdat['sys'].pop(ky)
        N = ard['_map_N'] = len(outdat['coords']['time'])
        parent_map = np.arange(N)
        ard['_map'] = parent_map[np.in1d(outdat['sys']['ensemble'], ard['ensemble'])]
    
    outdat['attrs']['coord_sys'] = {'XYZ': 'inst',
                                    'ENU': 'earth',
                                    'beam': 'beam'}[cfg['coord_sys_axes']]
    tmp = lib._status2data(outdat['sys']['status'])  # returns a dict
    
    # Instrument direction
    # 0: XUP, 1: XDOWN, 2: YUP, 3: YDOWN, 4: ZUP, 5: ZDOWN, 
    # 7: AHRS, handle as ZUP
    nortek_orient = {0:'horizontal', 1:'horizontal', 2:'horizontal',
                     3:'horizontal', 4:'up', 5:'down', 7:'AHRS'}
    outdat['attrs']['orientation'] = nortek_orient[tmp['orient_up'][0]]
    orient_status = {0:'fixed', 1:'auto_UD', 3: 'AHRS-3D'}
    outdat['attrs']['orient_status'] = orient_status[tmp['auto orientation'][0]]
    
    for ky in ['accel', 'angrt', 'mag']:
        for dky in outdat['data_vars'].keys():
            if dky == ky or dky.startswith(ky + '_'):
                outdat['attrs']['rotate_vars'].append(dky)
    if 'vel_bt' in outdat['data_vars']:
        outdat['attrs']['rotate_vars'].append('vel_bt')
        
    return outdat


def _reduce(data):
    """This function takes the output from `reorg`, and further simplifies the 
    data. Mostly this is combining system, environmental, and orientation data 
    --- from different data structures within the same ensemble --- by
    averaging.  
    """
    # Average these fields
    for ky in ['c_sound', 'temp', 'pressure',
               'temp_press', 'temp_clock', 'temp_mag',
               'batt_V']:
        grp = defs._get_group(ky)
        if grp is None:
            dnow = data
        else:
            dnow = data[grp]
        lib._reduce_by_average(dnow, ky, ky + '_b5')

    # Angle-averaging is treated separately
    for ky in ['heading', 'pitch', 'roll']:
        lib._reduce_by_average_angle(data['data_vars'], ky, ky + '_b5')

    dv = data['data_vars']
    da = data['attrs']
    data['coords']['range'] = ((np.arange(dv['vel'].shape[1])+1) *
                               da['cell_size'] +
                               da['blank_dist'])
    if 'vel_b5' in dv:
        data['coords']['range_b5'] = ((np.arange(dv['vel_b5'].shape[1])+1) *
                                      da['cell_size_b5'] +
                                      da['blank_dist_b5'])
    if 'echo_echo' in dv:
        dv['echo'] = dv.pop('echo_echo')
        data['coords']['range_echo'] = ((np.arange(dv['echo'].shape[0])+1) *
                                        da['cell_size_echo'] +
                                        da['blank_dist_echo'])

    if 'orientmat' in data['data_vars']:
        da['has_imu'] = 1 # logical
    else:
        da['has_imu'] = 0
    da['fs'] = da['filehead_config']['BURST'].pop('SR')
    tmat = da['filehead_config'].pop('XFBURST')
    tm = np.zeros((tmat['ROWS'], tmat['COLS']), dtype=np.float32)
    for irow in range(tmat['ROWS']):
        for icol in range(tmat['COLS']):
            tm[irow, icol] = tmat['M' + str(irow + 1) + str(icol + 1)]
    dv['beam2inst_orientmat'] = tm
