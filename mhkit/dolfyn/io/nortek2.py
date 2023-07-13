import numpy as np
from struct import unpack, calcsize
import warnings
from pathlib import Path
import logging
import json

from . import nortek2_defs as defs
from . import nortek2_lib as lib
from .base import _find_userdata, _create_dataset, _abspath
from ..rotate.vector import _euler2orient
from ..rotate.base import _set_coords
from ..rotate.api import set_declination
from ..time import epoch2dt64, _fill_time_gaps


def read_signature(filename, userdata=True, nens=None, rebuild_index=False,
                   debug=False, **kwargs):
    """
    Read a Nortek Signature (.ad2cp) datafile

    Parameters
    ----------
    filename : string
      The filename of the file to load.
    userdata : bool
      To search for and use a .userdata.json or not
    nens : None, int or 2-element tuple (start, stop)
      Number of pings or ensembles to read from the file. 
      Default is None, read entire file
    rebuild_index : bool
      Force rebuild of dolfyn-written datafile index. Useful for code updates.
      Default = False
    debug : bool
      Logs debugger ouput if true. Default = False

    Returns
    -------
    ds : xarray.Dataset
      An xarray dataset from the binary instrument data
    """

    # Start debugger logging
    if debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        filepath = Path(filename)
        logfile = filepath.with_suffix('.dolfyn.log')
        logging.basicConfig(filename=str(logfile),
                            filemode='w',
                            level=logging.NOTSET,
                            format='%(name)s - %(levelname)s - %(message)s')

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

    rdr = _Ad2cpReader(filename, rebuild_index=rebuild_index, debug=debug)
    d = rdr.readfile(nens[0], nens[1])
    rdr.sci_data(d)
    out = _reorg(d)
    _reduce(out)

    # Convert time to dt64 and fill gaps
    coords = out['coords']
    t_list = [t for t in coords if 'time' in t]
    for ky in t_list:
        tdat = coords[ky]
        tdat[tdat == 0] = np.NaN
        if np.isnan(tdat).any():
            tag = ky.lstrip('time')
            warnings.warn("Zero/NaN values found in '{}'. Interpolating and "
                          "extrapolating them. To identify which values were filled later, "
                          "look for 0 values in 'status{}'".format(ky, tag))
            tdat = _fill_time_gaps(tdat, sample_rate_hz=out['attrs']['fs'])
        coords[ky] = epoch2dt64(tdat).astype('datetime64[ns]')

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
        ds['orientmat'] = _euler2orient(
            ds['time'], ds['heading'], ds['pitch'], ds['roll'])

    if declin is not None:
        set_declination(ds, declin, inplace=True)

    # Convert config dictionary to json string
    for key in list(ds.attrs.keys()):
        if 'config' in key:
            ds.attrs[key] = json.dumps(ds.attrs[key])

    # Close handler
    if debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

    return ds


class _Ad2cpReader():
    def __init__(self, fname, endian=None, bufsize=None, rebuild_index=False,
                 debug=False):
        self.fname = fname
        self.debug = debug
        self._check_nortek(endian)
        self.f.seek(0, 2)  # Seek to end
        self._eof = self.f.tell()
        self._index = lib.get_index(fname,
                                    reload=rebuild_index,
                                    debug=debug)
        self._reopen(bufsize)
        self.filehead_config = self._read_filehead_config_string()
        self._ens_pos = self._index['pos'][lib._boolarray_firstensemble_ping(
            self._index)]
        self._lastblock_iswhole = self._calc_lastblock_iswhole()
        self._config = lib._calc_config(self._index)
        self._init_burst_readers()
        self.unknown_ID_count = {}

    def _calc_lastblock_iswhole(self, ):
        blocksize, blocksize_count = np.unique(np.diff(self._ens_pos),
                                               return_counts=True)
        standard_blocksize = blocksize[blocksize_count.argmax()]
        return (self._eof - self._ens_pos[-1]) == standard_blocksize

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
        self.f = open(_abspath(self.fname), 'rb', bufsize)

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
                    cfg['_config'], cfg['n_cells'])
            elif rdr_id == 23:
                self._burst_readers[rdr_id] = defs._calc_bt_struct(
                    cfg['_config'], cfg['n_beams'])
            else:
                self._burst_readers[rdr_id] = defs._calc_burst_struct(
                    cfg['_config'], cfg['n_beams'], cfg['n_cells'])

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
            outdat[ky]['long_name'] = self._burst_readers[ky].data_longnames()
            outdat[ky]['standard_name'] = self._burst_readers[ky].data_stdnames()
        return outdat

    def _read_hdr(self, do_cs=False):
        res = defs.header.read2dict(self.f, cs=do_cs)
        if res['sync'] != 165:
            raise Exception("Out of sync!")
        return res

    def _read_str(self, size):
        string = self.f.read(size)
        id = string[0]
        string = string[1:-1]
        return id, string

    def _read_burst(self, id, dat, c, echo=False):
        rdr = self._burst_readers[id]
        rdr.read_into(self.f, dat, c)

    def readfile(self, ens_start=0, ens_stop=None):
        # If the lastblock is not whole, we don't read it.
        # If it is, we do (don't subtract 1)
        nens_total = len(self._ens_pos) - int(not self._lastblock_iswhole)
        if ens_stop is None or ens_stop > nens_total:
            ens_stop = nens_total
        ens_start = int(ens_start)
        ens_stop = int(ens_stop)
        nens = ens_stop - ens_start
        outdat = self.init_data(ens_start, ens_stop)
        outdat['filehead_config'] = self.filehead_config
        print('Reading file %s ...' % self.fname)
        c = 0
        c26 = 0
        self.f.seek(self._ens_pos[ens_start], 0)
        while True:
            try:
                hdr = self._read_hdr()
            except IOError:
                return outdat
            id = hdr['id']
            if id in [21, 22, 23, 24, 28]:  # vel, bt, vel_b5, echo
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
                    rdr._cs_struct = defs.Struct(
                        '<' + '{}H'.format(int(rdr.nbyte // 2)))
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

            elif id in [27, 29, 30, 31, 35, 36]:  # bt record, DVL,
                # alt record, avg alt_raw record, raw echo, raw echo transmit
                if self.debug:
                    logging.debug(
                        "Skipped ID: 0x{:02X} ({:02d})\n".format(id, id))
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
                    if self.debug:
                        logging.warning('Unknown ID: 0x{:02X}!'.format(id))
                else:
                    self.unknown_ID_count[id] += 1
                self.f.seek(hdr['sz'], 1)

            c = self._advance_ens_count(c, ens_start, nens_total)

            if c >= nens:
                return outdat

    def _advance_ens_count(self, c, ens_start, nens_total):
        """This method advances the counter when appropriate to do so.
        """
        # It's unfortunate that all of this count checking is so
        # complex, but this is the best I could come up with right
        # now.
        try:
            # Checks to makes sure we're not already at the end of the
            # self._ens_pos array
            _posnow = self._ens_pos[c + ens_start + 1]
        except IndexError:
            # We are at the end of the array, set _posnow
            # We use "+1" here because we want the >= in the while
            # loop to fail for this case so that we go ahead and read
            # the next ping without advancing the ens counter.
            _posnow = self._eof + 1
        while (self.f.tell() >= _posnow):
            c += 1
            if c + ens_start + 1 >= nens_total:
                # Again check end of count list
                break
            try:
                # Same check as above.
                _posnow = self._ens_pos[c + ens_start + 1]
            except IndexError:
                _posnow = self._eof + 1
        return c

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
    """
    This function grabs the data from the dictionary of data types
    (organized by ID), and combines them into a single dictionary.
    """

    outdat = {'data_vars': {}, 'coords': {}, 'attrs': {},
              'units': {}, 'long_name': {}, 'standard_name': {},
              'sys': {}, 'altraw': {}}
    cfg = outdat['attrs']
    cfh = cfg['filehead_config'] = dat['filehead_config']
    cfg['inst_model'] = (cfh['ID'].split(',')[0][5:-1])
    cfg['inst_make'] = 'Nortek'
    cfg['inst_type'] = 'ADCP'

    for id, tag in [(21, ''), (22, '_avg'), (23, '_bt'), 
                    (24, '_b5'), (26, '_ast'), (28, '_echo')]:
        if id in [24, 26]:
            collapse_exclude = [0]
        else:
            collapse_exclude = []
        if id not in dat:
            continue
        dnow = dat[id]
        outdat['units'].update(dnow['units'])
        outdat['long_name'].update(dnow['long_name'])
        for ky in dnow['units']:
            if not dnow['standard_name'][ky]:
                dnow['standard_name'].pop(ky)
        outdat['standard_name'].update(dnow['standard_name'])
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
        cfg['n_cells' + tag] = tmp['n_cells']
        cfg['coord_sys_axes' + tag] = tmp['cy']
        cfg['n_beams' + tag] = tmp['n_beams']
        cfg['ambig_vel' +
            tag] = lib._collapse(dnow['ambig_vel'], name='ambig_vel')

        for ky in ['SerialNum', 'cell_size', 'blank_dist', 'nominal_corr',
                   'power_level_dB']:
            cfg[ky + tag] = lib._collapse(dnow[ky],
                                          exclude=collapse_exclude,
                                          name=ky)

        for ky in ['c_sound', 'temp', 'pressure', 'heading', 'pitch', 'roll',
                   'mag', 'accel', 'batt', 'temp_clock', 'error',
                   'status', 'ensemble',
                   ]:
            outdat['data_vars'][ky + tag] = dnow[ky]
            if 'ensemble' in ky:
                outdat['data_vars'][ky + tag] += 1
                outdat['units'][ky + tag] = '#'
                outdat['long_name'][ky + tag] = 'Ensemble Number'
                outdat['standard_name'][ky + tag] = 'number_of_observations'

        for ky in ['vel', 'amp', 'corr', 'prcnt_gd', 'echo', 'dist',
                   'orientmat', 'angrt', 'quaternions', 'ast_pressure',
                   'alt_dist', 'alt_quality', 'alt_status',
                   'ast_dist', 'ast_quality', 'ast_offset_time',
                   'altraw_nsamp', 'altraw_dsamp', 'altraw_samp',
                   'status0', 'fom', 'temp_press', 'press_std',
                   'pitch_std', 'roll_std', 'heading_std', 'xmit_energy',
                   ]:
            if ky in dnow:
                outdat['data_vars'][ky + tag] = dnow[ky]

    # Move 'altimeter raw' data to its own down-sampled structure
    if 26 in dat:
        ard = outdat['altraw']
        for ky in list(outdat['data_vars']):
            if ky.endswith('_ast'):
                grp = ky.split('.')[0]
                if '.' in ky and grp not in ard:
                    ard[grp] = {}
                ard[ky.rstrip('_ast')] = outdat['data_vars'].pop(ky)

        # Read altimeter status
        alt_status = lib._alt_status2data(outdat['data_vars']['alt_status'])
        for ky in alt_status:
            outdat['attrs'][ky] = lib._collapse(
                alt_status[ky].astype('uint8'), name=ky)
        outdat['data_vars'].pop('alt_status')

        # Power level index
        power = {0: 'high', 1: 'med-high', 2: 'med-low', 3: 'low'}
        outdat['attrs']['power_level_alt'] = power[outdat['attrs'].pop(
            'power_level_idx_alt')]

    # Read status data
    status0_vars = [x for x in outdat['data_vars'] if 'status0' in x]
    # Status data is the same across all tags, and there is always a 'status' and 'status0'
    status0_key = status0_vars[0]
    status0_data = lib._status02data(outdat['data_vars'][status0_key])
    status_key = status0_key.replace('0', '')
    status_data = lib._status2data(outdat['data_vars'][status_key])

    # Individual status codes
    # Wake up state
    wake = {0: 'bad power', 1: 'power on', 2: 'break', 3: 'clock'}
    outdat['attrs']['wakeup_state'] = wake[lib._collapse(
        status_data.pop('wakeup_state'), name=ky)]

    # Instrument direction
    # 0: XUP, 1: XDOWN, 2: YUP, 3: YDOWN, 4: ZUP, 5: ZDOWN,
    # 7: AHRS, handle as ZUP
    nortek_orient = {0: 'horizontal', 1: 'horizontal', 2: 'horizontal',
                     3: 'horizontal', 4: 'up', 5: 'down', 7: 'AHRS'}
    outdat['attrs']['orientation'] = nortek_orient[lib._collapse(
        status_data.pop('orient_up'), name='orientation')]

    # Orientation detection
    orient_status = {0: 'fixed', 1: 'auto_UD', 3: 'AHRS-3D'}
    outdat['attrs']['orient_status'] = orient_status[lib._collapse(
        status_data.pop('auto_orientation'), name='orient_status')]

    # Status variables
    for ky in ['low_volt_skip', 'active_config', 'telemetry_data', 'boost_running']:
        outdat['data_vars'][ky] = status_data[ky].astype('uint8')

    # Processor idle state - need to save as 1/0 per netcdf attribute limitations
    for ky in status0_data:
        outdat['attrs'][ky] = lib._collapse(
            status0_data[ky].astype('uint8'), name=ky)

    # Remove status0 variables - keep status variables as they useful for finding missing pings
    [outdat['data_vars'].pop(var) for var in status0_vars]

    # Set coordinate system
    if 21 not in dat:
        cfg['rotate_vars'] = []
        cy = cfg['coord_sys_axes_avg']
    else:
        cfg['rotate_vars'] = ['vel', ]
        cy = cfg['coord_sys_axes']
    outdat['attrs']['coord_sys'] = {'XYZ': 'inst',
                                    'ENU': 'earth',
                                    'beam': 'beam'}[cy]

    # Copy appropriate vars to rotate_vars
    for ky in ['accel', 'angrt', 'mag']:
        for dky in outdat['data_vars'].keys():
            if dky == ky or dky.startswith(ky + '_'):
                outdat['attrs']['rotate_vars'].append(dky)
    if 'vel_bt' in outdat['data_vars']:
        outdat['attrs']['rotate_vars'].append('vel_bt')
    if 'vel_avg' in outdat['data_vars']:
        outdat['attrs']['rotate_vars'].append('vel_avg')

    return outdat


def _reduce(data):
    """This function takes the output from `reorg`, and further simplifies the
    data. Mostly this is combining system, environmental, and orientation data
    --- from different data structures within the same ensemble --- by
    averaging.
    """
    
    dv = data['data_vars']
    dc = data['coords']
    da = data['attrs']

    # Average these fields
    for ky in ['c_sound', 'temp', 'pressure',
               'temp_press', 'temp_clock', 'batt']:
        lib._reduce_by_average(dv, ky, ky + '_b5')

    # Angle-averaging is treated separately
    for ky in ['heading', 'pitch', 'roll']:
        lib._reduce_by_average_angle(dv, ky, ky + '_b5')

    if 'vel' in dv:
        dc['range'] = ((np.arange(dv['vel'].shape[1])+1) *
                    da['cell_size'] +
                    da['blank_dist'])
        da['fs'] = da['filehead_config']['BURST']['SR']
        tmat = da['filehead_config']['XFBURST']
    if 'vel_avg' in dv:
        dc['range_avg'] = ((np.arange(dv['vel_avg'].shape[1])+1) *
                    da['cell_size_avg'] +
                    da['blank_dist_avg'])
        dv['orientmat'] = dv.pop('orientmat_avg')
        tmat = da['filehead_config']['XFAVG']
        da['fs'] = da['filehead_config']['PLAN']['MIAVG']
        da['avg_interval_sec'] = da['filehead_config']['AVG']['AI']
        da['bandwidth'] = da['filehead_config']['AVG']['BW']
    if 'vel_b5' in dv:
        dc['range_b5'] = ((np.arange(dv['vel_b5'].shape[1])+1) *
                          da['cell_size_b5'] +
                          da['blank_dist_b5'])
    if 'echo_echo' in dv:
        dv['echo'] = dv.pop('echo_echo')
        dc['range_echo'] = ((np.arange(dv['echo'].shape[0])+1) *
                            da['cell_size_echo'] +
                            da['blank_dist_echo'])

    if 'orientmat' in data['data_vars']:
        da['has_imu'] = 1  # logical
        # Signature AHRS rotation matrix returned in "inst->earth"
        # Change to dolfyn's "earth->inst"
        dv['orientmat'] = np.rollaxis(dv['orientmat'], 1)
    else:
        da['has_imu'] = 0

    theta = da['filehead_config']['BEAMCFGLIST'][0]
    if 'THETA=' in theta:
        da['beam_angle'] = int(theta[13:15])
    
    tm = np.zeros((tmat['ROWS'], tmat['COLS']), dtype=np.float32)
    for irow in range(tmat['ROWS']):
        for icol in range(tmat['COLS']):
            tm[irow, icol] = tmat['M' + str(irow + 1) + str(icol + 1)]
    dv['beam2inst_orientmat'] = tm

    # If burst velocity isn't used, need to copy one for 'time'
    if 'time' not in dc:
        for val in dc:
            if 'time' in val:
                time = val
        dc['time'] = dc[time]
