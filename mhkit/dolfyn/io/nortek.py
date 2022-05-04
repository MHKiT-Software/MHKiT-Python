import numpy as np
import xarray as xr
from struct import unpack
import warnings
from . import nortek_defs
from .base import _find_userdata, _create_dataset, _handle_nan, _abspath
from .. import time
from datetime import datetime
from ..tools import misc as tbx
from ..rotate.vector import _calc_omat
from ..rotate.base import _set_coords
from ..rotate import api as rot


def read_nortek(filename, userdata=True, debug=False, do_checksum=False,
                nens=None):
    """Read a classic Nortek (AWAC and Vector) datafile

    Parameters
    ----------
    filename : string
        Filename of Nortek file to read.
    userdata : True, False, or string of userdata.json filename
        (default ``True``) Whether to read the '<base-filename>.userdata.json' 
        file.
    do_checksum : bool (default False)
        Whether to perform the checksum of each data block.
    nens : None (default: read entire file), int, or 2-element tuple (start, stop)
        Number of pings to read from the file

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from the binary instrument data

    """
    userdata = _find_userdata(filename, userdata)

    with _NortekReader(filename, debug=debug, do_checksum=do_checksum,
                       nens=nens) as rdr:
        rdr.readfile()
    rdr.dat2sci()
    dat = rdr.data

    rotmat = None
    declin = None
    for nm in userdata:
        if 'rotmat' in nm:
            rotmat = userdata[nm]
        elif 'dec' in nm:
            declin = userdata[nm]
        else:
            dat['attrs'][nm] = userdata[nm]

    # NaN in time and orientation data
    dat = _handle_nan(dat)

    # Create xarray dataset from upper level dictionary
    ds = _create_dataset(dat)
    ds = _set_coords(ds, ref_frame=ds.coord_sys)

    if 'orientmat' not in ds:
        omat = _calc_omat(ds['heading'].values,
                          ds['pitch'].values,
                          ds['roll'].values,
                          ds.get('orientation_down', None))
        ds['orientmat'] = xr.DataArray(omat,
                                       coords={'earth': ['E', 'N', 'U'],
                                               'inst': ['X', 'Y', 'Z'],
                                               'time': ds['time']},
                                       dims=['earth', 'inst', 'time'])
    if rotmat is not None:
        rot.set_inst2head_rotmat(ds, rotmat, inplace=True)
    if declin is not None:
        rot.set_declination(ds, declin, inplace=True)

    ds['time'] = time.epoch2dt64(ds['time']).astype('datetime64[us]')

    return ds


def _bcd2char(cBCD):
    """
    Taken from the Nortek System Integrator
    Manual "Example Program" Chapter.
    """
    cBCD = min(cBCD, 153)
    c = (cBCD & 15)
    c += 10 * (cBCD >> 4)
    return c


def _bitshift8(val):
    return val >> 8


def _int2binarray(val, n):
    out = np.zeros(n, dtype='bool')
    for idx, n in enumerate(range(n)):
        out[idx] = val & (2 ** n)
    return out


class _NortekReader():
    """A class for reading reading nortek binary files.
    This reader currently only supports AWAC and Vector data formats.

    Parameters
    ----------
    fname : string
        Nortek file filename to read.
    endian : {'<','>'} (optional)
        Specifies if the file is in 'little' or 'big' endian format. By 
        default the reader will attempt to determine this.
    debug : {True, False*} (optional)
        Print debug/progress information?
    do_checksum : {True*, False} (optional)
        Specifies whether to perform the checksum.
    bufsize : int (default 100000)
        The size of the read buffer to use.
    nens : None (default: None, read all files), int, or 2-element tuple (start, stop).
        The number of pings to read from the file. By default, the entire file 
        is read.

    """
    _lastread = [None, None, None, None, None]
    fun_map = {'0x00': 'read_user_cfg',
               '0x04': 'read_head_cfg',
               '0x05': 'read_hw_cfg',
               '0x07': 'read_vec_checkdata',
               '0x10': 'read_vec_data',
               '0x11': 'read_vec_sysdata',
               '0x12': 'read_vec_hdr',
               '0x71': 'read_microstrain',
               '0x20': 'read_awac_profile',
               }

    def __init__(self, fname, endian=None, debug=False,
                 do_checksum=True, bufsize=100000, nens=None):
        self.fname = fname
        self._bufsize = bufsize
        self.f = open(_abspath(fname), 'rb', 1000)
        self.do_checksum = do_checksum
        self.filesize  # initialize the filesize.
        self.debug = debug
        self.c = 0
        self._dtypes = []
        self._n_start = 0
        try:
            len(nens)
        except TypeError:
            # not a tuple, so we assume None or int
            self._npings = nens
        else:
            if len(nens) != 2:
                raise TypeError('nens must be: None (), int, or len 2')
            warnings.warn("A 'start ensemble' is not yet supported "
                          "for the Nortek reader. This function will read "
                          "the entire file, then crop the beginning at "
                          "nens[0].")
            self._npings = nens[1]
            self._n_start = nens[0]
        if endian is None:
            if unpack('<HH', self.read(4)) == (1445, 24):
                endian = '<'
            elif unpack('>HH', self.read(4)) == (1445, 24):
                endian = '>'
            else:
                raise Exception("I/O error: could not determine the "
                                "'endianness' of the file.  Are you sure this is a Nortek "
                                "file?")
        self.endian = endian
        self.f.seek(0, 0)

        # This is the configuration data:
        self.config = {}
        err_msg = ("I/O error: The file does not "
                   "appear to be a Nortek data file.")
        # Read the header:
        if self.read_id() == 5:
            self.read_hw_cfg()
        else:
            raise Exception()
        if self.read_id() == 4:
            self.read_head_cfg()
        else:
            raise Exception(err_msg)
        if self.read_id() == 0:
            self.read_user_cfg()
        else:
            raise Exception(err_msg)
        if self.config['serialNum'][0:3].upper() == 'WPR':
            self.config['config_type'] = 'AWAC'
        elif self.config['serialNum'][0:3].upper() == 'VEC':
            self.config['config_type'] = 'ADV'
        # Initialize the instrument type:
        self._inst = self.config.pop('config_type')
        # This is the position after reading the 'hardware',
        # 'head', and 'user' configuration.
        pnow = self.pos

        # Run the appropriate initialization routine (e.g. init_ADV).
        getattr(self, 'init_' + self._inst)()
        self.f.close()  # This has a small buffer, so close it.
        # This has a large buffer...
        self.f = open(_abspath(fname), 'rb', bufsize)
        self.close = self.f.close
        if self._npings is not None:
            self.n_samp_guess = self._npings
        self.f.seek(pnow, 0)  # Seek to the previous position.

        props = self.data['attrs']
        if self.config['NBurst'] > 0:
            props['DutyCycle_NBurst'] = self.config['NBurst']
            props['DutyCycle_NCycle'] = (self.config['MeasInterval'] *
                                         self.config['fs'])
        self.burst_start = np.zeros(self.n_samp_guess, dtype='bool')
        props['fs'] = self.config['fs']
        props['coord_sys'] = {'XYZ': 'inst',
                              'ENU': 'earth',
                              'beam': 'beam'}[self.config['coord_sys_axes']]
        props['has_imu'] = 0  # Initiate attribute
        if self.debug:
            print('Init completed')

    @property
    def filesize(self,):
        if not hasattr(self, '_filesz'):
            pos = self.pos
            self.f.seek(0, 2)
            # Seek to the end of the file to determine the filesize.
            self._filesz = self.pos
            self.f.seek(pos, 0)  # Return to the initial position.
        return self._filesz

    @property
    def pos(self,):
        return self.f.tell()

    def init_ADV(self,):
        dat = self.data = {'data_vars': {}, 'coords': {}, 'attrs': {},
                           'units': {}, 'sys': {}}
        da = dat['attrs']
        dv = dat['data_vars']
        da['config'] = self.config
        da['inst_make'] = 'Nortek'
        da['inst_model'] = 'Vector'
        da['inst_type'] = 'ADV'
        da['rotate_vars'] = ['vel']
        da['freq'] = self.config['freq']
        da['SerialNum'] = self.config.pop('serialNum')
        dv['beam2inst_orientmat'] = self.config.pop('beam2inst_orientmat')
        da['Comments'] = self.config.pop('Comments')
        # No apparent way to determine how many samples are in a file
        dlta = self.code_spacing('0x11')
        self.config['fs'] = 512 / self.config['AvgInterval']
        self.n_samp_guess = int(self.filesize / dlta + 1)
        self.n_samp_guess *= int(self.config['fs'])

    def init_AWAC(self,):
        dat = self.data = {'data_vars': {}, 'coords': {}, 'attrs': {},
                           'units': {}, 'sys': {}}
        da = dat['attrs']
        dv = dat['data_vars']
        da['config'] = self.config
        da['inst_make'] = 'Nortek'
        da['inst_model'] = 'AWAC'
        da['inst_type'] = 'ADCP'
        da['SerialNum'] = self.config.pop('serialNum')
        dv['beam2inst_orientmat'] = self.config.pop('beam2inst_orientmat')
        da['Comments'] = self.config.pop('Comments')
        da['freq'] = self.config['freq']
        da['n_beams'] = self.config['NBeams']
        da['avg_interval'] = self.config['AvgInterval']
        da['rotate_vars'] = ['vel']
        space = self.code_spacing('0x20')
        if space == 0:
            # code spacing is zero if there's only 1 profile
            self.n_samp_guess = 1
        else:
            self.n_samp_guess = int(self.filesize / space + 1)
        self.config['fs'] = 1. / self.config['AvgInterval']

    def read(self, nbyte):
        byts = self.f.read(nbyte)
        if not (len(byts) == nbyte):
            raise EOFError('Reached the end of the file')
        return byts

    def findnext(self, do_cs=True):
        """Find the next data block by checking the checksum and the 
        sync byte(0xa5)
        """
        sum = np.uint16(int('0xb58c', 0))  # Initialize the sum
        cs = 0
        func = _bitshift8
        func2 = np.uint8
        if self.endian == '<':
            func = np.uint8
            func2 = _bitshift8
        while True:
            val = unpack(self.endian + 'H', self.read(2))[0]
            if func(val) == 165 and (not do_cs or cs == np.uint16(sum)):
                self.f.seek(-2, 1)
                return hex(func2(val))
            sum += cs
            cs = val

    def read_id(self,):
        """Read the next 'ID' from the file.
        """
        self._thisid_bytes = bts = self.read(2)
        tmp = unpack(self.endian + 'BB', bts)
        if self.debug:
            print('Position: {}, codes: {}'.format(self.f.tell(), tmp))
        if tmp[0] != 165:  # This catches a corrupted data block.
            if self.debug:
                print("Corrupted data block sync code (%d, %d) found "
                      "in ping %d. Searching for next valid code..." %
                      (tmp[0], tmp[1], self.c))
            val = int(self.findnext(do_cs=False), 0)
            self.f.seek(2, 1)
            if self.debug:
                print(' ...FOUND {} at position: {}.'.format(val, self.pos))
            return val
        return tmp[1]

    def readnext(self,):
        id = '0x%02x' % self.read_id()
        if id in self.fun_map:
            func_name = self.fun_map[id]
            out = getattr(self, func_name)()  # Should return None
            self._lastread = [func_name[5:]] + self._lastread[:-1]
            return out
        else:
            print('Unrecognized identifier: ' + id)
            self.f.seek(-2, 1)
            return 10

    def readfile(self, nlines=None):
        print('Reading file %s ...' % self.fname)
        retval = None
        try:
            while not retval:
                if self.c == nlines:
                    break
                retval = self.readnext()
                if retval == 10:
                    self.findnext()
                    retval = None
                if self._npings is not None and self.c >= self._npings:
                    if 'microstrain' in self._dtypes:
                        try:
                            self.readnext()
                        except:
                            pass
                    break
        except EOFError:
            print(' end of file at {} bytes.'.format(self.pos))
        else:
            print(' stopped at {} bytes.'.format(self.pos))
        self.c -= 1
        _crop_data(self.data, slice(0, self.c), self.n_samp_guess)

    def findnextid(self, id):
        if id.__class__ is str:
            id = int(id, 0)
        nowid = None
        while nowid != id:
            nowid = self.read_id()
            if nowid == 16:
                shift = 22
            else:
                sz = 2 * unpack(self.endian + 'H', self.read(2))[0]
                shift = sz - 4
            self.f.seek(shift, 1)
        return self.pos

    def code_spacing(self, searchcode, iternum=50):
        """
        Find the spacing, in bytes, between a specific hardware code.
        Repeat this * iternum * times(default 50).
        Returns the average spacing, in bytes, between the code.
        """
        p0 = self.findnextid(searchcode)
        for i in range(iternum):
            try:
                self.findnextid(searchcode)
            except EOFError:
                break
        if self.debug:
            print('p0={}, pos={}, i={}'.format(p0, self.pos, i))
        # Compute the average of the data size:
        return (self.pos - p0) / (i + 1)

    def checksum(self, byts):
        """Perform a checksum on `byts` and read the checksum value.
        """
        if self.do_checksum:
            if not np.sum(unpack(self.endian + str(int(1 + len(byts) / 2)) + 'H',
                                 self._thisid_bytes + byts)) + \
                    46476 - unpack(self.endian + 'H', self.read(2)):

                raise Exception("CheckSum Failed at {}".format(self.pos))
        else:
            self.f.seek(2, 1)

    def read_user_cfg(self,):
        # ID: '0x00 = 00
        if self.debug:
            print('Reading user configuration (0x00) ping #{} @ {}...'
                  .format(self.c, self.pos))
        cfg_u = self.config
        byts = self.read(508)
        tmp = unpack(self.endian +
                     '2x5H13H6s4HI8H2x90H180s6H4xH2x2H2xH30x8H',
                     byts)
        # the first two are the size.
        cfg_u['Transmit'] = {
            'pulse_length': tmp[0],
            'blank_distance': tmp[1],
            'receive_length': tmp[2],
            'time_between_pings': tmp[3],
            'time_between_bursts': tmp[4],
        }
        cfg_u['Npings'] = tmp[5]
        cfg_u['AvgInterval'] = tmp[6]
        cfg_u['NBeams'] = tmp[7]
        cfg_u['TimCtrlReg'] = _int2binarray(tmp[8], 16)
        # From the nortek system integrator manual
        # (note: bit numbering is zero-based)
        treg = cfg_u['TimCtrlReg'].astype(int)
        cfg_u['Profile_Timing'] = ['single', 'continuous'][treg[1]]
        cfg_u['Burst_Mode'] = bool(~treg[2])
        cfg_u['Power Level'] = treg[5] + 2 * treg[6] + 1
        cfg_u['sync-out'] = ['middle', 'end', ][treg[7]]
        cfg_u['Sample_on_Sync'] = bool(treg[8])
        cfg_u['Start_on_Sync'] = bool(treg[9])
        cfg_u['PwrCtrlReg'] = _int2binarray(tmp[9], 16)
        cfg_u['A1'] = tmp[10]
        cfg_u['B0'] = tmp[11]
        cfg_u['B1'] = tmp[12]
        cfg_u['CompassUpdRate'] = tmp[13]
        cfg_u['coord_sys_axes'] = ['ENU', 'XYZ', 'beam'][tmp[14]]
        cfg_u['NBins'] = tmp[15]
        cfg_u['BinLength'] = tmp[16]
        cfg_u['MeasInterval'] = tmp[17]
        cfg_u['DeployName'] = tmp[18].partition(b'\x00')[0].decode('utf-8')
        cfg_u['WrapMode'] = tmp[19]
        cfg_u['ClockDeploy'] = np.array(tmp[20:23])
        cfg_u['DiagInterval'] = tmp[23]
        cfg_u['Mode0'] = _int2binarray(tmp[24], 16)
        cfg_u['AdjSoundSpeed'] = tmp[25]
        cfg_u['NSampDiag'] = tmp[26]
        cfg_u['NBeamsCellDiag'] = tmp[27]
        cfg_u['NPingsDiag'] = tmp[28]
        cfg_u['ModeTest'] = _int2binarray(tmp[29], 16)
        cfg_u['AnaInAddr'] = tmp[30]
        cfg_u['SWVersion'] = tmp[31]
        cfg_u['VelAdjTable'] = np.array(tmp[32:122])
        cfg_u['Comments'] = tmp[122].partition(b'\x00')[0].decode('utf-8')
        cfg_u['Mode1'] = _int2binarray(tmp[123], 16)
        cfg_u['DynPercPos'] = tmp[124]
        cfg_u['T1w'] = tmp[125]
        cfg_u['T2w'] = tmp[126]
        cfg_u['T3w'] = tmp[127]
        cfg_u['NSamp'] = tmp[128]
        cfg_u['NBurst'] = tmp[129]
        cfg_u['AnaOutScale'] = tmp[130]
        cfg_u['CorrThresh'] = tmp[131]
        cfg_u['TiLag2'] = tmp[132]
        cfg_u['QualConst'] = np.array(tmp[133:141])
        self.checksum(byts)
        cfg_u['mode'] = {}
        cfg_u['mode']['user_sound'] = cfg_u['Mode0'][0]
        cfg_u['mode']['diagnostics_mode'] = cfg_u['Mode0'][1]
        cfg_u['mode']['analog_output_mode'] = cfg_u['Mode0'][2]
        cfg_u['mode']['output_format'] = ['Vector', 'ADV'][int(cfg_u['Mode0'][3])]  # noqa
        cfg_u['mode']['vel_scale'] = [1, 0.1][int(cfg_u['Mode0'][4])]
        cfg_u['mode']['serial_output'] = cfg_u['Mode0'][5]
        cfg_u['mode']['reserved_EasyQ'] = cfg_u['Mode0'][6]
        cfg_u['mode']['stage'] = cfg_u['Mode0'][7]
        cfg_u['mode']['output_power'] = cfg_u['Mode0'][8]
        cfg_u['mode']['mode_test_use_DSP'] = cfg_u['ModeTest'][0]
        cfg_u['mode']['mode_test_filter_output'] = ['total', 'correction_only'][int(cfg_u['ModeTest'][1])]  # noqa
        cfg_u['mode']['rate'] = ['1hz', '2hz'][int(cfg_u['Mode1'][0])]
        cfg_u['mode']['cell_position'] = ['fixed', 'dynamic'][int(cfg_u['Mode1'][1])]  # noqa
        cfg_u['mode']['dynamic_pos_type'] = ['pct of mean press', 'pct of min re'][int(cfg_u['Mode1'][2])]  # noqa

    def read_head_cfg(self,):
        # ID: '0x04 = 04
        cfg = self.config
        if self.debug:
            print('Reading head configuration (0x04) ping #{} @ {}...'
                  .format(self.c, self.pos))
        byts = self.read(220)
        tmp = unpack(self.endian + '2x3H12s176s22sH', byts)
        cfg['freq'] = tmp[1]
        cfg['beam2inst_orientmat'] = np.array(
            unpack(self.endian + '9h', tmp[4][8:26])).reshape(3, 3) / 4096.
        self.checksum(byts)

    def read_hw_cfg(self,):
        # ID 0x05 = 05
        cfg = self.config
        if self.debug:
            print('Reading hardware configuration (0x05) ping #{} @ {}...'
                  .format(self.c, self.pos))
        cfg_hw = cfg
        byts = self.read(44)
        tmp = unpack(self.endian + '2x14s6H12xI', byts)
        cfg_hw['serialNum'] = tmp[0][:8].decode('utf-8')
        cfg_hw['ProLogID'] = unpack('B', tmp[0][8:9])[0]
        cfg_hw['ProLogFWver'] = tmp[0][10:].decode('utf-8')
        cfg_hw['config'] = tmp[1]
        cfg_hw['freq'] = tmp[2]
        cfg_hw['PICversion'] = tmp[3]
        cfg_hw['HWrevision'] = tmp[4]
        cfg_hw['recSize'] = tmp[5] * 65536
        cfg_hw['status'] = tmp[6]
        cfg_hw['FWversion'] = tmp[7]
        self.checksum(byts)

    def rd_time(self, strng):
        """Read the time from the first 6bytes of the input string.
        """
        min, sec, day, hour, year, month = unpack('BBBBBB', strng[:6])
        return time.date2epoch(datetime(time._fullyear(_bcd2char(year)),
                                        _bcd2char(month),
                                        _bcd2char(day),
                                        _bcd2char(hour),
                                        _bcd2char(min),
                                        _bcd2char(sec)))[0]

    def _init_data(self, vardict):
        """Initialize the data object according to vardict.

        Parameters
        ----------
        vardict : (dict of :class:`<VarAttrs>`)
          The variable definitions in the :class:`<VarAttrs>` specify
          how to initialize each data variable.

        """
        shape_args = {'n': self.n_samp_guess}
        try:
            shape_args['nbins'] = self.config['NBins']
        except KeyError:
            pass
        for nm, va in list(vardict.items()):
            if va.group is None:
                # These have to stay separated.
                if nm not in self.data:
                    self.data[nm] = va._empty_array(**shape_args)
            else:
                if nm not in self.data[va.group]:
                    self.data[va.group][nm] = va._empty_array(**shape_args)
                    self.data['units'][nm] = va.units

    def read_vec_data(self,):
        # ID: 0x10 = 16
        c = self.c
        dat = self.data
        if self.debug:
            print('Reading vector velocity data (0x10) ping #{} @ {}...'
                  .format(self.c, self.pos))

        if 'vel' not in dat['data_vars']:
            self._init_data(nortek_defs.vec_data)
            self._dtypes += ['vec_data']

        byts = self.read(20)
        ds = dat['sys']
        dv = dat['data_vars']
        (ds['AnaIn2LSB'][c],
         ds['Count'][c],
         dv['PressureMSB'][c],
         ds['AnaIn2MSB'][c],
         dv['PressureLSW'][c],
         ds['AnaIn1'][c],
         dv['vel'][0, c],
         dv['vel'][1, c],
         dv['vel'][2, c],
         dv['amp'][0, c],
         dv['amp'][1, c],
         dv['amp'][2, c],
         dv['corr'][0, c],
         dv['corr'][1, c],
         dv['corr'][2, c]) = unpack(self.endian + '4B2H3h6B', byts)

        self.checksum(byts)
        self.c += 1

    def read_vec_checkdata(self,):
        # ID: 0x07 = 07
        if self.debug:
            print('Reading vector check data (0x07) ping #{} @ {}...'
                  .format(self.c, self.pos))
        byts0 = self.read(6)
        checknow = {}
        tmp = unpack(self.endian + '2x2H', byts0)  # The first two are size.
        checknow['Samples'] = tmp[0]
        n = checknow['Samples']
        checknow['First_samp'] = tmp[1]
        checknow['Amp1'] = tbx._nans(n, dtype=np.uint8) + 8
        checknow['Amp2'] = tbx._nans(n, dtype=np.uint8) + 8
        checknow['Amp3'] = tbx._nans(n, dtype=np.uint8) + 8
        byts1 = self.read(3 * n)
        tmp = unpack(self.endian + (3 * n * 'B'), byts1)
        for idx, nm in enumerate(['Amp1', 'Amp2', 'Amp3']):
            checknow[nm] = np.array(tmp[idx * n:(idx + 1) * n], dtype=np.uint8)
        self.checksum(byts0 + byts1)
        if 'checkdata' not in self.config:
            self.config['checkdata'] = checknow
        else:
            if not isinstance(self.config['checkdata'], list):
                self.config['checkdata'] = [self.config['checkdata']]
            self.config['checkdata'] += [checknow]

    def _sci_data(self, vardict):
        """Convert the data to scientific units accordint to vardict.

        Parameters
        ----------
        vardict : (dict of :class:`<VarAttrs>`)
          The variable definitions in the :class:`<VarAttrs>` specify
          how to scale each data variable.

        """
        for nm, vd in list(vardict.items()):
            if vd.group is None:
                dat = self.data
            else:
                dat = self.data[vd.group]
            retval = vd.sci_func(dat[nm])
            # This checks whether a new data object was created:
            # sci_func returns None if it modifies the existing data.
            if retval is not None:
                dat[nm] = retval

    def sci_vec_data(self,):
        self._sci_data(nortek_defs.vec_data)
        dat = self.data

        dat['data_vars']['pressure'] = (
            dat['data_vars']['PressureMSB'].astype('float32') * 65536 +
            dat['data_vars']['PressureLSW'].astype('float32')) / 1000.
        dat['units']['pressure'] = 'dbar'

        dat['data_vars'].pop('PressureMSB')
        dat['data_vars'].pop('PressureLSW')

        # Apply velocity scaling (1 or 0.1)
        dat['data_vars']['vel'] *= self.config['mode']['vel_scale']

    def read_vec_hdr(self,):
        # ID: '0x12 = 18
        if self.debug:
            print('Reading vector header data (0x12) ping #{} @ {}...'
                  .format(self.c, self.pos))
        byts = self.read(38)
        # The first two are size, the next 6 are time.
        tmp = unpack(self.endian + '8xH7B21x', byts)
        hdrnow = {}
        hdrnow['time'] = self.rd_time(byts[2:8])
        hdrnow['NRecords'] = tmp[0]
        hdrnow['Noise1'] = tmp[1]
        hdrnow['Noise2'] = tmp[2]
        hdrnow['Noise3'] = tmp[3]
        hdrnow['Spare0'] = byts[13:14].decode('utf-8')
        hdrnow['Corr1'] = tmp[5]
        hdrnow['Corr2'] = tmp[6]
        hdrnow['Corr3'] = tmp[7]
        hdrnow['Spare1'] = byts[17:].decode('utf-8')
        self.checksum(byts)
        if 'data_header' not in self.config:
            self.config['data_header'] = hdrnow
        else:
            if not isinstance(self.config['data_header'], list):
                self.config['data_header'] = [self.config['data_header']]
            self.config['data_header'] += [hdrnow]

    def read_vec_sysdata(self,):
        # ID: 0x11 = 17
        c = self.c
        if self.debug:
            print('Reading vector system data (0x11) ping #{} @ {}...'
                  .format(self.c, self.pos))
        dat = self.data
        if self._lastread[:2] == ['vec_checkdata', 'vec_hdr', ]:
            self.burst_start[c] = True
        if 'time' not in dat['coords']:
            self._init_data(nortek_defs.vec_sysdata)
            self._dtypes += ['vec_sysdata']
        byts = self.read(24)
        # The first two are size (skip them).
        dat['coords']['time'][c] = self.rd_time(byts[2:8])
        ds = dat['sys']
        dv = dat['data_vars']
        (dv['batt'][c],
         dv['c_sound'][c],
         dv['heading'][c],
         dv['pitch'][c],
         dv['roll'][c],
         dv['temp'][c],
         dv['error'][c],
         dv['status'][c],
         ds['AnaIn'][c]) = unpack(self.endian + '2H3hH2BH', byts[8:])
        self.checksum(byts)

    def sci_vec_sysdata(self,):
        """Translate the data in the vec_sysdata structure into 
        scientific units.
        """
        dat = self.data
        fs = dat['attrs']['fs']
        self._sci_data(nortek_defs.vec_sysdata)
        t = dat['coords']['time']
        dv = dat['data_vars']
        dat['sys']['_sysi'] = ~np.isnan(t)
        # These are the indices in the sysdata variables
        # that are not interpolated.
        nburst = self.config['NBurst']
        dv['orientation_down'] = tbx._nans(len(t), dtype='bool')
        if nburst == 0:
            num_bursts = 1
            nburst = len(t)
        else:
            num_bursts = int(len(t) // nburst + 1)
        for nb in range(num_bursts):
            iburst = slice(nb * nburst, (nb + 1) * nburst)
            sysi = dat['sys']['_sysi'][iburst]
            if len(sysi) == 0:
                break
            # Skip the first entry for the interpolation process
            inds = np.nonzero(sysi)[0][1:]
            arng = np.arange(len(t[iburst]), dtype=np.float64)
            if len(inds) >= 2:
                p = np.poly1d(np.polyfit(inds, t[iburst][inds], 1))
                t[iburst] = p(arng)
            elif len(inds) == 1:
                t[iburst] = ((arng - inds[0]) / (fs * 3600 * 24) +
                             t[iburst][inds[0]])
            else:
                t[iburst] = (t[iburst][0] + arng / (fs * 24 * 3600))

            tmpd = tbx._nans_like(dv['heading'][iburst])
            # The first status bit should be the orientation.
            tmpd[sysi] = dv['status'][iburst][sysi] & 1
            tbx.fillgaps(tmpd, extrapFlg=True)
            tmpd = np.nan_to_num(tmpd, nan=0)  # nans in pitch roll heading
            slope = np.diff(tmpd)
            tmpd[1:][slope < 0] = 1
            tmpd[:-1][slope > 0] = 0
            dv['orientation_down'][iburst] = tmpd.astype('bool')
        tbx.interpgaps(dv['batt'], t)
        tbx.interpgaps(dv['c_sound'], t)
        tbx.interpgaps(dv['heading'], t)
        tbx.interpgaps(dv['pitch'], t)
        tbx.interpgaps(dv['roll'], t)
        tbx.interpgaps(dv['temp'], t)

    def read_microstrain(self,):
        """Read ADV microstrain sensor (IMU) data
        """
        # 0x71 = 113
        if self.c == 0:
            print('Warning: First "microstrain data" block '
                  'is before first "vector system data" block.')
        else:
            self.c -= 1
        if self.debug:
            print('Reading vector microstrain data (0x71) ping #{} @ {}...'
                  .format(self.c, self.pos))
        byts0 = self.read(4)
        # The first 2 are the size, 3rd is count, 4th is the id.
        ahrsid = unpack(self.endian + '3xB', byts0)[0]
        if hasattr(self, '_ahrsid') and self._ahrsid != ahrsid:
            warnings.warn('AHRS_ID changes mid-file!')

        if ahrsid in [195, 204, 210, 211]:
            self._ahrsid = ahrsid

        c = self.c
        dat = self.data
        dv = dat['data_vars']
        da = dat['attrs']
        da['has_imu'] = 1  # logical
        if 'accel' not in dv:
            self._dtypes += ['microstrain']
            if ahrsid == 195:
                self._orient_dnames = ['accel', 'angrt', 'orientmat']
                dv['accel'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['angrt'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['orientmat'] = tbx._nans((3, 3, self.n_samp_guess),
                                            dtype=np.float32)
                rv = ['accel', 'angrt']
                if not all(x in da['rotate_vars'] for x in rv):
                    da['rotate_vars'].extend(rv)
                dat['units'].update({'accel': 'm/s^2',
                                     'angrt': 'rad/s'})

            if ahrsid in [204, 210]:
                self._orient_dnames = ['accel', 'angrt', 'mag', 'orientmat']
                dv['accel'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['angrt'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['mag'] = tbx._nans((3, self.n_samp_guess),
                                      dtype=np.float32)
                rv = ['accel', 'angrt', 'mag']
                if not all(x in da['rotate_vars'] for x in rv):
                    da['rotate_vars'].extend(rv)
                if ahrsid == 204:
                    dv['orientmat'] = tbx._nans((3, 3, self.n_samp_guess),
                                                dtype=np.float32)
                dat['units'].update({'accel': 'm/s^2',
                                     'angrt': 'rad/s',
                                     'mag': 'gauss'})

            elif ahrsid == 211:
                self._orient_dnames = ['angrt', 'accel', 'mag']
                dv['angrt'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['accel'] = tbx._nans((3, self.n_samp_guess),
                                        dtype=np.float32)
                dv['mag'] = tbx._nans((3, self.n_samp_guess),
                                      dtype=np.float32)
                rv = ['angrt', 'accel', 'mag']
                if not all(x in da['rotate_vars'] for x in rv):
                    da['rotate_vars'].extend(rv)
                dat['units'].update({'accel': 'm/s^2',
                                     'angrt': 'rad/s',
                                     'mag': 'gauss'})
        byts = ''
        if ahrsid == 195:  # 0xc3
            byts = self.read(64)
            dt = unpack(self.endian + '6f9f4x', byts)
            (dv['angrt'][:, c],
             dv['accel'][:, c]) = (dt[0:3], dt[3:6],)
            dv['orientmat'][:, :, c] = ((dt[6:9], dt[9:12], dt[12:15]))
        elif ahrsid == 204:  # 0xcc
            byts = self.read(78)
            # This skips the "DWORD" (4 bytes) and the AHRS checksum
            # (2 bytes)
            dt = unpack(self.endian + '18f6x', byts)
            (dv['accel'][:, c],
             dv['angrt'][:, c],
             dv['mag'][:, c]) = (dt[0:3], dt[3:6], dt[6:9],)
            dv['orientmat'][:, :, c] = ((dt[9:12], dt[12:15], dt[15:18]))
        elif ahrsid == 211:
            byts = self.read(42)
            dt = unpack(self.endian + '9f6x', byts)
            (dv['angrt'][:, c],
             dv['accel'][:, c],
             dv['mag'][:, c]) = (dt[0:3], dt[3:6], dt[6:9],)
        else:
            print('Unrecognized IMU identifier: ' + str(ahrsid))
            self.f.seek(-2, 1)
            return 10
        self.checksum(byts0 + byts)
        self.c += 1  # reset the increment

    def sci_microstrain(self,):
        """Rotate orientation data into ADV coordinate system.
        """
        # MS = MicroStrain
        dv = self.data['data_vars']
        for nm in self._orient_dnames:
            # Rotate the MS orientation data (in MS coordinate system)
            # to be consistent with the ADV coordinate system.
            # (x,y,-z)_ms = (z,y,x)_adv
            (dv[nm][2],
             dv[nm][0]) = (dv[nm][0],
                           -dv[nm][2].copy())
        if 'orientmat' in self._orient_dnames:
            # MS coordinate system is in North-East-Down (NED),
            # we want East-North-Up (ENU)
            dv['orientmat'][:, 2] *= -1
            (dv['orientmat'][:, 0],
             dv['orientmat'][:, 1]) = (dv['orientmat'][:, 1],
                                       dv['orientmat'][:, 0].copy())
        if 'accel' in dv:
            # This value comes from the MS 3DM-GX3 MIP manual
            dv['accel'] *= 9.80665
        if self._ahrsid in [195, 211]:
            # These are DAng and DVel, so we convert them to angrt, accel here
            dv['angrt'] *= self.config['fs']
            dv['accel'] *= self.config['fs']

    def read_awac_profile(self,):
        # ID: '0x20' = 32
        dat = self.data
        if self.debug:
            print('Reading AWAC velocity data (0x20) ping #{} @ {}...'
                  .format(self.c, self.pos))
        nbins = self.config['NBins']
        if 'temp' not in dat['data_vars']:
            self._init_data(nortek_defs.awac_profile)
            self._dtypes += ['awac_profile']

        # Note: docs state there is 'fill' byte at the end, if nbins is odd,
        # but doesn't appear to be the case
        n = self.config['NBeams']
        byts = self.read(116 + n*3 * nbins)
        c = self.c
        dat['coords']['time'][c] = self.rd_time(byts[2:8])
        ds = dat['sys']
        dv = dat['data_vars']
        (dv['error'][c],
         ds['AnaIn1'][c],
         dv['batt'][c],
         dv['c_sound'][c],
         dv['heading'][c],
         dv['pitch'][c],
         dv['roll'][c],
         p_msb,
         dv['status'][c],
         p_lsw,
         dv['temp'][c],) = unpack(self.endian + '7HBB2H', byts[8:28])
        dv['pressure'][c] = (65536 * p_msb + p_lsw)
        # The nortek system integrator manual specifies an 88byte 'spare'
        # field, therefore we start at 116.
        tmp = unpack(self.endian + str(n * nbins) + 'h' +
                     str(n * nbins) + 'B', byts[116:116 + n*3 * nbins])
        for idx in range(n):
            dv['vel'][idx, :, c] = tmp[idx * nbins: (idx + 1) * nbins]
            dv['amp'][idx, :, c] = tmp[(idx + n) * nbins: (idx + n+1) * nbins]
        self.checksum(byts)
        self.c += 1

    def sci_awac_profile(self,):
        self._sci_data(nortek_defs.awac_profile)
        # Calculate the ranges.
        cs_coefs = {2000: 0.0239,
                    1000: 0.0478,
                    600: 0.0797,
                    400: 0.1195}
        h_ang = 25 * (np.pi / 180)  # Head angle is 25 degrees for all awacs.
        # Cell size
        cs = round(float(self.config['BinLength']) / 256. *
                   cs_coefs[self.config['freq']] * np.cos(h_ang), ndigits=2)
        # Blanking distance
        bd = round(self.config['Transmit']['blank_distance'] *
                   0.0229 * np.cos(h_ang) - cs, ndigits=2)

        r = (np.float32(np.arange(self.config['NBins']))+1)*cs + bd
        self.data['coords']['range'] = r
        self.data['attrs']['cell_size'] = cs
        self.data['attrs']['blank_dist'] = bd

    def dat2sci(self,):
        for nm in self._dtypes:
            getattr(self, 'sci_' + nm)()
        for nm in ['data_header', 'checkdata']:
            if nm in self.config and isinstance(self.config[nm], list):
                self.config[nm] = _recatenate(self.config[nm])

    def __exit__(self, type, value, trace):
        self.close()

    def __enter__(self):
        return self


def _crop_data(obj, range, n_lastdim):
    for nm, dat in obj.items():
        if isinstance(dat, np.ndarray) and (dat.shape[-1] == n_lastdim):
            obj[nm] = dat[..., range]


def _recatenate(obj):
    out = type(obj[0])()
    for ky in list(obj[0].keys()):
        if ky in ['__data_groups__', '_type']:
            continue
        val0 = obj[0][ky]
        if isinstance(val0, np.ndarray) and val0.size > 1:
            out[ky] = np.concatenate([val[ky][..., None] for val in obj],
                                     axis=-1)
        else:
            out[ky] = np.array([val[ky] for val in obj])
    return out
