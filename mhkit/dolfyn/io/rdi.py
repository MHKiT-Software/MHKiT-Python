import numpy as np
import xarray as xr
import warnings
from os.path import getsize
from pathlib import Path
import logging

from . import base
from .. import time as tmlib
from . import rdi_lib as lib
from . import rdi_defs as defs
from ..rotate.rdi import _calc_beam_orientmat, _calc_orientmat
from ..rotate.base import _set_coords
from ..rotate.api import set_declination


def read_rdi(
    filename,
    userdata=None,
    nens=None,
    debug_level=-1,
    vmdas_search=False,
    winriver=False,
    **kwargs,
) -> xr.Dataset:
    """
    Read a TRDI binary data file.

    Parameters
    ----------
    filename : string
      Filename of TRDI file to read.
    userdata : True, False, or string of userdata.json filename
      Whether to read the '<base-filename>.userdata.json' file. Default = True
    nens : None, int or 2-element tuple (start, stop)
      Number of pings or ensembles to read from the file.
      Default is None, read entire file
    debug_level : int
      Debug level [0 - 2]. Default = -1
    vmdas_search : bool
      Search from the end of each ensemble for the VMDAS navigation
      block.  The byte offsets are sometimes incorrect. Default = False
    winriver : bool
      If file is winriver or not. Automatically set by dolfyn, this is helpful
      for debugging. Default = False

    Returns
    -------
    ds : xarray.Dataset
      An xarray dataset from the binary instrument data
    """
    # Start debugger logging
    if debug_level >= 0:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        filepath = Path(filename)
        logfile = filepath.with_suffix(".dolfyn.log")
        logging.basicConfig(
            filename=str(logfile),
            filemode="w",
            level=logging.NOTSET,
            format="%(name)s - %(levelname)s - %(message)s",
        )

    # Reads into a dictionary of dictionaries using netcdf naming conventions
    # Should be easier to debug
    rdr = _RDIReader(
        filename, debug_level=debug_level, vmdas_search=vmdas_search, winriver=winriver
    )
    datNB, datBB = rdr.load_data(nens=nens)

    dats = [dat for dat in [datNB, datBB] if dat is not None]

    # Read in userdata
    userdata = base._find_userdata(filename, userdata)
    dss = []
    for dat in dats:
        for nm in userdata:
            dat["attrs"][nm] = userdata[nm]

        # Pass one if only one ds returned
        if not np.isfinite(dat["coords"]["time"][0]):
            continue

        # GPS data not necessarily sampling at the same rate as ADCP DAQ.
        if "time_gps" in dat["coords"]:
            dat = base._remove_gps_duplicates(dat)

        # Convert time coords to dt64
        t_coords = [t for t in dat["coords"] if "time" in t]
        for ky in t_coords:
            dat["coords"][ky] = tmlib.epoch2dt64(dat["coords"][ky])

        # Convert time vars to dt64
        t_data = [t for t in dat["data_vars"] if "time" in t]
        for ky in t_data:
            dat["data_vars"][ky] = tmlib.epoch2dt64(dat["data_vars"][ky])

        # Create xarray dataset from upper level dictionary
        ds = base._create_dataset(dat)
        ds = _set_coords(ds, ref_frame=ds.coord_sys)

        # Create orientation matrices
        if "beam2inst_orientmat" not in ds:
            ds["beam2inst_orientmat"] = xr.DataArray(
                _calc_beam_orientmat(ds.beam_angle, ds.beam_pattern == "convex"),
                coords={"x1": [1, 2, 3, 4], "x2": [1, 2, 3, 4]},
                dims=["x1", "x2"],
                attrs={"units": "1", "long_name": "Rotation Matrix"},
            )

        if "orientmat" not in ds:
            ds["orientmat"] = _calc_orientmat(ds)

        # Check magnetic declination if provided via software and/or userdata
        _set_rdi_declination(ds, filename, inplace=True)

        dss += [ds]

    if len(dss) == 2:
        warnings.warn(
            "\nTwo profiling configurations retrieved from file" "\nReturning first."
        )

    # Close handler
    if debug_level >= 0:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

    return dss[0]


def _set_rdi_declination(dat, fname, inplace):
    """
    If magnetic_var_deg is set, this means that the declination is already
    included in the heading and in the velocity data.
    """

    declin = dat.attrs.pop("declination", None)  # userdata declination

    if dat.attrs["magnetic_var_deg"] != 0:  # from TRDI software if set
        dat.attrs["declination"] = dat.attrs["magnetic_var_deg"]
        dat.attrs["declination_in_orientmat"] = 1  # logical

    if dat.attrs["magnetic_var_deg"] != 0 and declin is not None:
        warnings.warn(
            "'magnetic_var_deg' is set to {:.2f} degrees in the binary "
            "file '{}', AND 'declination' is set in the 'userdata.json' "
            "file. DOLfYN WILL USE THE VALUE of {:.2f} degrees in "
            "userdata.json. If you want to use the value in "
            "'magnetic_var_deg', delete the value from userdata.json and "
            "re-read the file.".format(dat.attrs["magnetic_var_deg"], fname, declin)
        )
        dat.attrs["declination"] = declin

    if declin is not None:
        set_declination(dat, declin, inplace)


class _RDIReader:
    def __init__(
        self, fname, navg=1, debug_level=-1, vmdas_search=False, winriver=False
    ):
        self.fname = base._abspath(fname)
        print("\nReading file {} ...".format(fname))
        self._debug_level = debug_level
        self._vmdas_search = vmdas_search
        self._winrivprob = winriver
        self._vm_source = 0
        self._pos = 0
        self.progress = 0
        self._cfac32 = np.float32(180 / 2**31)  # signed 32 to float
        self._cfac16 = np.float32(180 / 2**15)  # unsigned16 to float
        self._fixoffset = 0
        self._nbyte = 0
        self.n_cells_diff = 0
        self.n_cells_sl = 0
        self.cs_diff = 0
        self.cs = []
        self.cfg = {}
        self.cfgbb = {}
        self.hdr = {}
        self.f = lib.bin_reader(self.fname)

        # Check header, double buffer, and get filesize
        self._filesize = getsize(self.fname)
        space = self.code_spacing()  # '0x7F'
        self._npings = self._filesize // space
        if self._debug_level > -1:
            logging.info("Done: {}".format(self.cfg))
            logging.info("self._bb {}".format(self._bb))
            logging.info("self.cfgbb: {}".format(self.cfgbb))
        self.f.seek(self._pos, 0)
        self.n_avg = navg

        self.ensemble = lib._ensemble(self.n_avg, self.cfg["n_cells"])
        if self._bb:
            self.ensembleBB = lib._ensemble(self.n_avg, self.cfgbb["n_cells"])

        self.vars_read = lib._variable_setlist(["time"])
        if self._bb:
            self.vars_readBB = lib._variable_setlist(["time"])

    def code_spacing(self, iternum=50):
        """
        Returns the average spacing, in bytes, between pings.
        Repeat this * iternum * times(default 50).
        """
        fd = self.f
        p0 = self._pos
        # Get basic header data and check dual profile
        if not self.read_hdr():
            raise RuntimeError("No header in this file")
        self._bb = self.check_for_double_buffer()

        # Turn off debugging to check code spacing
        debug_level = self._debug_level
        self._debug_level = -1
        for i in range(iternum):
            try:
                self.read_hdr()
            except:
                break
        # Compute the average of the data size:
        size = (self._pos - p0) / (i + 1)
        self.f = fd
        self._pos = p0
        self._debug_level = debug_level
        return size

    def read_hdr(self):
        """Scan file until 7f7f is found"""
        if not self.search_buffer():
            return False
        self._pos = self.f.tell() - 2
        self.read_hdrseg()
        return True

    def read_hdrseg(self):
        fd = self.f
        hdr = self.hdr
        hdr["nbyte"] = fd.read_i16(1)
        spare = fd.read_ui8(1)
        ndat = fd.read_ui8(1)
        hdr["dat_offsets"] = fd.read_ui16(ndat)
        self._nbyte = 4 + ndat * 2

    def check_for_double_buffer(self):
        """
        VMDAS will record two buffers in NB or NB/BB mode, so we need to
        figure out if that is happening here
        """
        found = False
        pos = self.f.pos
        if self._debug_level > -1:
            logging.info(self.hdr)
            logging.info("pos {}".format(pos))
        self.id_positions = {}
        for offset in self.hdr["dat_offsets"]:
            self.f.seek(offset + pos - self.hdr["dat_offsets"][0], rel=0)
            id = self.f.read_ui16(1)
            self.id_positions[id] = offset
            if self._debug_level > -1:
                logging.info("id {} offset {}".format(id, offset))
            if id == 1:
                defs.read_fixed(self, bb=True)
                found = True
            elif id == 0:
                defs.read_fixed(self, bb=False)
            elif id == 16:
                defs.read_fixed_sl(self)  # bb=True
            elif id == 8192:
                self._vmdas_search = True
        return found

    def load_data(self, nens=None):
        """Main function run after reader class is initiated."""
        if nens is None:
            # Attempt to overshoot WinRiver2 or *Pro filesize
            if (self.cfg["coord_sys"] == "ship") or (
                self.cfg["inst_model"]
                in [
                    "RiverPro",
                    "StreamPro",
                ]
            ):
                self._nens = int(self._filesize / self.hdr["nbyte"] / self.n_avg * 1.1)
            else:
                # Attempt to overshoot other instrument filesizes
                self._nens = int(self._npings / self.n_avg)
        elif nens.__class__ is tuple or nens.__class__ is list:
            raise Exception("    `nens` must be a integer")
        else:
            self._nens = nens
        if self._debug_level > -1:
            logging.info("  taking data from pings 0 - %d" % self._nens)
            logging.info("  %d ensembles will be produced.\n" % self._nens)
        self.init_data()

        for iens in range(self._nens):
            if not self.read_buffer():
                self.remove_end(iens)
                break
            self.ensemble.clean_data()
            if self._bb:
                self.ensembleBB.clean_data()
            ens = [self.ensemble]
            vars = [self.vars_read]
            datl = [self.outd]
            cfgl = [self.cfg]
            if self._bb:
                ens += [self.ensembleBB]
                vars += [self.vars_readBB]
                datl += [self.outdBB]
                cfgl += [self.cfgbb]

            for var, en, dat in zip(vars, ens, datl):
                for nm in var:
                    dat = self.save_profiles(dat, nm, en, iens)
                # reset flag after all variables run
                self.n_cells_diff = 0

                # Set clock
                clock = en.rtc[:, :]
                if clock[0, 0] < 100:
                    clock[0, :] += defs.century
                try:
                    dates = tmlib.date2epoch(
                        tmlib.datetime(
                            *clock[:6, 0], microsecond=int(float(clock[6, 0]) * 10000)
                        )
                    )[0]
                except ValueError:
                    warnings.warn(
                        "Invalid time stamp in ping {}.".format(
                            int(self.ensemble.number[0])
                        )
                    )
                    dat["coords"]["time"][iens] = np.nan
                else:
                    dat["coords"]["time"][iens] = np.median(dates)

        # Finalize dataset (runs through both nb and bb)
        for dat, cfg in zip(datl, cfgl):
            dat, cfg = self.cleanup(dat, cfg)
            dat = self.finalize(dat)
            if "vel_bt" in dat["data_vars"]:
                dat["attrs"]["rotate_vars"].append("vel_bt")

        datbb = self.outdBB if self._bb else None
        return self.outd, datbb

    def init_data(self):
        """Initiate data structure"""
        outd = {
            "data_vars": {},
            "coords": {},
            "attrs": {},
            "units": {},
            "long_name": {},
            "standard_name": {},
            "sys": {},
        }
        outd["attrs"]["inst_make"] = "TRDI"
        outd["attrs"]["inst_type"] = "ADCP"
        outd["attrs"]["rotate_vars"] = [
            "vel",
        ]
        # Currently RDI doesn't use IMUs
        outd["attrs"]["has_imu"] = 0
        if self._bb:
            outdbb = {
                "data_vars": {},
                "coords": {},
                "attrs": {},
                "units": {},
                "long_name": {},
                "standard_name": {},
                "sys": {},
            }
            outdbb["attrs"]["inst_make"] = "TRDI"
            outdbb["attrs"]["inst_type"] = "ADCP"
            outdbb["attrs"]["rotate_vars"] = [
                "vel",
            ]
            outdbb["attrs"]["has_imu"] = 0

        # Preallocate variables and data sizes
        for nm in defs.data_defs:
            outd = lib._idata(
                outd, nm, sz=lib._get_size(nm, self._nens, self.cfg["n_cells"])
            )
        self.outd = outd

        if self._bb:
            for nm in defs.data_defs:
                outdbb = lib._idata(
                    outdbb, nm, sz=lib._get_size(nm, self._nens, self.cfgbb["n_cells"])
                )
            self.outdBB = outdbb
            if self._debug_level > 1:
                logging.info(np.shape(outdbb["data_vars"]["vel"]))

        if self._debug_level > 1:
            logging.info("{} ncells, not BB".format(self.cfg["n_cells"]))
            if self._bb:
                logging.info("{} ncells, BB".format(self.cfgbb["n_cells"]))

    def read_buffer(self):
        """Read through the file"""
        fd = self.f
        self.ensemble.k = -1  # so that k+=1 gives 0 on the first loop.
        if self._bb:
            self.ensembleBB.k = -1  # so that k+=1 gives 0 on the first loop.
        self.print_progress()
        hdr = self.hdr
        while self.ensemble.k < self.ensemble.n_avg - 1:
            if not self.search_buffer():
                return False
            startpos = fd.tell() - 2
            self.read_hdrseg()
            if self._debug_level > -1:
                logging.info("Read Header", hdr)
            byte_offset = self._nbyte + 2
            self._read_vmdas = False
            for n in range(len(hdr["dat_offsets"])):
                id = fd.read_ui16(1)
                if self._debug_level > 0:
                    logging.info(f"n {n}: {id} {id:04x}")
                self.print_pos()
                retval = self.read_dat(id)

                if retval == "FAIL":
                    break
                byte_offset += self._nbyte
                if n < (len(hdr["dat_offsets"]) - 1):
                    oset = hdr["dat_offsets"][n + 1] - byte_offset
                    if oset != 0:
                        if self._debug_level > 0:
                            logging.debug("  %s: Adjust location by %d\n" % (id, oset))
                        fd.seek(oset, 1)
                    byte_offset = hdr["dat_offsets"][n + 1]
                else:
                    if hdr["nbyte"] - 2 != byte_offset:
                        if not self._winrivprob:
                            if self._debug_level > 0:
                                logging.debug(
                                    "  {:d}: Adjust location by {:d}\n".format(
                                        id, hdr["nbyte"] - 2 - byte_offset
                                    )
                                )
                            self.f.seek(hdr["nbyte"] - 2 - byte_offset, 1)
                    byte_offset = hdr["nbyte"] - 2
            # Check for vmdas again because vmdas doesn't set the offsets
            # correctly, and we need this info:
            if not self._read_vmdas and self._vmdas_search:
                if self._debug_level > 0:
                    logging.info("Searching for vmdas nav data. Going to next ensemble")
                self.search_buffer()
                # now go back to where vmdas would be:
                fd.seek(-98, 1)
                id = self.f.read_ui16(1)
                if id is not None:
                    if self._debug_level > 0:
                        logging.info(f"Found {id:04d}")
                    if id == 8192:
                        self.read_dat(id)
            readbytes = fd.tell() - startpos
            offset = hdr["nbyte"] + 2 - readbytes
            self.check_offset(offset, readbytes)
            self.print_pos(byte_offset=byte_offset)

        return True

    def search_buffer(self):
        """
        Check to see if the next bytes indicate the beginning of a
        data block.  If not, search for the next data block, up to
        _search_num times.
        """
        fd = self.f
        id = fd.read_ui8(2)
        if id is None:
            return False
        cfgid = list(id)
        pos_7f79 = False
        search_cnt = 0

        if self._debug_level > -1:
            logging.info("pos {}".format(fd.pos))
            logging.info("cfgid0: [{:x}, {:x}]".format(*cfgid))
        # If not [127, 127] or if the file ends in the next ensemble
        while (cfgid != [127, 127]) or self.check_eof():
            if cfgid == [127, 121]:
                # Search for the next header or the end of the file
                skipbytes = fd.read_i16(1)
                fd.seek(skipbytes - 2, 1)
                id = fd.read_ui8(2)
                if id is None:  # EOF
                    return False
                cfgid = list(id)
                pos_7f79 = True
            else:
                # Search til we find something or hit the end of the file
                search_cnt += 1
                nextbyte = fd.read_ui8(1)
                if nextbyte is None:  # EOF
                    return False
                cfgid[0] = cfgid[1]
                cfgid[1] = nextbyte

        if pos_7f79 and self._debug_level > -1:
            logging.info("Skipped junk data: [{:x}, {:x}]".format(*[127, 121]))

        if search_cnt > 0:
            if self._debug_level > 0:
                logging.info(
                    "  Searched {} bytes to find next "
                    "valid ensemble start [{:x}, {:x}]\n".format(search_cnt, *cfgid)
                )

        return True

    def check_eof(self):
        """Returns True if next header is bad or at end of file."""
        fd = self.f
        out = True
        numbytes = fd.read_i16(1)
        # Search for next config id
        if numbytes > 0:
            fd.seek(numbytes - 2, 1)
            cfgid = fd.read_ui8(2)
            if cfgid is None:
                if self._debug_level > 1:
                    logging.info("EOF")
                return True
            # Make sure one is found, either 7f7f or 7f79
            if len(cfgid) == 2:
                fd.seek(-numbytes - 2, 1)
                if cfgid[0] == 127 and cfgid[1] in [127, 121]:
                    out = False
        else:
            fd.seek(-2, 1)
        return out

    def print_progress(self):
        """Print the buffer progress, used for debugging."""
        self.progress = self.f.tell()
        if self._debug_level > 1:
            logging.debug(
                "  pos %0.0fmb/%0.0fmb\n"
                % (self.f.tell() / 1048576, self._filesize / 1048576)
            )
        if (self.f.tell() - self.progress) < 1048576:
            return

    def print_pos(self, byte_offset=-1):
        """Print the position in the file, used for debugging."""
        if self._debug_level > 1:
            if hasattr(self, "ensemble"):
                k = self.ensemble.k
            else:
                k = 0
            logging.debug(
                f"  pos: {self.f.tell()}, pos_: {self._pos}, nbyte: {self._nbyte}, k: {k}, byte_offset: {byte_offset}"
            )

    def read_dat(self, id):
        """Main function map used to read or skip stored IDs"""
        function_map = {
            # 0000 1st profile fixed leader
            0: (defs.read_fixed, [False]),
            # 0001 2nd profile fixed leader
            1: (defs.read_fixed, [True]),
            # 0010 Surface layer fixed leader (RiverPro & StreamPro)
            16: (defs.read_fixed_sl, []),
            # 0080 1st profile variable leader
            128: (defs.read_var, [False]),
            # 0081 2nd profile variable leader
            129: (defs.read_var, [True]),
            # 0100 1st profile velocity
            256: (defs.read_vel, [0]),
            # 0101 2nd profile velocity
            257: (defs.read_vel, [1]),
            # 0103 Waves first leader
            259: (defs.skip_Nbyte, [74]),
            # 0110 Surface layer velocity (RiverPro & StreamPro)
            272: (defs.read_vel, [2]),
            # 0200 1st profile correlation
            512: (defs.read_corr, [0]),
            # 0201 2nd profile correlation
            513: (defs.read_corr, [1]),
            # 0203 Waves data
            515: (defs.skip_Nbyte, [186]),
            # 020C Ambient sound profile
            524: (defs.skip_Nbyte, [4]),
            # 0210 Surface layer correlation (RiverPro & StreamPro)
            528: (defs.read_corr, [2]),
            # 0300 1st profile amplitude
            768: (defs.read_amp, [0]),
            # 0301 2nd profile amplitude
            769: (defs.read_amp, [1]),
            # 0302 Beam 5 Sum of squared velocities
            770: (defs.skip_Ncol, []),
            # 0303 Waves last leader
            771: (defs.skip_Ncol, [18]),
            # 0310 Surface layer amplitude (RiverPro & StreamPro)
            784: (defs.read_amp, [2]),
            # 0400 1st profile % good
            1024: (defs.read_prcnt_gd, [0]),
            # 0401 2nd profile pct good
            1025: (defs.read_prcnt_gd, [1]),
            # 0403 Waves HPR data
            1027: (defs.skip_Nbyte, [6]),
            # 0410 Surface layer pct good (RiverPro & StreamPro)
            1040: (defs.read_prcnt_gd, [2]),
            # 0500 1st profile status
            1280: (defs.read_status, [0]),
            # 0501 2nd profile status
            1281: (defs.read_status, [1]),
            # 0510 Surface layer status (RiverPro & StreamPro)
            1296: (defs.read_status, [2]),
            1536: (defs.read_bottom, []),  # 0600 bottom tracking
            1793: (defs.skip_Ncol, [4]),  # 0701 number of pings
            1794: (defs.skip_Ncol, [4]),  # 0702 sum of squared vel
            1795: (defs.skip_Ncol, [4]),  # 0703 sum of velocities
            2560: (defs.skip_Ncol, []),  # 0A00 Beam 5 velocity
            2816: (defs.skip_Ncol, []),  # 0B00 Beam 5 correlation
            3072: (defs.skip_Ncol, []),  # 0C00 Beam 5 amplitude
            3328: (defs.skip_Ncol, []),  # 0D00 Beam 5 pct_good
            # Fixed attitude data format for Ocean Surveyor ADCPs
            3000: (defs.skip_Nbyte, [32]),
            3841: (defs.skip_Nbyte, [38]),  # 0F01 Beam 5 leader
            8192: (defs.read_vmdas, []),  # 2000
            # 2013 Navigation parameter data
            8211: (defs.skip_Nbyte, [83]),
            8226: (defs.read_winriver2, []),  # 2022
            8448: (defs.read_winriver, []),  # 2100
            8449: (defs.read_winriver, []),  # 2101
            8450: (defs.read_winriver, []),  # 2102
            8451: (defs.read_winriver, []),  # 2103
            8452: (defs.read_winriver, []),  # 2104
            # 3200 Transformation matrix
            12800: (defs.skip_Nbyte, [32]),
            # 3000 Fixed attitude data format for Ocean Surveyor ADCPs
            12288: (defs.skip_Nbyte, [32]),
            12496: (defs.skip_Nbyte, [24]),  # 30D0
            12504: (defs.skip_Nbyte, [48]),  # 30D8
            # 4100 beam 5 range
            16640: (defs.read_alt, []),
            # 4400 Firmware status data (RiverPro & StreamPro)
            17408: (defs.skip_Nbyte, [28]),
            # 4401 Auto mode setup (RiverPro & StreamPro)
            17409: (defs.skip_Nbyte, [82]),
            # 5803 High resolution bottom track velocity
            22531: (defs.skip_Nbyte, [68]),
            # 5804 Bottom track range
            22532: (defs.skip_Nbyte, [21]),
            # 5901 ISM (IMU) data
            22785: (defs.skip_Nbyte, [65]),
            # 5902 Ping attitude
            22786: (defs.skip_Nbyte, [105]),
            # 7001 ADC data
            28673: (defs.skip_Nbyte, [14]),
        }
        # Call the correct function:
        if self._debug_level > 1:
            logging.debug(f"Trying to Read {id}")
        if id in function_map:
            if self._debug_level > 1:
                logging.info("  Reading code {}...".format(hex(id)))
            retval = function_map.get(id)[0](self, *function_map[id][1])
            if retval:
                return retval
            if self._debug_level > 1:
                logging.info("    success!")
        else:
            self.read_nocode(id)

    def read_nocode(self, id):
        """Identify filler or unknown bytes and bypass them"""
        # Skipping bytes from codes 0340-30FC, commented if needed
        hxid = hex(id)
        if hxid[2:4] == "30":
            logging.warning("Skipping bytes from codes 0340-30FC")
            # I want to count the number of 1s in the middle 4 bits
            # of the 2nd two bytes.
            # 60 is a 0b00111100 mask
            nflds = bin(int(hxid[3]) & 60).count("1") + bin(int(hxid[4]) & 60).count(
                "1"
            )
            # I want to count the number of 1s in the highest
            # 2 bits of byte 3
            # 3 is a 0b00000011 mask:
            dfac = bin(int(hxid[3], 0) & 3).count("1")
            defs.skip_Nbyte(self, 12 * nflds * dfac)
        else:
            if self._debug_level > -1:
                logging.warning("  Unrecognized ID code: %0.4X" % id)
            self.skip_nocode(id)

    def skip_nocode(self, id):
        """
        Skips bytes when an ID code is not found in the function map.

        This method calculates the byte length to skip based on the positions
        of known ID codes and uses this length to bypass filler or irrelevant data.

        Parameters
        ----------
        id : int
            The ID code that is not present in the function map.
        """
        offsets = list(self.id_positions.values())
        idx = np.where(offsets == self.id_positions[id])[0][0]
        byte_len = offsets[idx + 1] - offsets[idx] - 2

        defs.skip_Nbyte(self, byte_len)
        if self._debug_level > -1:
            logging.debug(f"Skipping ID code {id}\n")

    def check_offset(self, offset, readbytes):
        """
        Checks and adjusts the file position based on the distance to the nearest function ID.

        If the provided `offset` differs from the expected value and `_fixoffset` is zero,
        this method updates `_fixoffset` and adjusts the file position in the data file
        (`self.f`) accordingly. This adjustment is logged if `_debug_level` is set to a
        positive value.

        Parameters
        ----------
        offset : int
            The current offset from the expected position.
        readbytes : int
            The number of bytes that have been read so far.
        """
        fd = self.f
        if offset != 4 and self._fixoffset == 0:
            if self._debug_level > 0:
                if fd.tell() == self._filesize:
                    logging.error(
                        " EOF reached unexpectedly - discarding this last ensemble\n"
                    )
                else:
                    logging.debug(
                        "  Adjust location by {:d} (readbytes={:d},hdr['nbyte']={:d})\n".format(
                            offset, readbytes, self.hdr["nbyte"]
                        )
                    )
            self._fixoffset = offset - 4
        fd.seek(4 + self._fixoffset, 1)

    def remove_end(self, iens):
        """
        Removes incomplete measurements from the dataset.

        This method cleans up any partially read data by truncating measurements
        to the specified ensemble index (`iens`). This is typically called upon
        reaching the end of the file to ensure only complete data is retained.

        Parameters
        ----------
        iens : int
            The index up to which data is considered complete and should be retained.
        """
        dat = self.outd
        if self._debug_level > 0:
            logging.info("  Encountered end of file.  Cleaning up data.")
        for nm in self.vars_read:
            lib._setd(dat, nm, lib._get(dat, nm)[..., :iens])

    def save_profiles(self, dat, nm, en, iens):
        """
        Reformats profile measurements in the retrieved measurements.

        This method processes profile measurements from individual pings,
        adapting to changing cell counts and cell sizes as needed (from the WinRiver2
        program with one of the ***Pro ADCPs).

        Parameters
        ----------
        dat : dict
            Raw data dictionary
        nm : str
            The name of the profile variable
        en : dict
            The dictionary containing ensemble profiles
        iens : int
            The index of the current ensemble

        Returns
        -------
        dict
            The updated dataset dictionary with the reformatted profile measurements.
        """
        ds = lib._get(dat, nm)
        if self.n_avg == 1:
            bn = en[nm][..., 0]
        else:
            bn = np.nanmean(en[nm], axis=-1)

        # If n_cells has changed (RiverPro/StreamPro WinRiver transects)
        if len(ds.shape) == 3:
            if "_sl" in nm:
                # This works here b/c the max number of surface layer cells
                # is smaller than the min number of normal profile cells used.
                # Extra nan cells created after this if-statement
                # are trimmed off in self.cleanup.
                bn = bn[: self.cfg["n_cells_sl"]]
            else:
                # Set bn to current ping size
                bn = bn[: self.cfg["n_cells"]]
                # If n_cells has increased, we also need to increment defs
                if self.n_cells_diff > 0:
                    a = np.empty((self.n_cells_diff, ds.shape[1], ds.shape[2])) * np.nan
                    ds = np.append(ds, a.astype(ds.dtype), axis=0)
                    lib._setd(dat, nm, ds)
            # If the number of cells decreases, set extra cells to nan instead of
            # whatever is stuck in memory
            if ds.shape[0] != bn.shape[0]:
                n_cells = ds.shape[0] - bn.shape[0]
                a = np.empty((n_cells, bn.shape[1])) * np.nan
                bn = np.append(bn, a.astype(ds.dtype), axis=0)

        # Keep track of when the cell size changes
        if self.cs_diff:
            self.cs.append([iens, self.cfg["cell_size"]])
            self.cs_diff = 0

        # Then copy the ensemble to the dataset.
        ds[..., iens] = bn
        lib._setd(dat, nm, ds)

        return dat

    def cleanup(self, dat, cfg):
        """
        Cleans up recorded data by adjusting variable cell sizes and profile ranges.

        This method handles adjustments when cell sizes change during data collection,
        performing depth-bin averaging for smaller cells if needed. It also updates
        the configuration data, range coordinates, and manages any surface layer profiles.

        Parameters
        ----------
        dat : dict
            The dataset dictionary containing data variables and coordinates to be cleaned up.
        cfg : dict
            Configuration dictionary, which is updated with cell size, range, and additional
            attributes after cleanup.

        Returns
        -------
        tuple
            - dict : The updated dataset dictionary with cleaned data.
            - dict : The updated configuration dictionary with new attributes.
        """
        # Clean up changing cell size, if necessary
        cs = np.array(self.cs, dtype=np.float32)
        cell_sizes = cs[:, 1]

        # If cell sizes change, depth-bin average the smaller cell sizes
        if len(self.cs) > 1:
            bins_to_merge = cell_sizes.max() / cell_sizes
            idx_start = cs[:, 0].astype(int)
            idx_end = np.append(cs[1:, 0], self._nens).astype(int)

            dv = dat["data_vars"]
            for var in dv:
                if (len(dv[var].shape) == 3) and ("_sl" not in var):
                    # Create a new NaN var to save data in
                    new_var = (np.zeros(dv[var].shape) * np.nan).astype(dv[var].dtype)
                    # For each cell size change, reshape and bin-average
                    for id1, id2, b in zip(idx_start, idx_end, bins_to_merge):
                        array = np.transpose(dv[var][..., id1:id2])
                        bin_arr = np.transpose(np.mean(self.reshape(array, b), axis=-1))
                        new_var[: len(bin_arr), :, id1:id2] = bin_arr
                    # Reset data. This often leaves nan data at farther ranges
                    dv[var] = new_var

        # Set cell size and range
        cfg["n_cells"] = self.ensemble["n_cells"]
        cfg["cell_size"] = round(cell_sizes.max(), 3)
        dat["coords"]["range"] = (
            cfg["bin1_dist_m"] + np.arange(cfg["n_cells"]) * cfg["cell_size"]
        ).astype(np.float32)

        # Save configuration data as attributes
        for nm in cfg:
            dat["attrs"][nm] = cfg[nm]

        # Clean up surface layer profiles
        if "surface_layer" in cfg:  # RiverPro/StreamPro
            dat["coords"]["range_sl"] = (
                cfg["bin1_dist_m_sl"]
                + np.arange(0, self.n_cells_sl) * cfg["cell_size_sl"]
            )
            # Trim off extra nan data
            dv = dat["data_vars"]
            for var in dv:
                if "sl" in var:
                    dv[var] = dv[var][: self.n_cells_sl]
            dat["attrs"]["rotate_vars"].append("vel_sl")

        return dat, cfg

    def reshape(self, arr, n_bin=None):
        """
        Reshapes the input array `arr` to a shape of (..., n, n_bin).

        Parameters
        ----------
        arr : np.ndarray
            The array to reshape. The last dimension of `arr` will be divided into
            `(n, n_bin)` based on the value of `n_bin`.
        n_bin : int or float, optional
            The bin size for reshaping. If `n_bin` is an integer, it divides the
            last dimension directly. If not, indices are adjusted, and excess
            bins may be removed. Default is None.

        Returns
        -------
        np.ndarray
            The reshaped array with shape (..., n, n_bin).
        """

        out = np.zeros(
            list(arr.shape[:-1]) + [int(arr.shape[-1] // n_bin), int(n_bin)],
            dtype=arr.dtype,
        )
        shp = out.shape
        if np.mod(n_bin, 1) == 0:
            # n_bin needs to be int
            n_bin = int(n_bin)
            # If n_bin is an integer, we can do this simply.
            out[..., :n_bin] = (arr[..., : (shp[-2] * shp[-1])]).reshape(shp, order="C")
        else:
            inds = np.arange(np.prod(shp[-2:])) * n_bin // int(n_bin)
            # If there are too many indices, drop one bin
            if inds[-1] >= arr.shape[-1]:
                inds = inds[: -int(n_bin)]
                shp[-2] -= 1
                out = out[..., 1:, :]
            n_bin = int(n_bin)
            out[..., :n_bin] = (arr[..., inds]).reshape(shp, order="C")
            n_bin = int(n_bin)

        return out

    def finalize(self, dat):
        """
        This method cleans up the dataset by removing any attributes that were
        defined but not loaded, updates configuration attributes, and sets the
        sampling frequency (fs) based on the data source program. Additionally,
        it adjusts the axes of certain variables as defined in data_defs.

        Parameters
        ----------
        dat : dict
            The dataset dictionary to be finalized. This dictionary is modified
            in place by removing unused attributes, setting configuration values
            as attributes, and calculating `fs`.

        Returns
        -------
        dict
            The finalized dataset dictionary with cleaned attributes and added metadata.
        """
        for nm in set(defs.data_defs.keys()) - self.vars_read:
            lib._pop(dat, nm)
        for nm in self.cfg:
            dat["attrs"][nm] = self.cfg[nm]

        # VMDAS and WinRiver have different set sampling frequency
        da = dat["attrs"]
        if ("sourceprog" in da) and (
            da["sourceprog"].lower() in ["vmdas", "winriver", "winriver2"]
        ):
            da["fs"] = round(1 / np.median(np.diff(dat["coords"]["time"])), 2)
        else:
            da["fs"] = 1 / (da["sec_between_ping_groups"] * da["pings_per_ensemble"])

        for nm in defs.data_defs:
            shp = defs.data_defs[nm][0]
            if len(shp) and shp[0] == "nc" and lib._in_group(dat, nm):
                lib._setd(dat, nm, np.swapaxes(lib._get(dat, nm), 0, 1))

        return dat
