import numpy as np
import xarray as xr
import warnings
from os.path import getsize
from pathlib import Path
import logging

from .rdi_lib import bin_reader
from . import rdi_defs as defs
from .base import _find_userdata, _create_dataset, _abspath
from .. import time as tmlib
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
):
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
    userdata = _find_userdata(filename, userdata)
    dss = []
    for dat in dats:
        for nm in userdata:
            dat["attrs"][nm] = userdata[nm]

        # Pass one if only one ds returned
        if not np.isfinite(dat["coords"]["time"][0]):
            continue

        # GPS data not necessarily sampling at the same rate as ADCP DAQ.
        if "time_gps" in dat["coords"]:
            dat = _remove_gps_duplicates(dat)

        # Convert time coords to dt64
        t_coords = [t for t in dat["coords"] if "time" in t]
        for ky in t_coords:
            dat["coords"][ky] = tmlib.epoch2dt64(dat["coords"][ky])

        # Convert time vars to dt64
        t_data = [t for t in dat["data_vars"] if "time" in t]
        for ky in t_data:
            dat["data_vars"][ky] = tmlib.epoch2dt64(dat["data_vars"][ky])

        # Create xarray dataset from upper level dictionary
        ds = _create_dataset(dat)
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

        # VMDAS applies gps correction on velocity in .ENX files only
        if filename.rsplit(".")[-1] == "ENX":
            ds.attrs["vel_gps_corrected"] = 1
        else:  # (not ENR or ENS) or WinRiver files
            ds.attrs["vel_gps_corrected"] = 0

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


def _remove_gps_duplicates(dat):
    """
    Removes duplicate and nan timestamp values in 'time_gps' coordinate,
    and add hardware (ADCP DAQ) timestamp corresponding to GPS acquisition
    (in addition to the GPS unit's timestamp).
    """

    dat["data_vars"]["hdwtime_gps"] = dat["coords"]["time"]

    # Remove duplicate timestamp values, if applicable
    dat["coords"]["time_gps"], idx = np.unique(
        dat["coords"]["time_gps"], return_index=True
    )
    # Remove nan values, if applicable
    nan = np.zeros(dat["coords"]["time"].shape, dtype=bool)
    if any(np.isnan(dat["coords"]["time_gps"])):
        nan = np.isnan(dat["coords"]["time_gps"])
        dat["coords"]["time_gps"] = dat["coords"]["time_gps"][~nan]

    for key in dat["data_vars"]:
        if ("gps" in key) or ("nmea" in key):
            dat["data_vars"][key] = dat["data_vars"][key][idx]
            if sum(nan) > 0:
                dat["data_vars"][key] = dat["data_vars"][key][~nan]

    return dat


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
        self.fname = _abspath(fname)
        print("\nReading file {} ...".format(fname))
        self._debug_level = debug_level
        self._vmdas_search = vmdas_search
        self._winrivprob = winriver
        self._vm_source = 0
        self._pos = 0
        self.progress = 0
        self._cfac = 180 / 2**31
        self._fixoffset = 0
        self._nbyte = 0
        self.n_cells_diff = 0
        self.n_cells_sl = 0
        self.cs_diff = 0
        self.cs = []
        self.cfg = {}
        self.cfgbb = {}
        self.hdr = {}
        self.f = bin_reader(self.fname)

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

        self.ensemble = defs._ensemble(self.n_avg, self.cfg["n_cells"])
        if self._bb:
            self.ensembleBB = defs._ensemble(self.n_avg, self.cfgbb["n_cells"])

        self.vars_read = defs._variable_setlist(["time"])
        if self._bb:
            self.vars_readBB = defs._variable_setlist(["time"])

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
        """
        Scan file until 7f7f is found
        """
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
                self.read_fixed(bb=True)
                found = True
            elif id == 0:
                self.read_fixed(bb=False)
            elif id == 16:
                self.read_fixed_sl()  # bb=True
            elif id == 8192:
                self._vmdas_search = True
        return found

    def load_data(self, nens=None):
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
                        tmlib.datetime(*clock[:6, 0], microsecond=clock[6, 0] * 10000)
                    )[0]
                except ValueError:
                    warnings.warn(
                        "Invalid time stamp in ping {}.".format(
                            int(self.ensemble.number[0])
                        )
                    )
                    dat["coords"]["time"][iens] = np.NaN
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
            outd = defs._idata(
                outd, nm, sz=defs._get_size(nm, self._nens, self.cfg["n_cells"])
            )
        self.outd = outd

        if self._bb:
            for nm in defs.data_defs:
                outdbb = defs._idata(
                    outdbb, nm, sz=defs._get_size(nm, self._nens, self.cfgbb["n_cells"])
                )
            self.outdBB = outdbb
            if self._debug_level > 1:
                logging.info(np.shape(outdbb["data_vars"]["vel"]))

        if self._debug_level > 1:
            logging.info("{} ncells, not BB".format(self.cfg["n_cells"]))
            if self._bb:
                logging.info("{} ncells, BB".format(self.cfgbb["n_cells"]))

    def read_buffer(self):
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

    def print_progress(self):
        self.progress = self.f.tell()
        if self._debug_level > 1:
            logging.debug(
                "  pos %0.0fmb/%0.0fmb\n"
                % (self.f.tell() / 1048576.0, self._filesize / 1048576.0)
            )
        if (self.f.tell() - self.progress) < 1048576:
            return

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
        """
        Returns True if next header is bad or at end of file.
        """
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
        function_map = {
            # 0000 1st profile fixed leader
            0: (self.read_fixed, []),
            # 0001 2nd profile fixed leader
            1: (self.read_fixed, [True]),
            # 0010 Surface layer fixed leader (RiverPro & StreamPro)
            16: (self.read_fixed_sl, []),
            # 0080 1st profile variable leader
            128: (self.read_var, [0]),
            # 0081 2nd profile variable leader
            129: (self.read_var, [1]),
            # 0100 1st profile velocity
            256: (self.read_vel, [0]),
            # 0101 2nd profile velocity
            257: (self.read_vel, [1]),
            # 0103 Waves first leader
            259: (self.skip_Nbyte, [74]),
            # 0110 Surface layer velocity (RiverPro & StreamPro)
            272: (self.read_vel, [2]),
            # 0200 1st profile correlation
            512: (self.read_corr, [0]),
            # 0201 2nd profile correlation
            513: (self.read_corr, [1]),
            # 0203 Waves data
            515: (self.skip_Nbyte, [186]),
            # 020C Ambient sound profile
            524: (self.skip_Nbyte, [4]),
            # 0210 Surface layer correlation (RiverPro & StreamPro)
            528: (self.read_corr, [2]),
            # 0300 1st profile amplitude
            768: (self.read_amp, [0]),
            # 0301 2nd profile amplitude
            769: (self.read_amp, [1]),
            # 0302 Beam 5 Sum of squared velocities
            770: (self.skip_Ncol, []),
            # 0303 Waves last leader
            771: (self.skip_Ncol, [18]),
            # 0310 Surface layer amplitude (RiverPro & StreamPro)
            784: (self.read_amp, [2]),
            # 0400 1st profile % good
            1024: (self.read_prcnt_gd, [0]),
            # 0401 2nd profile pct good
            1025: (self.read_prcnt_gd, [1]),
            # 0403 Waves HPR data
            1027: (self.skip_Nbyte, [6]),
            # 0410 Surface layer pct good (RiverPro & StreamPro)
            1040: (self.read_prcnt_gd, [2]),
            # 0500 1st profile status
            1280: (self.read_status, [0]),
            # 0501 2nd profile status
            1281: (self.read_status, [1]),
            # 0510 Surface layer status (RiverPro & StreamPro)
            1296: (self.read_status, [2]),
            1536: (self.read_bottom, []),  # 0600 bottom tracking
            1793: (self.skip_Ncol, [4]),  # 0701 number of pings
            1794: (self.skip_Ncol, [4]),  # 0702 sum of squared vel
            1795: (self.skip_Ncol, [4]),  # 0703 sum of velocities
            2560: (self.skip_Ncol, []),  # 0A00 Beam 5 velocity
            2816: (self.skip_Ncol, []),  # 0B00 Beam 5 correlation
            3072: (self.skip_Ncol, []),  # 0C00 Beam 5 amplitude
            3328: (self.skip_Ncol, []),  # 0D00 Beam 5 pct_good
            # Fixed attitude data format for Ocean Surveyor ADCPs
            3000: (self.skip_Nbyte, [32]),
            3841: (self.skip_Nbyte, [38]),  # 0F01 Beam 5 leader
            8192: (self.read_vmdas, []),  # 2000
            # 2013 Navigation parameter data
            8211: (self.skip_Nbyte, [83]),
            8226: (self.read_winriver2, []),  # 2022
            8448: (self.read_winriver, [38]),  # 2100
            8449: (self.read_winriver, [97]),  # 2101
            8450: (self.read_winriver, [45]),  # 2102
            8451: (self.read_winriver, [60]),  # 2103
            8452: (self.read_winriver, [38]),  # 2104
            # 3200 Transformation matrix
            12800: (self.skip_Nbyte, [32]),
            # 3000 Fixed attitude data format for Ocean Surveyor ADCPs
            12288: (self.skip_Nbyte, [32]),
            12496: (self.skip_Nbyte, [24]),  # 30D0
            12504: (self.skip_Nbyte, [48]),  # 30D8
            # 4100 beam 5 range
            16640: (self.read_alt, []),
            # 4400 Firmware status data (RiverPro & StreamPro)
            17408: (self.skip_Nbyte, [28]),
            # 4401 Auto mode setup (RiverPro & StreamPro)
            17409: (self.skip_Nbyte, [82]),
            # 5803 High resolution bottom track velocity
            22531: (self.skip_Nbyte, [68]),
            # 5804 Bottom track range
            22532: (self.skip_Nbyte, [21]),
            # 5901 ISM (IMU) data
            22785: (self.skip_Nbyte, [65]),
            # 5902 Ping attitude
            22786: (self.skip_Nbyte, [105]),
            # 7001 ADC data
            28673: (self.skip_Nbyte, [14]),
        }
        # Call the correct function:
        if self._debug_level > 1:
            logging.debug(f"Trying to Read {id}")
        if id in function_map:
            if self._debug_level > 1:
                logging.info("  Reading code {}...".format(hex(id)))
            retval = function_map.get(id)[0](*function_map[id][1])
            if retval:
                return retval
            if self._debug_level > 1:
                logging.info("    success!")
        else:
            self.read_nocode(id)

    def read_fixed(self, bb=False):
        self.read_cfgseg(bb=bb)
        self._nbyte += 2
        if self._debug_level > -1:
            logging.info("Read Fixed")

        # Check if n_cells has increased (for winriver transect files)
        if hasattr(self, "ensemble"):
            self.n_cells_diff = self.cfg["n_cells"] - self.ensemble["n_cells"]
            # Increase n_cells if greater than 0
            if self.n_cells_diff > 0:
                self.ensemble = defs._ensemble(self.n_avg, self.cfg["n_cells"])
                if self._debug_level > 0:
                    logging.warning(
                        f"Maximum number of cells increased to {self.cfg['n_cells']}"
                    )

    def read_fixed_sl(self):
        # Surface layer profile
        cfg = self.cfg
        cfg["surface_layer"] = 1
        n_cells = self.f.read_ui8(1)
        # Check if n_cells is greater than what was used in prior profiles
        if n_cells > self.n_cells_sl:
            self.n_cells_sl = n_cells
            if self._debug_level > 0:
                logging.warning(
                    f"Maximum number of surface layer cells increased to {n_cells}"
                )
        cfg["n_cells_sl"] = n_cells
        # Assuming surface layer profile cell size never changes
        cfg["cell_size_sl"] = self.f.read_ui16(1) * 0.01
        cfg["bin1_dist_m_sl"] = round(self.f.read_ui16(1) * 0.01, 4)

        if self._debug_level > -1:
            logging.info("Read Surface Layer Config")
        self._nbyte = 2 + 5

    def read_cfgseg(self, bb=False):
        cfgstart = self.f.tell()

        if bb:
            cfg = self.cfgbb
        else:
            cfg = self.cfg
        fd = self.f
        tmp = fd.read_ui8(5)
        prog_ver0 = tmp[0]
        cfg["prog_ver"] = tmp[0] + tmp[1] / 100.0
        cfg["inst_model"] = defs.adcp_type.get(tmp[0], "unrecognized firmware version")
        config = tmp[2:4]
        cfg["beam_angle"] = [15, 20, 30][(config[1] & 3)]
        beam5 = [0, 1][int((config[1] & 16) == 16)]
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
            if self._debug_level > 0:
                logging.info(f"Number of cells set to {cfg['n_cells']}")
        cfg["pings_per_ensemble"] = fd.read_ui16(1)
        # Check if cell size has changed
        cs = fd.read_ui16(1) * 0.01
        if ("cell_size" not in cfg) or (cs != cfg["cell_size"]):
            self.cs_diff = cs if "cell_size" not in cfg else (cs - cfg["cell_size"])
            cfg["cell_size"] = cs
            if self._debug_level > 0:
                logging.info(f"Cell size set to {cfg['cell_size']}")
        cfg["blank_dist"] = fd.read_ui16(1) * 0.01
        cfg["profiling_mode"] = fd.read_ui8(1)
        cfg["min_corr_threshold"] = fd.read_ui8(1)
        cfg["n_code_reps"] = fd.read_ui8(1)
        cfg["min_prcnt_gd"] = fd.read_ui8(1)
        cfg["max_error_vel"] = fd.read_ui16(1) / 1000
        cfg["sec_between_ping_groups"] = np.sum(
            np.array(fd.read_ui8(3)) * np.array([60.0, 1.0, 0.01])
        )
        coord_sys = fd.read_ui8(1)
        cfg["coord_sys"] = ["beam", "inst", "ship", "earth"][((coord_sys >> 3) & 3)]
        cfg["use_pitchroll"] = ["no", "yes"][(coord_sys & 4) == 4]
        cfg["use_3beam"] = ["no", "yes"][(coord_sys & 2) == 2]
        cfg["bin_mapping"] = ["no", "yes"][(coord_sys & 1) == 1]
        cfg["heading_misalign_deg"] = fd.read_i16(1) * 0.01
        cfg["magnetic_var_deg"] = fd.read_i16(1) * 0.01
        cfg["sensors_src"] = np.binary_repr(fd.read_ui8(1), 8)
        cfg["sensors_avail"] = np.binary_repr(fd.read_ui8(1), 8)
        cfg["bin1_dist_m"] = round(fd.read_ui16(1) * 0.01, 4)
        cfg["transmit_pulse_m"] = fd.read_ui16(1) * 0.01
        cfg["water_ref_cells"] = list(fd.read_ui8(2))  # list for attrs
        cfg["false_target_threshold"] = fd.read_ui8(1)
        fd.seek(1, 1)
        cfg["transmit_lag_m"] = fd.read_ui16(1) * 0.01
        self._nbyte = 40

        if cfg["prog_ver"] >= 8.14:
            cpu_serialnum = fd.read_ui8(8)
            self._nbyte += 8
        if cfg["prog_ver"] >= 8.24:
            cfg["bandwidth"] = fd.read_ui16(1)
            self._nbyte += 2
        if cfg["prog_ver"] >= 16.05:
            cfg["power_level"] = fd.read_ui8(1)
            self._nbyte += 1
        if cfg["prog_ver"] >= 16.27:
            # cfg['navigator_basefreqindex'] = fd.read_ui8(1)
            fd.seek(1, 1)
            cfg["serialnum"] = fd.read_ui32(1)
            cfg["beam_angle"] = fd.read_ui8(1)
            self._nbyte += 6

        self.configsize = self.f.tell() - cfgstart
        if self._debug_level > -1:
            logging.info("Read Config")

    def read_var(self, bb=False):
        """Read variable leader"""
        fd = self.f
        if bb:
            ens = self.ensembleBB
        else:
            ens = self.ensemble
        ens.k += 1
        ens = self.ensemble
        k = ens.k
        self.vars_read += [
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
        self._nbyte = 2 + 40

        cfg = self.cfg
        if cfg["inst_model"].lower() == "broadband":
            if cfg["prog_ver"] >= 5.55:
                fd.seek(15, 1)
                cent = fd.read_ui8(1)
                ens.rtc[:, k] = fd.read_ui8(7)
                ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
                self._nbyte += 23
        elif cfg["inst_model"].lower() == "ocean surveyor":
            fd.seek(16, 1)  # 30 bytes all set to zero, 14 read above
            self._nbyte += 16
            if cfg["prog_ver"] > 23:
                fd.seek(2, 1)
                self._nbyte += 2
        else:
            ens.error_status[k] = np.binary_repr(fd.read_ui32(1), 32)
            self.vars_read += ["pressure", "pressure_std"]
            self._nbyte += 4
            if cfg["prog_ver"] >= 8.13:
                # Added pressure sensor stuff in 8.13
                fd.seek(2, 1)
                ens.pressure[k] = fd.read_ui32(1) / 1000  # dPa to dbar
                ens.pressure_std[k] = fd.read_ui32(1) / 1000
                self._nbyte += 10
            if cfg["prog_ver"] >= 8.24:
                # Spare byte added 8.24
                fd.seek(1, 1)
                self._nbyte += 1
            if cfg["prog_ver"] >= 16.05:
                # Added more fields with century in clock
                cent = fd.read_ui8(1)
                ens.rtc[:, k] = fd.read_ui8(7)
                ens.rtc[0, k] = ens.rtc[0, k] + cent * 100
                self._nbyte += 8
            if cfg["prog_ver"] >= 56:
                fd.seek(1)  # lag near bottom flag
                self._nbyte += 1

        if self._debug_level > -1:
            logging.info("Read Var")

    def switch_profile(self, bb):
        if bb == 1:
            ens = self.ensembleBB
            cfg = self.cfgbb
            # Placeholder for dual profile mode
            # Solution for vmdas profile in bb spot (vs nb)
            tag = ""
        elif bb == 2:
            ens = self.ensemble
            cfg = self.cfg
            tag = "_sl"
        else:
            ens = self.ensemble
            cfg = self.cfg
            tag = ""

        return ens, cfg, tag

    def read_vel(self, bb=0):
        ens, cfg, tg = self.switch_profile(bb)
        self.vars_read += ["vel" + tg]
        n_cells = cfg["n_cells" + tg]

        k = ens.k
        vel = np.array(self.f.read_i16(4 * n_cells)).reshape((n_cells, 4)) * 0.001
        ens["vel" + tg][:n_cells, :, k] = vel
        self._nbyte = 2 + 4 * n_cells * 2
        if self._debug_level > -1:
            logging.info("Read Vel")

    def read_corr(self, bb=0):
        ens, cfg, tg = self.switch_profile(bb)
        self.vars_read += ["corr" + tg]
        n_cells = cfg["n_cells" + tg]

        k = ens.k
        ens["corr" + tg][:n_cells, :, k] = np.array(
            self.f.read_ui8(4 * n_cells)
        ).reshape((n_cells, 4))
        self._nbyte = 2 + 4 * n_cells
        if self._debug_level > -1:
            logging.info("Read Corr")

    def read_amp(self, bb=0):
        ens, cfg, tg = self.switch_profile(bb)
        self.vars_read += ["amp" + tg]
        n_cells = cfg["n_cells" + tg]

        k = ens.k
        ens["amp" + tg][:n_cells, :, k] = np.array(
            self.f.read_ui8(4 * n_cells)
        ).reshape((n_cells, 4))
        self._nbyte = 2 + 4 * n_cells
        if self._debug_level > -1:
            logging.info("Read Amp")

    def read_prcnt_gd(self, bb=0):
        ens, cfg, tg = self.switch_profile(bb)
        self.vars_read += ["prcnt_gd" + tg]
        n_cells = cfg["n_cells" + tg]

        ens["prcnt_gd" + tg][:n_cells, :, ens.k] = np.array(
            self.f.read_ui8(4 * n_cells)
        ).reshape((n_cells, 4))
        self._nbyte = 2 + 4 * n_cells
        if self._debug_level > -1:
            logging.info("Read PG")

    def read_status(self, bb=0):
        ens, cfg, tg = self.switch_profile(bb)
        self.vars_read += ["status" + tg]
        n_cells = cfg["n_cells" + tg]

        ens["status" + tg][:n_cells, :, ens.k] = np.array(
            self.f.read_ui8(4 * n_cells)
        ).reshape((n_cells, 4))
        self._nbyte = 2 + 4 * n_cells
        if self._debug_level > -1:
            logging.info("Read Status")

    def read_bottom(self):
        self.vars_read += ["dist_bt", "vel_bt", "corr_bt", "amp_bt", "prcnt_gd_bt"]
        fd = self.f
        ens = self.ensemble
        k = ens.k
        cfg = self.cfg
        if self._vm_source == 2:
            self.vars_read += ["latitude_gps", "longitude_gps"]
            fd.seek(2, 1)
            long1 = fd.read_ui16(1)
            fd.seek(6, 1)
            ens.latitude_gps[k] = fd.read_i32(1) * self._cfac
            if ens.latitude_gps[k] == 0:
                ens.latitude_gps[k] = np.NaN
        else:
            fd.seek(14, 1)
        ens.dist_bt[:, k] = fd.read_ui16(4) * 0.01
        ens.vel_bt[:, k] = fd.read_i16(4) * 0.001
        ens.corr_bt[:, k] = fd.read_ui8(4)
        ens.amp_bt[:, k] = fd.read_ui8(4)
        ens.prcnt_gd_bt[:, k] = fd.read_ui8(4)
        if self._vm_source == 2:
            fd.seek(2, 1)
            ens.longitude_gps[k] = (long1 + 65536 * fd.read_ui16(1)) * self._cfac
            if ens.longitude_gps[k] > 180:
                ens.longitude_gps[k] = ens.longitude_gps[k] - 360
            if ens.longitude_gps[k] == 0:
                ens.longitude_gps[k] = np.NaN
            fd.seek(16, 1)
            qual = fd.read_ui8(1)
            if qual == 0:
                if self._debug_level > 0:
                    logging.info(
                        "  qual==%d,%f %f"
                        % (qual, ens.latitude_gps[k], ens.longitude_gps[k])
                    )
                ens.latitude_gps[k] = np.NaN
                ens.longitude_gps[k] = np.NaN
            fd.seek(71 - 45 - 16 - 17, 1)
            self._nbyte = 2 + 68
        else:
            # Skip reference layer data
            fd.seek(26, 1)
            self._nbyte = 2 + 68
        if cfg["prog_ver"] >= 5.3:
            fd.seek(7, 1)  # skip to rangeMsb bytes
            ens.dist_bt[:, k] = ens.dist_bt[:, k] + fd.read_ui8(4) * 655.36
            self._nbyte += 11
        if cfg["prog_ver"] >= 16.2 and (cfg.get("sourceprog") != "WINRIVER"):
            fd.seek(4, 1)  # not documented
            self._nbyte += 4
        if cfg["prog_ver"] >= 56.1:
            fd.seek(4, 1)  # not documented
            self._nbyte += 4

        if self._debug_level > -1:
            logging.info("Read Bottom Track")

    def read_alt(self):
        """Read altimeter (vertical beam range)"""
        fd = self.f
        ens = self.ensemble
        k = ens.k
        self.vars_read += ["alt_dist", "alt_rssi", "alt_eval", "alt_status"]
        ens.alt_eval[k] = fd.read_ui8(1)  # evaluation amplitude
        ens.alt_rssi[k] = fd.read_ui8(1)  # RSSI amplitude
        ens.alt_dist[k] = fd.read_ui32(1) / 1000  # range to surface/seafloor
        ens.alt_status[k] = fd.read_ui8(1)  # status bit flags
        self._nbyte = 7 + 2
        if self._debug_level > -1:
            logging.info("Read Altimeter")

    def read_vmdas(self):
        """Read VMDAS Navigation block"""
        fd = self.f
        self.cfg["sourceprog"] = "VMDAS"
        ens = self.ensemble
        k = ens.k
        if self._vm_source != 1 and self._debug_level > -1:
            logging.info("  \n***** Apparently a VMDAS file \n\n")
        self._vm_source = 1
        self.vars_read += [
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
        utc_time_first_fix = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) / 10)))
        ens.clock_offset_UTC_gps[k] = (
            fd.read_i32(1) / 1000
        )  # "PC clock offset from UTC" in ms
        latitude_first_gps = fd.read_i32(1) * self._cfac
        longitude_first_gps = fd.read_i32(1) * self._cfac

        # Last lat/lon position prior to current ADCP ping
        utc_time_fix = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) / 10)))
        ens.time_gps[k] = tmlib.date2epoch(date_utc + utc_time_fix)[0]
        ens.latitude_gps[k] = fd.read_i32(1) * self._cfac
        ens.longitude_gps[k] = fd.read_i32(1) * self._cfac

        ens.avg_speed_gps[k] = fd.read_ui16(1) / 1000
        ens.avg_dir_gps[k] = fd.read_ui16(1) * 180 / 2**15  # avg true track
        fd.seek(2, 1)  # avg magnetic track
        ens.speed_made_good_gps[k] = fd.read_ui16(1) / 1000
        ens.dir_made_good_gps[k] = fd.read_ui16(1) * 180 / 2**15
        fd.seek(2, 1)  # reserved
        ens.flags_gps[k] = int(np.binary_repr(fd.read_ui16(1)))
        fd.seek(6, 1)  # reserved, ADCP ensemble #

        # ADCP date time
        utim = fd.read_ui8(4)
        date_adcp = tmlib.datetime(utim[0] + utim[1] * 256, utim[3], utim[2])
        time_adcp = tmlib.timedelta(milliseconds=(int(fd.read_ui32(1) / 10)))

        ens.pitch_gps[k] = fd.read_ui16(1) * 180 / 2**15
        ens.roll_gps[k] = fd.read_ui16(1) * 180 / 2**15
        ens.heading_gps[k] = fd.read_ui16(1) * 180 / 2**15

        fd.seek(10, 1)
        self._nbyte = 2 + 76

        if self._debug_level > -1:
            logging.info("Read VMDAS")
        self._read_vmdas = True

    def read_winriver2(self):
        startpos = self.f.tell()
        self._winrivprob = True
        self.cfg["sourceprog"] = "WinRiver2"
        ens = self.ensemble
        k = ens.k
        if self._debug_level > -1:
            logging.info("Read WinRiver2")
        self._vm_source = 3

        spid = self.f.read_ui16(1)  # NMEA specific IDs
        if spid in [4, 104]:  # GGA
            sz = self.f.read_ui16(1)
            dtime = self.f.read_f64(1)
            if sz <= 43:  # If no sentence, data is still stored in nmea format
                empty_gps = self.f.reads(sz - 2)
                self.f.seek(2, 1)
            else:  # TRDI rewrites the nmea string into their format if one is found
                start_string = self.f.reads(6)
                if not isinstance(start_string, str):
                    if self._debug_level > 0:
                        logging.warning(
                            f"Invalid GGA string found in ensemble {k}," " skipping..."
                        )
                    return "FAIL"
                self.f.seek(1, 1)
                gga_time = self.f.reads(9)
                time = tmlib.timedelta(
                    hours=int(gga_time[0:2]),
                    minutes=int(gga_time[2:4]),
                    seconds=int(gga_time[4:6]),
                    milliseconds=int(float(gga_time[6:]) * 1000),
                )
                clock = self.ensemble.rtc[:, :]
                if clock[0, 0] < 100:
                    clock[0, :] += defs.century
                date = tmlib.datetime(*clock[:3, 0]) + time
                ens.time_gps[k] = tmlib.date2epoch(date)[0]
                self.f.seek(1, 1)
                ens.latitude_gps[k] = self.f.read_f64(1)
                tcNS = self.f.reads(1)  # 'N' or 'S'
                if tcNS == "S":
                    ens.latitude_gps[k] *= -1
                ens.longitude_gps[k] = self.f.read_f64(1)
                tcEW = self.f.reads(1)  # 'E' or 'W'
                if tcEW == "W":
                    ens.longitude_gps[k] *= -1
                ens.fix_gps[k] = self.f.read_ui8(1)  # gps fix type/quality
                ens.n_sat_gps[k] = self.f.read_ui8(1)  # of satellites
                # horizontal dilution of precision
                ens.hdop_gps[k] = self.f.read_f32(1)
                ens.elevation_gps[k] = self.f.read_f32(1)  # altitude
                m = self.f.reads(1)  # altitude unit, 'm'
                h_geoid = self.f.read_f32(1)  # height of geoid
                m2 = self.f.reads(1)  # geoid unit, 'm'
                ens.rtk_age_gps[k] = self.f.read_f32(1)
                station_id = self.f.read_ui16(1)
            self.vars_read += [
                "time_gps",
                "longitude_gps",
                "latitude_gps",
                "fix_gps",
                "n_sat_gps",
                "hdop_gps",
                "elevation_gps",
                "rtk_age_gps",
            ]
            self._nbyte = self.f.tell() - startpos + 2

        elif spid in [5, 105]:  # VTG
            sz = self.f.read_ui16(1)
            dtime = self.f.read_f64(1)
            if sz <= 22:  # if no data
                empty_gps = self.f.reads(sz - 2)
                self.f.seek(2, 1)
            else:
                start_string = self.f.reads(6)
                if not isinstance(start_string, str):
                    if self._debug_level > 0:
                        logging.warning(
                            f"Invalid VTG string found in ensemble {k}," " skipping..."
                        )
                    return "FAIL"
                self.f.seek(1, 1)
                true_track = self.f.read_f32(1)
                t = self.f.reads(1)  # 'T'
                magn_track = self.f.read_f32(1)
                m = self.f.reads(1)  # 'M'
                speed_knot = self.f.read_f32(1)
                kts = self.f.reads(1)  # 'N'
                speed_kph = self.f.read_f32(1)
                kph = self.f.reads(1)  # 'K'
                mode = self.f.reads(1)
                # knots -> m/s
                ens.speed_over_grnd_gps[k] = speed_knot / 1.944
                ens.dir_over_grnd_gps[k] = true_track
            self.vars_read += ["speed_over_grnd_gps", "dir_over_grnd_gps"]
            self._nbyte = self.f.tell() - startpos + 2

        elif spid in [6, 106]:  # 'DBT' depth sounder
            sz = self.f.read_ui16(1)
            dtime = self.f.read_f64(1)
            if sz <= 20:
                empty_gps = self.f.reads(sz - 2)
                self.f.seek(2, 1)
            else:
                start_string = self.f.reads(6)
                if not isinstance(start_string, str):
                    if self._debug_level > 0:
                        logging.warning(
                            f"Invalid DBT string found in ensemble {k}," " skipping..."
                        )
                    return "FAIL"
                self.f.seek(1, 1)
                depth_ft = self.f.read_f32(1)
                ft = self.f.reads(1)  # 'f'
                depth_m = self.f.read_f32(1)
                m = self.f.reads(1)  # 'm'
                depth_fathom = self.f.read_f32(1)
                f = self.f.reads(1)  # 'F'
                ens.dist_nmea[k] = depth_m
            self.vars_read += ["dist_nmea"]
            self._nbyte = self.f.tell() - startpos + 2

        elif spid in [7, 107]:  # 'HDT'
            sz = self.f.read_ui16(1)
            dtime = self.f.read_f64(1)
            if sz <= 14:
                empty_gps = self.f.reads(sz - 2)
                self.f.seek(2, 1)
            else:
                start_string = self.f.reads(6)
                if not isinstance(start_string, str):
                    if self._debug_level > 0:
                        logging.warning(
                            f"Invalid HDT string found in ensemble {k}," " skipping..."
                        )
                    return "FAIL"
                self.f.seek(1, 1)
                ens.heading_gps[k] = self.f.read_f64(1)
                tt = self.f.reads(1)
            self.vars_read += ["heading_gps"]
            self._nbyte = self.f.tell() - startpos + 2

    def read_winriver(self, nbt):
        self._winrivprob = True
        self.cfg["sourceprog"] = "WINRIVER"
        if self._vm_source not in [2, 3]:
            if self._debug_level > -1:
                logging.warning(
                    "\n***** Apparently a WINRIVER file - "
                    "Raw NMEA data handler not yet implemented\n"
                )
            self._vm_source = 2
        startpos = self.f.tell()
        sz = self.f.read_ui16(1)
        tmp = self.f.reads(sz - 2)
        self._nbyte = self.f.tell() - startpos + 2

    def skip_Ncol(self, n_skip=1):
        self.f.seek(n_skip * self.cfg["n_cells"], 1)
        self._nbyte = 2 + n_skip * self.cfg["n_cells"]

    def skip_Nbyte(self, n_skip):
        self.f.seek(n_skip, 1)
        self._nbyte = 2 + n_skip

    def read_nocode(self, id):
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
            self.skip_Nbyte(12 * nflds * dfac)
        else:
            if self._debug_level > -1:
                logging.warning("  Unrecognized ID code: %0.4X" % id)
            self.skip_nocode(id)

    def skip_nocode(self, id):
        # Skipping bytes if ID isn't known
        offsets = list(self.id_positions.values())
        idx = np.where(offsets == self.id_positions[id])[0][0]
        byte_len = offsets[idx + 1] - offsets[idx] - 2

        self.skip_Nbyte(byte_len)
        if self._debug_level > -1:
            logging.debug(f"Skipping ID code {id}\n")

    def check_offset(self, offset, readbytes):
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
        dat = self.outd
        if self._debug_level > 0:
            logging.info("  Encountered end of file.  Cleaning up data.")
        for nm in self.vars_read:
            defs._setd(dat, nm, defs._get(dat, nm)[..., :iens])

    def save_profiles(self, dat, nm, en, iens):
        ds = defs._get(dat, nm)
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
                    defs._setd(dat, nm, ds)
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
        defs._setd(dat, nm, ds)

        return dat

    def cleanup(self, dat, cfg):
        # Clean up changing cell size, if necessary
        cs = np.array(self.cs)
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
        cfg["cell_size"] = cell_sizes.max()
        dat["coords"]["range"] = (
            cfg["bin1_dist_m"] + np.arange(cfg["n_cells"]) * cfg["cell_size"]
        )

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
        Reshape the array `arr` to shape (...,n,n_bin).
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
            inds = (np.arange(np.prod(shp[-2:])) * n_bin // int(n_bin)).astype(int)
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
        Remove the attributes from the data that were never loaded.
        """

        for nm in set(defs.data_defs.keys()) - self.vars_read:
            defs._pop(dat, nm)
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
            if len(shp) and shp[0] == "nc" and defs._in_group(dat, nm):
                defs._setd(dat, nm, np.swapaxes(defs._get(dat, nm), 0, 1))

        return dat
