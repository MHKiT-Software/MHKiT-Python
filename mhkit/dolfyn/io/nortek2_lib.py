import struct
import os.path as path
import numpy as np
from logging import getLogger
import warnings
from .. import time
from .base import _abspath


def _reduce_by_average(data, ky0, ky1):
    # Average two arrays together, if they both exist.
    if ky1 in data:
        tmp = data.pop(ky1)
        if ky0 in data:
            data[ky0] += tmp
            data[ky0] = data[ky0] / 2
        else:
            data[ky0] = tmp


def _reduce_by_average_angle(data, ky0, ky1, degrees=True):
    # Average two arrays of angles together, if they both exist.
    if degrees:
        rad_fact = np.pi / 180
    else:
        rad_fact = 1
    if ky1 in data:
        if ky0 in data:
            data[ky0] = (
                np.angle(
                    np.exp(1j * data.pop(ky0) * rad_fact)
                    + np.exp(1j * data.pop(ky1) * rad_fact)
                )
                / rad_fact
            )
        else:
            data[ky0] = data.pop(ky1)


# This is the data-type of the index file.
# This must match what is written-out by the create_index function.
_index_version = 1
_index_dtype = {
    None: np.dtype(
        [
            ("ens", np.uint64),
            ("pos", np.uint64),
            ("ID", np.uint16),
            ("config", np.uint16),
            ("beams_cy", np.uint16),
            ("_blank", np.uint16),
            ("year", np.uint8),
            ("month", np.uint8),
            ("day", np.uint8),
            ("hour", np.uint8),
            ("minute", np.uint8),
            ("second", np.uint8),
            ("usec100", np.uint16),
        ]
    ),
    1: np.dtype(
        [
            ("ens", np.uint64),
            ("hw_ens", np.uint32),
            ("pos", np.uint64),
            ("ID", np.uint16),
            ("config", np.uint16),
            ("beams_cy", np.uint16),
            ("_blank", np.uint16),
            ("year", np.uint8),
            ("month", np.uint8),
            ("day", np.uint8),
            ("hour", np.uint8),
            ("minute", np.uint8),
            ("second", np.uint8),
            ("usec100", np.uint16),
            ("d_ver", np.uint8),
        ]
    ),
}


def _calc_time(year, month, day, hour, minute, second, usec, zero_is_bad=True):
    dt = np.zeros(year.shape, dtype="float")
    for idx, (y, mo, d, h, mi, s, u) in enumerate(
        zip(year, month, day, hour, minute, second, usec)
    ):
        if (
            zero_is_bad
            and mo == 0
            and d == 0
            and h == 0
            and mi == 0
            and s == 0
            and u == 0
        ):
            continue
        try:
            # Note that month is zero-based, seconds since Jan 1 1970
            dt[idx] = time.date2epoch(time.datetime(y, mo + 1, d, h, mi, s, u))[0]
        except ValueError:
            # One of the time values is out-of-range (e.g., mi > 60)
            # This probably indicates a corrupted byte, so we just insert None.
            dt[idx] = None
    # None -> NaN in this step
    return dt


def __check_header(infile_obj, eof):
    def find_all(s, c):
        idx = s.find(c)
        while idx != -1:
            yield idx
            idx = s.find(c, idx + 1)

    # Save current position
    current_pos = infile_obj.tell()

    # Close and reopen file with larger buffer
    infile_obj.close()
    int32_max = np.iinfo(np.int32).max
    if eof >= int32_max:
        buffer = int32_max
    else:
        buffer = eof
    fin = open(_abspath(infile_obj.name), "rb", buffer)
    fin.seek(current_pos, 1)

    # Search for multiple saved headers
    pk = fin.peek(1)
    found = [i for i in find_all(pk, b"GETCLOCKSTR")]
    if found:
        start_idx = found[0] - 11  # assuming next header is 10 bytes
    else:
        start_idx = 0

    return fin, start_idx


def _create_index(infile, outfile, init_pos, eof, debug):
    logging = getLogger()
    print("Indexing {}...".format(infile), end="")
    fin = open(_abspath(infile), "rb")
    fout = open(_abspath(outfile), "wb")
    fout.write(b"Index Ver:")
    fout.write(struct.pack("<H", _index_version))
    ids = [21, 22, 23, 24, 26, 28, 27, 29, 30, 31, 35, 36]
    # Saved: burst, avg, bt, vel_b5, alt_raw, echo
    # Not saved: bt record, DVL, alt record, avg alt_raw record, raw echo, raw echo transmit
    ens = dict.fromkeys(ids, 0)
    N = dict.fromkeys(ids, 0)
    config = 0
    last_ens = dict.fromkeys(ids, -1)
    seek_2ens = {
        21: 40,  # 0x15 burst
        22: 40,  # 0x16 average
        23: 42,  # 0x17 bottom track, starts from "42"
        24: 40,  # 0x18 interleaved burst (beam 5)
        26: 40,  # 0x1A burst altimeter
        27: 40,  # 0x1B DVL bottom track
        28: 40,  # 0x1C echo sounder
        29: 40,  # 0x1D DVL water track
        30: 40,  # 0x1E altimeter
        31: 40,  # 0x1F avg altimeter
        35: 40,  # 0x23 raw echo sounder
        36: 40,  # 0x24 raw tx echo sounder
        48: 40,  # 0x30 processed wave
        # 160: 40, # 0xA0 string (header info, GPS NMEA data)
        # 192: 40, # 0xC0 Nortek Data format 8 record
    }
    pos = 0
    header_check_flag = 0
    # leave room for header plus other data (12 + 76)
    while pos <= (eof - 88):
        if init_pos and not pos:
            fin.seek(init_pos, 1)
        try:
            dat = struct.unpack("<BBBBhhh", fin.read(10))
        except Exception as e:
            logging.error(e)
            logging.error("Header is shorter than 10 bytes: %10d\n" % (pos))
            break
        if dat[0] != 165:
            if debug:
                logging.debug("Lost sync byte at pos: %10d\n" % (pos))
        if dat[1] != 10:
            if dat[1] == 12:
                fin.seek(-10, 1)
                dat = struct.unpack("<BBBBIhh", fin.read(12))
            else:
                # Lost header
                if debug:
                    logging.debug("Header is not 10 or 12 bytes: %10d\n" % (pos))
                # check for if file was terminated and restarted
                # by searching for start of header configuration string
                if not header_check_flag:
                    fin, to_skip = __check_header(fin, eof)
                    # Will run long for large files
                    if to_skip:
                        if debug:
                            logging.info(
                                "Found new header configuration string as position: %10d\n"
                                % (pos + to_skip)
                            )
                        fin.seek(to_skip, 1)
                        continue
                    else:
                        # Don't run more often than necessary
                        header_check_flag = 1
                        if debug:
                            logging.info("No new header configuration string found")
        if dat[2] in ids:
            idk = dat[2]
            # version, byte offset to actual data, configuration bit mask
            d_ver, d_off, config = struct.unpack("<BBH", fin.read(4))
            if d_ver not in [1, 3, 7]:
                # data record definition version number (1=100 apparently)
                # increases occasionally with instrument firmware updates
                continue
            fin.seek(4, 1)  # instrument serial number
            yr, mo, dy, h, m, s, u = struct.unpack("6BH", fin.read(8))
            fin.seek(14, 1)  # soundspeed, temp, pres, h,p,r
            beams_cy = struct.unpack("<H", fin.read(2))[0]  # cell_size
            fin.seek(seek_2ens[dat[2]], 1)
            # should end at byte 74 or 76
            ens[idk] = struct.unpack("<I", fin.read(4))[0]

            if last_ens[idk] > 0:
                if (ens[idk] == 1) or (ens[idk] < last_ens[idk]):
                    # Covers all id keys saved in "burst mode"
                    # Covers ID keys not saved in sequential order
                    ens[idk] = last_ens[idk] + 1

            if last_ens[idk] > 0 and last_ens[idk] != ens[idk]:
                N[idk] += 1

            fout.write(
                struct.pack(
                    "<QIQ4H6BHB",
                    N[idk],
                    ens[idk],
                    pos,
                    idk,
                    config,
                    beams_cy,
                    0,
                    yr,
                    mo + 1,
                    dy,
                    h,
                    m,
                    s,
                    u,
                    d_ver,
                )
            )
            fin.seek(dat[4] - (36 + seek_2ens[idk]), 1)
            last_ens[idk] = ens[idk]

            if debug:
                # File Position: Valid ID keys (1A, 10), Hex ID, Length in bytes, Ensemble #, Last Ensemble Found'
                # hex: [18, 15, 1C, 17] = [vel_b5, vel, echo, bt]
                logging.info(
                    "%10d: %02X, %d, %02X, %d, %d, %d, %d\n"
                    % (
                        pos,
                        dat[0],
                        dat[1],
                        dat[2],
                        dat[4],
                        N[idk],
                        ens[idk],
                        last_ens[idk],
                    )
                )
        else:
            if dat[4] < 0:
                if debug:
                    logging.info("Invalid skip byte at pos: %10d\n" % (pos))
                break
            fin.seek(dat[4], 1)
        # Update for while loop check
        pos = fin.tell()

    fin.close()
    fout.close()
    print(" Done.")


def _check_index(idx, infile, fix_hw_ens=False, dp=False):
    logging = getLogger()
    uid = np.unique(idx["ID"])
    if fix_hw_ens:
        hwe = idx["hw_ens"]
    else:
        hwe = idx["hw_ens"].copy()
    ens = idx["ens"]
    N_id = len(uid)

    # Are there better ways to detect dual profile?
    if (22 in uid) and ((21 in uid) or (28 in uid)):
        msg = "Dual Profile detected... Two datasets will be returned."
        warnings.warn(msg)
        logging.warning(msg)
        dp = True

    # This loop fixes 'skips' inside the file
    for id in uid:
        # These are the indices for this ID
        inds = np.nonzero(idx["ID"] == id)[0]
        # These are bad steps in the indices for this ID
        ibad = np.nonzero(np.diff(inds) > N_id)[0]

        # Check if spacing is equal for dual profiling ADCPs
        if dp:
            skip_size = np.diff(ibad)
            n_skip, count = np.unique(skip_size, return_counts=True)
            # If multiple skips are of the same size, assume okay
            for n, c in zip(n_skip, count):
                if c > 1:
                    skip_size[skip_size == n] = 0
            # assume last "ibad" element is always good for dp's
            mask = np.append(skip_size, 0).astype(bool) if any(skip_size) else []
            ibad = ibad[mask]
        for ib in ibad:
            # The ping number reported here may not be quite right if
            # the ensemble count is wrong.
            logging.warning(
                "Skipped ping (ID: {}) in file {} at ensemble {}.".format(
                    id, infile, idx["ens"][inds[ib + 1] - 1]
                )
            )
            hwe[inds[(ib + 1) :]] += 1
            ens[inds[(ib + 1) :]] += 1

    return dp


def _boolarray_firstensemble_ping(index):
    """
    Return a boolean of the index that indicates only the first ping in
    each ensemble.
    """
    dens = np.ones(index["ens"].shape, dtype="bool")
    if any(index["ens"]):
        dens[1:] = np.diff(index["ens"]) != 0
    return dens


def get_index(infile, pos=0, eof=2**32, rebuild=False, debug=False, dp=False):
    """
    This function reads ad2cp.index files

    Parameters
    ----------
    infile: str
      Path and filename of ad2cp datafile, not including ".index"
    reload: bool
      If true, ignore existing .index file and create a new one
    debug: bool
      If true, run code in debug mode

    Returns
    -------
    out: tuple
      Tuple containing info held within index file
    """

    index_file = infile + ".index"
    if not path.isfile(index_file) or rebuild or debug:
        _create_index(infile, index_file, pos, eof, debug)
    f = open(_abspath(index_file), "rb")
    file_head = f.read(12)
    if file_head[:10] == b"Index Ver:":
        index_ver = struct.unpack("<H", file_head[10:])[0]
    else:
        # This is pre-versioning the index files
        index_ver = None
        f.seek(0, 0)
    out = np.fromfile(f, dtype=_index_dtype[index_ver])
    f.close()
    dp = _check_index(out, infile, dp=dp)
    return out, dp


def crop_ensembles(infile, outfile, range):
    """
    This function is for cropping certain pings out of an AD2CP
    file to create a new AD2CP file. It properly grabs the header from
    infile.

    The range is the `ensemble/ping` counter as defined in the first column
    of the INDEX.

    Parameters
    ----------
    infile: str
      Path of ad2cp filename (with .ad2cp file extension)
    outfile: str
      Path for new, cropped ad2cp file (with .ad2cp file extension)
    range: list
      2 element list of start and end ensemble (or time index)
    """

    idx, dp = get_index(infile)
    with open(_abspath(infile), "rb") as fin:
        with open(_abspath(outfile), "wb") as fout:
            fout.write(fin.read(idx["pos"][0]))
            i0 = np.nonzero(idx["ens"] == range[0])[0][0]
            ie = np.nonzero(idx["ens"] == range[1])[0][0]
            pos = idx["pos"][i0]
            nbyte = idx["pos"][ie] - pos
            fin.seek(pos, 0)
            fout.write(fin.read(nbyte))


class _BitIndexer:
    def __init__(self, data):
        self.data = data

    @property
    def _data_is_array(
        self,
    ):
        return isinstance(self.data, np.ndarray)

    @property
    def nbits(
        self,
    ):
        if self._data_is_array:
            return self.data.dtype.itemsize * 8
        else:
            raise ValueError(
                "You must specify the end-range " "for non-ndarray input data."
            )

    def _get_out_type(self, mask):
        # The mask indicates how big this item is.
        if not self._data_is_array:
            return None
        if mask < 2:
            return bool
        if mask < 2**8:
            return np.uint8
        elif mask < 2**16:
            return np.uint16
        elif mask < 2**32:
            return np.uint32
        else:
            return np.uint64

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)
        if slc.step not in [1, None]:
            raise ValueError("Slice syntax for `_getbits` does " "not support steps")
        start = slc.start
        stop = slc.stop
        if start is None:
            start = 0
        if stop is None:
            stop = self.nbits
        mask = 2 ** (stop - start) - 1
        out = (self.data >> start) & mask
        ot = self._get_out_type(mask)
        if ot is not None:
            out = out.astype(ot)
        return out


def _getbit(val, n):
    return bool((val >> n) & 1)


def _headconfig_int2dict(val, mode="burst"):
    """
    Convert the burst Configuration bit-mask to a dict of bools.

    mode: {'burst', 'bt'}
       For 'burst' configs, or 'bottom-track' configs.
    """

    if (mode == "burst") or (mode == "avg"):
        return dict(
            press_valid=_getbit(val, 0),
            temp_valid=_getbit(val, 1),
            compass_valid=_getbit(val, 2),
            tilt_valid=_getbit(val, 3),
            # bit 4 is unused
            vel=_getbit(val, 5),
            amp=_getbit(val, 6),
            corr=_getbit(val, 7),
            le=_getbit(val, 8),
            altraw=_getbit(val, 9),
            ast=_getbit(val, 10),
            echo=_getbit(val, 11),
            ahrs=_getbit(val, 12),
            p_gd=_getbit(val, 13),
            std=_getbit(val, 14),
            # bit 15 is unused
        )
    elif mode == "bt":
        return dict(
            press_valid=_getbit(val, 0),
            temp_valid=_getbit(val, 1),
            compass_valid=_getbit(val, 2),
            tilt_valid=_getbit(val, 3),
            # bit 4 is unused
            vel=_getbit(val, 5),
            # bits 6-7 unused
            dist=_getbit(val, 8),
            fom=_getbit(val, 9),
            ahrs=_getbit(val, 10),
            # bits 10-15 unused
        )


def _status02data(val):
    # This is detailed in the 6.1.2 of the Nortek Signature
    # Integrators Guide (2017)
    bi = _BitIndexer(val)
    out = {}
    if any(bi[15]):  # 'status0_in_use'
        out["proc_idle_less_3pct"] = bi[0]
        out["proc_idle_less_6pct"] = bi[1]
        out["proc_idle_less_12pct"] = bi[2]

    return out


def _status2data(val):
    # This is detailed in the 6.1.2 of the Nortek Signature
    # Integrators Guide (2017)
    bi = _BitIndexer(val)
    out = {}
    out["wakeup_state"] = bi[28:32]
    out["orient_up"] = bi[25:28]
    out["auto_orientation"] = bi[22:25]
    out["previous_wakeup_state"] = bi[18:22]
    out["low_volt_skip"] = bi[17]
    out["active_config"] = bi[16]
    out["echo_index"] = bi[12:16]
    out["telemetry_data"] = bi[11]
    out["boost_running"] = bi[10]
    out["echo_freq_bin"] = bi[5:10]
    # 2,3,4 unused
    out["bd_scaling"] = bi[1]  # if True: cm scaling of blanking dist
    # 0 unused
    return out


def _alt_status2data(val):
    # This is detailed in the 6.1.2 of the Nortek Signature
    # Integrators Guide (2017)
    bi = _BitIndexer(val)
    out = {}
    out["tilt_over_5deg"] = bi[0]
    out["tilt_over_10deg"] = bi[1]
    out["multibeam_alt"] = bi[2]
    out["n_beams_alt"] = bi[3:7]
    out["power_level_idx_alt"] = bi[7:10]

    return out


def _beams_cy_int2dict(val, id):
    """Convert the beams/coordinate-system bytes to a dict of values."""
    if id == 28:  # 0x1C (echosounder)
        return dict(n_cells=val)
    elif id in [26, 31]:
        return dict(n_cells=val & (2**10 - 1), cy="beam", n_beams=1)
    return dict(
        n_cells=val & (2**10 - 1),
        cy=["ENU", "XYZ", "beam", None][val >> 10 & 3],
        n_beams=val >> 12,
    )


def _isuniform(vec, exclude=[]):
    if len(exclude):
        return len(set(np.unique(vec)) - set(exclude)) <= 1
    return np.all(vec == vec[0])


def _collapse(vec, name=None, exclude=[]):
    """
    Check that the input vector is uniform, then collapse it to a
    single value, otherwise raise a warning.
    """

    if _isuniform(vec):
        return vec[0]
    elif _isuniform(vec, exclude=exclude):
        return list(set(np.unique(vec)) - set(exclude))[0]
    else:
        uniq, idx, counts = np.unique(vec, return_index=True, return_counts=True)

        if all(e == counts[0] for e in counts):
            val = max(vec)  # pings saved out of order, but equal # of pings
        else:
            val = vec[idx[np.argmax(counts)]]

        if not set(uniq) == set([0, val]) and set(counts) == set([1, np.max(counts)]):
            # warn when the 'wrong value' is not just a single zero.
            warnings.warn(
                "The variable {} is expected to be uniform, but it is not.\n"
                "Values found: {} (counts: {}).\n"
                "Using the most common value: {}".format(
                    name, list(uniq), list(counts), val
                )
            )

        return val


def _calc_config(index):
    """
    Calculate the configuration information (e.g., number of pings,
    number of beams, struct types, etc.) from the index data.

    Returns
    =======
    config : dict
        A dict containing the key information for initializing arrays.
    """

    ids = np.unique(index["ID"])
    config = {}
    for id in ids:
        if id not in [21, 22, 23, 24, 26, 28, 31]:
            continue
        if id == 23:
            type = "bt"
        elif (id == 22) or (id == 31):
            type = "avg"
        else:
            type = "burst"
        inds = index["ID"] == id
        _config = index["config"][inds]
        _beams_cy = index["beams_cy"][inds]

        # Check that these variables are consistent
        if not _isuniform(_config):
            raise Exception("config are not identical for id: 0x{:X}.".format(id))
        if not _isuniform(_beams_cy):
            err = True
            if id == 23:
                # change in "n_cells" doesn't matter
                lob = np.unique(_beams_cy)
                beams = list(map(_beams_cy_int2dict, lob, 23 * np.ones(lob.size)))
                if all([d["cy"] for d in beams]) and all([d["n_beams"] for d in beams]):
                    err = False
            if err:
                raise Exception("beams_cy are not identical for id: 0x{:X}.".format(id))

        # Now that we've confirmed they are the same:
        config[id] = _headconfig_int2dict(_config[0], mode=type)
        config[id].update(_beams_cy_int2dict(_beams_cy[0], id))
        config[id]["_config"] = _config[0]
        config[id]["_beams_cy"] = _beams_cy[0]
        config[id]["type"] = type
        config[id].pop("cy", None)

    return config
