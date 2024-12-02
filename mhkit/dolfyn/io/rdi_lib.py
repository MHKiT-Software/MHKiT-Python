import numpy as np
from struct import unpack
from os.path import expanduser

from .rdi_defs import data_defs


class bin_reader:
    """
    Reads binary data files. It is mostly for development purposes, to
    simplify learning a data file's format. Reading binary data files should
    minimize the number of calls to struct.unpack and file.read because many
    calls to these functions (i.e. using the code in this module) are slow.
    """

    _size_factor = {"B": 1, "b": 1, "H": 2, "h": 2, "L": 4, "l": 4, "f": 4, "d": 8}
    _frmt = {
        np.uint8: "B",
        np.int8: "b",
        np.uint16: "H",
        np.int16: "h",
        np.uint32: "L",
        np.int32: "l",
        float: "f",
        np.float32: "f",
        np.double: "d",
        np.float64: "d",
    }

    @property
    def pos(self):
        return self.f.tell()

    def __init__(self, fname, endian="<", checksum_size=None, debug_level=0):
        """
        Default to little-endian '<'...
        *checksum_size* is in bytes, if it is None or False, this
         function does not perform checksums.
        """
        self.endian = endian
        self.f = open(expanduser(fname), "rb")
        self.f.seek(0, 2)
        self.fsize = self.tell()
        self.f.seek(0, 0)
        self.close = self.f.close
        if checksum_size:
            self.cs = self.checksum(0, checksum_size)
        else:
            self.cs = checksum_size
        self.debug_level = debug_level

    def checksum(self):
        """
        The next byte(s) are the expected checksum.  Perform the checksum.
        """
        if self.cs:
            cs = self.read(1, self.cs._frmt)
            self.cs(cs, True)
        else:
            raise Exception("CheckSum not requested for this file")

    def tell(self):
        return self.f.tell()

    def seek(self, pos, rel=1):
        return self.f.seek(pos, rel)

    def reads(self, n):
        """
        Read a string of n characters.
        """
        val = self.f.read(n)
        self.cs and self.cs.add(val)
        try:
            val = val.decode("utf-8")
        except:
            if self.debug_level > 5:
                print("ERROR DECODING: {}".format(val))
            pass
        return val

    def read(self, n, frmt):
        val = self.f.read(n * self._size_factor[frmt])
        if not val:  # If val is empty we are at the end of the file.
            return None
        self.cs and self.cs.add(val)
        if n == 1:
            return unpack(self.endian + frmt * n, val)[0]
        else:
            return np.array(unpack(self.endian + frmt * n, val))

    def read_ui8(self, n):
        return self.read(n, "B")

    def read_f32(self, n):
        return self.read(n, "f")

    def read_f64(self, n):
        return self.read(n, "d")

    def read_i8(self, n):
        return self.read(n, "b")

    def read_ui16(self, n):
        return self.read(n, "H")

    def read_i16(self, n):
        return self.read(n, "h")

    def read_ui32(self, n):
        return self.read(n, "L")

    def read_i32(self, n):
        return self.read(n, "l")


class _variable_setlist(set):
    def __iadd__(self, vals):
        if vals[0] not in self:
            self |= set(vals)
        return self


class _ensemble:
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
            setattr(
                self,
                nm,
                np.zeros(_get_size(nm, n=navg, ncell=n_cells), dtype=data_defs[nm][2]),
            )

    def clean_data(self):
        self["vel"][self["vel"] == -32.768] = np.nan


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
    if dtype.startswith("float"):
        arr[:] = np.nan
    dat[group][nm] = arr
    dat["units"][nm] = units
    dat["long_name"][nm] = long_name
    if standard_name:
        dat["standard_name"][nm] = standard_name
    return dat


def _get_size(name, n=None, ncell=0):
    sz = list(data_defs[name][0])  # create a copy!
    if "nc" in sz:
        sz.insert(sz.index("nc"), ncell)
        sz.remove("nc")
    if n is None:
        return tuple(sz)
    return tuple(sz + [n])
