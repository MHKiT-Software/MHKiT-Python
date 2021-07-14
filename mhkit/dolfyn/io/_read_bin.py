# This file reads binary data files.
# It is mostly for development purposes, to simplify learning a data
# file's format.  For final use, reading binary data files should
# minimize the number of calls to struct.unpack and file.read because
# many calls to these functions (i.e. using the code in this module)
# are slow.
import numpy as np
from struct import unpack
from os.path import expanduser

ics = 0  # This is a holder for the checksum index


class eofException(Exception):
    pass

class CheckSumError(Exception):
    pass

# class checksum(object):
#     # In that case, it *might* be nice to include functionality for nested
#     # checksums, but that seems a bit overkill at this point.
#     def __init__(self, file, val, size, error_behavior='exception'):
#         """
#         Value *val* to initialize the checksum with,
#         and the *size* of the checksum (in bytes, currently this can only be 1,2,4 or 8).
#         """
#         self.file = file
#         self.init(val, size, error_behavior)

#     def init(self, val, size, error_behavior='exception'):
#         self._cs = val
#         self._size = size
#         self._rem = ''
#         self._frmt = {1: 'B', 2: 'H', 4: 'L', 8: 'Q'}[size]
#         self._mask = (2 ** (8 * size) - 1)
#         ## I'd like to add different behavior for the checksum
#         self._error_behavior = error_behavior

#     def error(self, val):
#         if val == 'rem':
#             message = 'A remainder exists in the checksum.'
#         else:
#             message = 'Checksum failed at %d, with a difference of %d.' % (
#                 self.file.tell(), val)
#         if self._error_behavior == 'warning':
#             print('Warning: ' + message)
#         elif self._error_behavior == 'silent':
#             pass
#         else:
#             raise CheckSumError(message)

#     def __call__(self, val, remove_val=False):
#         """
#         Compare the checksum to *val*.
#         *remove_val* specifies whether *val* should be removed from
#          self._cs because it was already added to it.
#         """
#         if self._rem:
#             self.error('rem')
#         retval = (self._cs - val * remove_val & self._mask) - val
#         if retval:
#             self.error(retval)
#         return retval  # returns zero if their is no remainder

#     def add(self, valcs):
#         """
#         Add the data in *valcs* to the checksum.
#         """
#         if self._rem:  # If the cs remainder is not empty:
#             lr = self._rem.__len__()
#             ics = self._size - lr
#             self._rem, valcs = self._rem + valcs[:ics], valcs[ics:]
#             if lr == self._size:
#                 self._cs += unpack(self.file.endian + self._frmt, self._rem)[0]
#                 self._rem = ''
#         if valcs:  # If valcs is not empty:
#             ics = (valcs.__len__() / self._size) * self._size
#             for v in unpack(self.file.endian + self._frmt * (ics / self._size), valcs[:ics]):
#                 self._cs += v
#             self._rem += valcs[ics:]

#     __iadd__ = add


class bin_reader(object):
    #### I may want to write this class in C at some point, to speed things up.
    _size_factor = {'B': 1, 'b': 1, 'H': 2,
                    'h': 2, 'L': 4, 'l': 4, 'f': 4, 'd': 8}
    _frmt = {np.uint8: 'B', np.int8: 'b',
             np.uint16: 'H', np.int16: 'h',
             np.uint32: 'L', np.int32: 'l',
             float: 'f', np.float32: 'f',
             np.double: 'd', np.float64: 'd',
             }

    @property
    def pos(self,):
        return self.f.tell()

    # def __enter__(self,):
    #     return self

    # def __exit__(self, type, value, traceback):
    #     self.close()

    def __init__(self, fname, endian='<', checksum_size=None, debug_level=0):
        """
        Default to little-endian '<'...
        *checksum_size* is in bytes, if it is None or False, this
         function does not perform checksums.
        """
        self.endian = endian
        self.f = open(expanduser(fname), 'rb')
        self.f.seek(0, 2)
        self.fsize = self.tell()
        self.f.seek(0, 0)
        self.close = self.f.close
        #if progbar_size is not None:
        #    self.progbar=progress_bar(self.fsize,progbar_size)
        if checksum_size:
            pass # This is never run?
            #self.cs = checksum(self, 0, checksum_size)
        else:
            self.cs = checksum_size
        self.debug_level = debug_level

    def checksum(self,):
        """
        The next byte(s) are the expected checksum.  Perform the checksum.
        """
        if self.cs:
            cs = self.read(1, self.cs._frmt)
            self.cs(cs, True)
        else:
            raise CheckSumError('CheckSum not requested for this file')

    def tell(self,):
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
            val = val.decode('utf-8')
        except:
            if self.debug_level > 5:
                print("ERROR DECODING: {}".format(val))
            pass
        return val

    def read(self, n, frmt):
        val = self.f.read(n * self._size_factor[frmt])
        if not val:  # If val is empty we are at the end of the file.
            raise eofException
        self.cs and self.cs.add(val)
        if n == 1:
            return unpack(self.endian + frmt * n, val)[0]
        else:
            return np.array(unpack(self.endian + frmt * n, val))

    def read_ui8(self, n):
        return self.read(n, 'B')

    def read_float(self, n):
        return self.read(n, 'f')

    def read_double(self, n):
        return self.read(n, 'd')

    read_f32 = read_float
    read_f64 = read_double

    def read_i8(self, n):
        return self.read(n, 'b')

    def read_ui16(self, n):
        return self.read(n, 'H')

    def read_i16(self, n):
        return self.read(n, 'h')

    def read_ui32(self, n):
        return self.read(n, 'L')

    def read_i32(self, n):
        return self.read(n, 'l')

    # def read_nbytes(self, n):
    #     self.f.read(n)
