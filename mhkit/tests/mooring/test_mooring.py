import unittest
from os.path import abspath, dirname, isfile, join, normpath, relpath

import xarray as xr

import mhkit.mooring as mooring

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,'..','..','..','examples','data','mooring'))

class TestIOmooring(unittest.TestCase):

    def test_moordyn_out(self):
        fpath = join(datadir, 'Test.MD.out')
        inputpath = join(datadir, "testInput.MD.dat")
        ds = mooring.io.moordyn(fpath, input_file=inputpath)
        isinstance(ds, xr.Dataset)

if __name__ == '__main__':
    unittest.main()