import unittest
from os.path import abspath, dirname, isfile, join, normpath, relpath

import pandas as pd

import mhkit.mooring as mooring

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,'..','..','..','examples','data','mooring'))

class TestIOmooring(unittest.TestCase):

    def test_moordyn_out(self):
        fpath = join(datadir, 'FAST.MD.out')
        df = mooring.io.moordyn_out(fpath)
        isinstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()