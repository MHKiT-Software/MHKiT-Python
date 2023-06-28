import unittest
from os.path import abspath, dirname, isfile, join, normpath, relpath
from matplotlib.animation import FuncAnimation

import xarray as xr

import mhkit.mooring as mooring

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,'..','..','..','examples','data','mooring'))

class TestMooring(unittest.TestCase):

    def test_moordyn_out(self):
        fpath = join(datadir, 'Test.MD.out')
        inputpath = join(datadir, "TestInput.MD.dat")
        ds = mooring.io.moordyn(fpath, input_file=inputpath)
        isinstance(ds, xr.Dataset)

    def test_lay_length(self):
        fpath = join(datadir, 'line1_test.nc')
        ds = xr.open_dataset(fpath)
        laylengths = mooring.general.lay_length(ds, depth=-56, tolerance=0.25)
        laylength = laylengths.mean().values
        self.assertAlmostEqual(laylength, 45.0, 1)

    def test_animate_3d(self):
        fpath = join(datadir, 'line1_test.nc')
        ds = xr.open_dataset(fpath)
        dsani = ds.sel(Time=slice(0,10))
        ani = mooring.graphics.animate_3d(dsani, interval=10, repeat=True, 
                                         xlabel='X-axis',ylabel='Y-axis',zlabel='Depth [m]', title='Mooring Line Example')
        isinstance(ani, FuncAnimation)

    def test_animate_2d(self):
        fpath = join(datadir, 'line1_test.nc')
        ds = xr.open_dataset(fpath)
        dsani = ds.sel(Time=slice(0,10))
        ani2d = mooring.graphics.animate_2d(dsani, xaxis='x',yaxis='z', repeat=True, 
                                   xlabel='X-axis',ylabel='Depth [m]', title='Mooring Line Example')
        isinstance(ani2d, FuncAnimation)

if __name__ == '__main__':
    unittest.main()