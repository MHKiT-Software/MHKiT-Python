from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from random import seed, randint
import matplotlib.pylab as plt
from datetime import datetime
import xarray.testing as xrt
import mhkit.wave as wave
from io import StringIO
import pandas as pd
import numpy as np
import contextlib
import unittest
import netCDF4
import inspect
import pickle
import time
import json
import sys
import os


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../../examples/data/wave')))


class TestPlotResouceCharacterizations(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        f_name= 'Hm0_Te_46022.json'
        self.Hm0Te = pd.read_json(join(datadir,f_name))
    @classmethod
    def tearDownClass(self):
        pass
    def test_plot_avg_annual_energy_matrix(self):

        filename = abspath(join(testdir, 'avg_annual_scatter_table.png'))
        if isfile(filename):
            os.remove(filename)

        Hm0Te = self.Hm0Te
        Hm0Te.drop(Hm0Te[Hm0Te.Hm0 > 20].index, inplace=True)
        J = np.random.random(len(Hm0Te))*100

        plt.figure()
        fig = wave.graphics.plot_avg_annual_energy_matrix(Hm0Te.Hm0,
            Hm0Te.Te, J, Hm0_bin_size=0.5, Te_bin_size=1)
        plt.savefig(filename, format='png')
        plt.close()

        self.assertTrue(isfile(filename))

    def test_plot_monthly_cumulative_distribution(self):

        filename = abspath(join(testdir, 'monthly_cumulative_distribution.png'))
        if isfile(filename):
            os.remove(filename)

        a = pd.date_range(start='1/1/2010',  periods=10000, freq='h')
        S = pd.Series(np.random.random(len(a)) , index=a)
        ax=wave.graphics.monthly_cumulative_distribution(S)
        plt.savefig(filename, format='png')
        plt.close()

        self.assertTrue(isfile(filename))


if __name__ == '__main__':
    unittest.main()
