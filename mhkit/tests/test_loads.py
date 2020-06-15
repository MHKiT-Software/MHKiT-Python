# import statements
import unittest
from os.path import abspath, dirname, join, isfile
import pandas as pd 
import numpy as np
from mhkit import utils
from mhkit import loads
from pandas.testing import assert_frame_equal

testdir = dirname(abspath(__file__))

class TestLoads(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        filepath1 = 'data/data_loads.csv'
        self.df = pd.read_csv(filepath1)
        filepath2 = 'data/data_loads_dfmeans.csv'
        self.dfmeans = pd.read_csv(filepath2)
        filepath3 = 'data/data_loads_dfmaxs.csv'
        self.dfmaxs = pd.read_csv(filepath3)
        filepath4 = 'data/data_loads_dfmins.csv'
        self.dfmins = pd.read_csv(filepath4)
        filepath5 = 'data/data_loads_dfstd.csv'
        self.dfstd = pd.read_csv(filepath5)
        self.bin_means = pd.read_csv('data/data_loads_binmeans.csv')
        self.bin_means_std = pd.read_csv('data/data_loads_binmeans_std.csv')
        self.var_dict = [
            ('TB_ForeAft',4),
            ('BL1_FlapMom',10)
        ]
        self.fatigue_tower = 3804
        self.fatigue_blade = 1388

    def test_bin_stats(self):
        # create array containg wind speeds to use as bin edges
        b_edges = np.arange(3,20,1)
        # apply function for means
        [b_means, b_means_std] = loads.bin_stats(self.dfmeans,self.dfmeans['uWind_80m'],b_edges)

        # check that slices are equal
        assert_frame_equal(self.bin_means,b_means)
        assert_frame_equal(self.bin_means_std,b_means_std)

    def test_get_DELs(self):
        DEL = loads.get_DELs(self.df,self.var_dict,binNum=100,t=600)

        err_tower = np.abs((self.fatigue_tower-DEL['TB_ForeAft'])/self.fatigue_tower)
        err_blade = np.abs((self.fatigue_blade-DEL['BL1_FlapMom'])/self.fatigue_tower)

        self.assertLess(err_tower,0.05)
        self.assertLess(err_blade,0.05)

    def test_scatplotter(self):
        savepath = abspath(join(testdir, 'test_scatplotter.png'))
        loads.statplotter(self.dfmeans['uWind_80m'],self.dfmeans['TB_ForeAft'],self.dfmaxs['TB_ForeAft'],self.dfmins['TB_ForeAft'],
            vstdev=self.dfstd['TB_ForeAft'],xlabel='Wind Speed [m/s]',ylabel='Tower Base Mom [kNm]',savepath=savepath)
        
        self.assertTrue(isfile(savepath))

    def test_binplotter(self):
        # load in data
        bin_maxs = pd.read_csv('data/data_loads_binmaxs.csv')
        bin_maxs_std = pd.read_csv('data/data_loads_binmaxs_std.csv')
        bin_mins = pd.read_csv('data/data_loads_binmins.csv')
        bin_mins_std = pd.read_csv('data/data_loads_binmins_std.csv')
        # define some extra input variables
        savepath = abspath(join(testdir, 'test_binplotter.png'))
        bcenters = np.arange(3.5,25.5,step=1)
        variab = 'TB_ForeAft'
        # decleration of inputs to be used in plotting
        bmean = self.bin_means[variab]
        bmax = bin_maxs[variab]
        bmin = bin_mins[variab]
        bstdmean = self.bin_means_std[variab]
        bstdmax = bin_maxs_std[variab]
        bstdmin = bin_mins_std[variab]

        # create plot
        loads.binplotter(bcenters,bmean,bmax,bmin,bstdmean,bstdmax,bstdmin,
            xlabel='Wind Speed [m/s]',ylabel=variab,title='Binned Stats',savepath=savepath)

        self.assertTrue(isfile(savepath))