from os.path import abspath, dirname, join, isfile
from pandas.testing import assert_frame_equal
from mhkit import utils
from mhkit import loads
import pandas as pd 
import numpy as np
import unittest
import json

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

class TestLoads(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        loads_data_file = join(datadir, "loads_data_dict.json")
        with open(loads_data_file, 'r') as fp:
            data_dict = json.load(fp)
        # convert dictionaries into dataframes
        data = {
                key: pd.DataFrame(data_dict[key]) 
                for key in data_dict
               }
        self.data = data


        self.fatigue_tower = 3804
        self.fatigue_blade = 1388

    def test_bin_stats(self):
        # create array containg wind speeds to use as bin edges
        b_edges = np.arange(3,26,1)
        # apply function for means
        load_means =self.data['means']
        bin_against = load_means['uWind_80m']
        [b_means, b_means_std] = loads.bin_stats(load_means, bin_against, b_edges)

        assert_frame_equal(self.data['bin_means'],b_means)
        assert_frame_equal(self.data['bin_means_std'],b_means_std)

    def test_damage_equivalent_loads(self):
        loads_data = self.data['loads']
        tower_load = loads_data['TB_ForeAft']
        blade_load = loads_data['BL1_FlapMom']
        DEL_tower = loads.damage_equivalent_load(tower_load, 4,bin_num=100,t=600)
        DEL_blade = loads.damage_equivalent_load(blade_load,10,bin_num=100,t=600)

        err_tower = np.abs((self.fatigue_tower-DEL_tower)/self.fatigue_tower)
        err_blade = np.abs((self.fatigue_blade-DEL_blade)/self.fatigue_tower)

        self.assertTrue((err_tower < 0.05).all())
        self.assertTrue((err_blade < 0.05).all())

    def test_plot_statistics(self):
        savepath = abspath(join(testdir, 'test_scatplotter.png'))

        loads.plot_statistics( self.data['means']['uWind_80m'],
                           self.data['means']['TB_ForeAft'],
                           self.data['maxs']['TB_ForeAft'],
                           self.data['mins']['TB_ForeAft'],
                    vstdev=self.data['std']['TB_ForeAft'],
                    xlabel='Wind Speed [m/s]',
                    ylabel='Tower Base Mom [kNm]',
                    savepath=savepath)
        
        self.assertTrue(isfile(savepath))

    def test_binplotter(self):
        bin_maxs     = self.data['bin_maxs']
        bin_maxs_std = self.data['bin_maxs_std']
        bin_mins     = self.data['bin_mins']
        bin_mins_std = self.data['bin_maxs_std']

        # define some extra input variables
        savepath = abspath(join(testdir, 'test_binplotter.png'))
        bcenters = np.arange(3.5,25.5,step=1)
        variab = 'TB_ForeAft'

        # decleration of inputs to be used in plotting
        bmean = self.data['bin_means'][variab]
        bmax  = self.data['bin_maxs'][variab]
        bmin  = self.data['bin_mins'][variab]
        bstdmean = self.data['bin_means_std'][variab]
        bstdmax = bin_maxs_std[variab]
        bstdmin = bin_mins_std[variab]

        # create plot
        loads.plot_bin_statistics(bcenters,bmean,bmax,bmin,bstdmean,bstdmax,bstdmin,
            xlabel='Wind Speed [m/s]',ylabel=variab,title='Binned Stats',savepath=savepath)

        self.assertTrue(isfile(savepath))

if __name__ == '__main__':
    unittest.main()
