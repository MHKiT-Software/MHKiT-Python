from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from mhkit import utils
from mhkit import loads
import pandas as pd 
import numpy as np
import unittest
import json

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../examples/data')))

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

    def test_bin_statistics(self):
        # create array containg wind speeds to use as bin edges
        bin_edges = np.arange(3,26,1)
        
        # Apply function to calculate means
        load_means =self.data['means']
        bin_against = load_means['uWind_80m']
        [b_means, b_means_std] = loads.bin_statistics(load_means, bin_against, bin_edges)

        assert_frame_equal(self.data['bin_means'],b_means)
        assert_frame_equal(self.data['bin_means_std'],b_means_std)

    def test_damage_equivalent_loads(self):
        loads_data = self.data['loads']
        tower_load = loads_data['TB_ForeAft']
        blade_load = loads_data['BL1_FlapMom']
        DEL_tower = loads.damage_equivalent_load(tower_load, 4,bin_num=100,data_length=600)
        DEL_blade = loads.damage_equivalent_load(blade_load,10,bin_num=100,data_length=600)

        err_tower = np.abs((self.fatigue_tower-DEL_tower)/self.fatigue_tower)
        err_blade = np.abs((self.fatigue_blade-DEL_blade)/self.fatigue_tower)

        self.assertTrue((err_tower < 0.05).all())
        self.assertTrue((err_blade < 0.05).all())

    def test_plot_statistics(self):
        # Define path
        savepath = abspath(join(testdir, 'test_scatplotter.png'))
        
        # Generate plot
        loads.plot_statistics( self.data['means']['uWind_80m'],
                           self.data['means']['TB_ForeAft'],
                           self.data['maxs']['TB_ForeAft'],
                           self.data['mins']['TB_ForeAft'],
                    y_stdev=self.data['std']['TB_ForeAft'],
                    xlabel='Wind Speed [m/s]',
                    ylabel='Tower Base Mom [kNm]',
                    savepath=savepath)
        
        self.assertTrue(isfile(savepath))

    def test_plot_bin_statistics(self):
        # Define signal name, path, and bin centers
        savepath = abspath(join(testdir, 'test_binplotter.png'))
        bin_centers = np.arange(3.5,25.5,step=1)
        signal_name = 'TB_ForeAft'

        # Specify inputs to be used in plotting
        bin_mean = self.data['bin_means'][signal_name]
        bin_max  = self.data['bin_maxs'][signal_name]
        bin_min  = self.data['bin_mins'][signal_name]
        bin_mean_std = self.data['bin_means_std'][signal_name]
        bin_max_std = self.data['bin_maxs_std'][signal_name]
        bin_min_std = self.data['bin_mins_std'][signal_name]

        # Generate plot
        loads.plot_bin_statistics(bin_centers,bin_mean,bin_max,bin_min,bin_mean_std,bin_max_std,bin_min_std,
            xlabel='Wind Speed [m/s]',ylabel=signal_name,title='Binned Stats',savepath=savepath)

        self.assertTrue(isfile(savepath))

if __name__ == '__main__':
    unittest.main()
