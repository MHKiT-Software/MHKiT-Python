from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from mhkit import utils
import mhkit.loads as loads
import pandas as pd 
import numpy as np
import unittest
import json

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../examples/data/loads')))

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

        # import blade cal data
        blade_data = pd.read_csv(join(datadir,'blade_cal.csv'),header=None)
        blade_data.columns = ['flap_raw','edge_raw','flap_scaled','edge_scaled']
        self.blade_data = blade_data
        self.flap_offset = 9.19906E-05
        self.edge_offset = -0.000310854
        self.blade_matrix = [1034671.4,-126487.28,82507.959,1154090.7]

    def test_bin_statistics(self):
        # create array containg wind speeds to use as bin edges
        bin_edges = np.arange(3,26,1)
        
        # Apply function to calculate means
        load_means =self.data['means']
        bin_against = load_means['uWind_80m']
        [b_means, b_means_std] = loads.general.bin_statistics(load_means, bin_against, bin_edges)

        assert_frame_equal(self.data['bin_means'],b_means)
        assert_frame_equal(self.data['bin_means_std'],b_means_std)
  
    def test_blade_moments(self):
        flap_raw = self.blade_data['flap_raw']
        flap_offset = self.flap_offset
        edge_raw = self.blade_data['edge_raw']
        edge_offset = self.edge_offset

        M_flap, M_edge = loads.general.blade_moments(self.blade_matrix,flap_offset,flap_raw,edge_offset,edge_raw)

        for i,j in zip(M_flap,self.blade_data['flap_scaled']):
            self.assertAlmostEqual(i,j,places=1)
        for i,j in zip(M_edge,self.blade_data['edge_scaled']):
            self.assertAlmostEqual(i,j,places=1)


    def test_damage_equivalent_loads(self):
        loads_data = self.data['loads']
        tower_load = loads_data['TB_ForeAft']
        blade_load = loads_data['BL1_FlapMom']
        DEL_tower = loads.general.damage_equivalent_load(tower_load, 4,bin_num=100,data_length=600)
        DEL_blade = loads.general.damage_equivalent_load(blade_load,10,bin_num=100,data_length=600)

        self.assertAlmostEqual(DEL_tower,self.fatigue_tower,delta=self.fatigue_tower*0.04)
        self.assertAlmostEqual(DEL_blade,self.fatigue_blade,delta=self.fatigue_blade*0.04)


    def test_plot_statistics(self):
        # Define path
        savepath = abspath(join(testdir, 'test_scatplotter.png'))
        
        # Generate plot
        loads.graphics.plot_statistics( self.data['means']['uWind_80m'],
                               self.data['means']['TB_ForeAft'],
                               self.data['maxs']['TB_ForeAft'],
                               self.data['mins']['TB_ForeAft'],
                               y_stdev=self.data['std']['TB_ForeAft'],
                               x_label='Wind Speed [m/s]',
                               y_label='Tower Base Mom [kNm]',
                               save_path=savepath)
        
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
        loads.graphics.plot_bin_statistics(bin_centers, 
                                  bin_mean, bin_max, bin_min,
                                  bin_mean_std, bin_max_std, bin_min_std,
                                  x_label='Wind Speed [m/s]',
                                  y_label=signal_name,
                                  title='Binned Stats',
                                  save_path=savepath)

        self.assertTrue(isfile(savepath))


if __name__ == '__main__':
    unittest.main()
