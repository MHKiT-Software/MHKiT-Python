import unittest
from os.path import abspath, dirname, join
import numpy as np
import pandas as pd
import mhkit.river as river

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

class TestDevice(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.diameter = 1
        self.height = 2
        self.width = 3
        self.diameters = [1,2,3,4]

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_circular(self):
        eq, ca = river.device.circular(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)

    def test_ducted(self):
        eq, ca =river.device.ducted(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)
    
    def test_rectangular(self):
        eq, ca = river.device.rectangular(self.height, self.width)
        self.assertAlmostEqual(eq, 2.76, places=2)
        self.assertAlmostEqual(ca, self.height*self.width, places=2)

    def test_multiple_circular(self):
        eq, ca = river.device.multiple_circular(self.diameters)
        self.assertAlmostEqual(eq, 5.48, places=2)
        self.assertAlmostEqual(ca, 23.56, places=2)

class TestResource(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.data = pd.read_csv(join(datadir, 'tanana_discharge_data.csv'), index_col=0, 
                           parse_dates=True)
        self.data.columns = ['Q']
        
        self.DV_curve = pd.read_csv(join(datadir, 'tanana_DV_curve.csv'))
        self.DV_curve.columns = ['D', 'V']
        
        self.VP_curve = pd.read_csv(join(datadir, 'tanana_VP_curve.csv'))
        self.VP_curve.columns = ['V', 'P']
        
        self.results = pd.read_csv(join(datadir, 'tanana_test_results.csv'), index_col=0, 
                              parse_dates=True)

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_exceedance_probability(self):
        self.results['f'] = river.resource.exceedance_probability(self.data.Q)
        self.assertAlmostEqual((self.results['f'] - self.results['F_control']).sum(), 0.00, places=2 )

    def test_discharge_to_velocity(self):
        p, r2 = river.resource.polynomial_fit(self.DV_curve['D'], self.DV_curve['V'],3)        
        self.results['V'] = river.resource.discharge_to_velocity(self.data.Q, p)
        self.assertAlmostEqual((self.results['V'] - self.results['V_control']).sum(), 0.00, places=2 )
        
    def test_velocity_to_power(self):
        p, r2 = river.resource.polynomial_fit(self.DV_curve['D'], self.DV_curve['V'],3)        
        self.results['V'] = river.resource.discharge_to_velocity(self.data.Q, p)
        p2, r22 = river.resource.polynomial_fit(self.VP_curve['V'], self.VP_curve['P'],2)
        cut_in  = self.DV_curve['V'].min()
        cut_out = self.DV_curve['V'].max()
        self.results['P'] = river.resource.velocity_to_power(self.results['V'], p2,cut_in, cut_out)
        self.assertAlmostEqual((self.results['P'] - self.results['P_control']).sum(), 0.00, places=2 )


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass
        
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_load_usgs_data_instantaneous(self):
        file_name = join(datadir, 'USGS_08313000_Jan2019_instantaneous.json')
        data = river.io.read_usgs_file(file_name)
        
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual(data.shape, (2972, 1)) # 4 data points are missing
        
    def test_load_usgs_data_daily(self):
        file_name = join(datadir, 'USGS_08313000_Jan2019_daily.json')
        data = river.io.read_usgs_file(file_name)

        expected_index = pd.date_range('2019-01-01', '2019-01-31', freq='D')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual((data.index == expected_index.tz_localize('UTC')).all(), True)
        self.assertEqual(data.shape, (31, 1))

if __name__ == '__main__':
    unittest.main() 

