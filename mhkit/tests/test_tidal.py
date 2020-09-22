import unittest
from os.path import abspath, dirname, join, isfile, normpath, relpath
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mhkit.tidal as tidal

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../examples/data/tidal')))


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass
        
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_load_noaa_data(self):
        file_name = join(datadir, 's08010.json')
        data, metadata = tidal.io.read_noaa_json(file_name)
        self.assertTrue(np.all(data.columns == ['s','d','b']) )
        self.assertEqual(data.shape, (18890, 3))

    def test_request_noaa_data(self):
        data, metadata = tidal.io.request_noaa_data(station='s08010', parameter='currents',
                                       start_date='20180101', end_date='20180102',
                                       proxy=None, write_json=None)
        self.assertTrue(np.all(data.columns == ['s','d','b']) )
        self.assertEqual(data.shape, (92, 3))
        

class TestResource(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        file_name = join(datadir, 's08010.json')
        self.data, self.metadata = tidal.io.read_noaa_json(file_name)
        self.data.s = self.data.s / 100. # convert to m/s


    @classmethod
    def tearDownClass(self):
        pass
    
    def test_exceedance_probability(self):
        df = pd.DataFrame.from_records( {'vals': np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9])} )
        df['F'] = tidal.resource.exceedance_probability(df.vals)
        self.assertEqual(df['F'].min(), 10)
        self.assertEqual(df['F'].max(), 90)


    def test_principal_flow_directions(self):    
        width_direction=10
        direction1, direction2 = tidal.resource.principal_flow_directions(self.data.d, width_direction)
        self.assertEqual(direction1,172.0) 
        self.assertEqual(round(direction2,1),round(352.3,1))                                                                                   
    
    def test_plot_current_timeseries(self):
        filename = abspath(join(testdir, 'tidal_plot_current_timeseries.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        tidal.graphics.plot_current_timeseries(self.data.d, self.data.s, 172)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        
    def test_plot_joint_probability_distribution(self):
        filename = abspath(join(testdir, 'tidal_plot_joint_probability_distribution.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        tidal.graphics.plot_joint_probability_distribution(self.data.d, self.data.s, 1, 0.1)
        plt.savefig(f'{filename}')
        plt.close()
        
        self.assertTrue(isfile(filename))
    
    def test_plot_rose(self):
        filename = abspath(join(testdir, 'tidal_plot_rose.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        tidal.graphics.plot_rose(self.data.d, self.data.s, 1, 0.1)
        plt.savefig(f'{filename}')
        plt.close()
        
        self.assertTrue(isfile(filename))
    

class TestPerformance(unittest.TestCase):

    def test_tip_speed_ratio(self):
        rotor_speed = [15,16,17,18] # create array of rotor speeds
        rotor_diameter = 77 # diameter of rotor for GE 1.5
        inflow_speed = [13,13,13,13] # array of wind speeds
        TSR_answer = [4.7,5.0,5.3,5.6]
        
        TSR = tidal.performance.tip_speed_ratio(np.asarray(rotor_speed)/60,rotor_diameter,inflow_speed)

        for i,j in zip(TSR,TSR_answer):
            self.assertAlmostEqual(i,j,delta=0.05)

    def test_power_coefficient(self):
        # data obtained from power performance report of wind turbine
        inflow_speed = [4,6,8,10,12,14,16,18,20]
        power_out = np.asarray([59,304,742,1200,1400,1482,1497,1497,1511])
        capture_area = 4656.63
        rho = 1.225
        Cp_answer = [0.320,0.493,0.508,0.421,0.284,0.189,0.128,0.090,0.066]
        
        Cp = tidal.performance.power_coefficient(power_out*1000,inflow_speed,capture_area,rho)

        for i,j in zip(Cp,Cp_answer):
            self.assertAlmostEqual(i,j,places=2)


if __name__ == '__main__':
    unittest.main() 

