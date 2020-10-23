import unittest
from os.path import abspath, dirname, join, isfile, normpath, relpath
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mhkit.river as river

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../examples/data/river')))


class TestPerformance(unittest.TestCase):
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
        eq, ca = river.performance.circular(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)

    def test_ducted(self):
        eq, ca =river.performance.ducted(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)
    
    def test_rectangular(self):
        eq, ca = river.performance.rectangular(self.height, self.width)
        self.assertAlmostEqual(eq, 2.76, places=2)
        self.assertAlmostEqual(ca, self.height*self.width, places=2)

    def test_multiple_circular(self):
        eq, ca = river.performance.multiple_circular(self.diameters)
        self.assertAlmostEqual(eq, 5.48, places=2)
        self.assertAlmostEqual(ca, 23.56, places=2)

    def test_tip_speed_ratio(self):
        rotor_speed = [15,16,17,18] # create array of rotor speeds
        rotor_diameter = 77 # diameter of rotor for GE 1.5
        inflow_speed = [13,13,13,13] # array of wind speeds
        TSR_answer = [4.7,5.0,5.3,5.6]
        
        TSR = river.performance.tip_speed_ratio(np.asarray(rotor_speed)/60,rotor_diameter,inflow_speed)

        for i,j in zip(TSR,TSR_answer):
            self.assertAlmostEqual(i,j,delta=0.05)

    def test_power_coefficient(self):
        # data obtained from power performance report of wind turbine
        inflow_speed = [4,6,8,10,12,14,16,18,20]
        power_out = np.asarray([59,304,742,1200,1400,1482,1497,1497,1511])
        capture_area = 4656.63
        rho = 1.225
        Cp_answer = [0.320,0.493,0.508,0.421,0.284,0.189,0.128,0.090,0.066]
        
        Cp = river.performance.power_coefficient(power_out*1000,inflow_speed,capture_area,rho)

        for i,j in zip(Cp,Cp_answer):
            self.assertAlmostEqual(i,j,places=2)        

class TestResource(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.data = pd.read_csv(join(datadir, 'tanana_discharge_data.csv'), index_col=0, 
                           parse_dates=True)
        self.data.columns = ['Q']
              
        self.results = pd.read_csv(join(datadir, 'tanana_test_results.csv'), index_col=0, 
                              parse_dates=True)

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_Froude_number(self):
        v = 2
        h = 5
        Fr = river.resource.Froude_number(v, h)
        self.assertAlmostEqual(Fr, 0.286, places=3)
    
    def test_exceedance_probability(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = pd.Series(np.arange(9))
        # Rank order for non-repeating elements simply adds 1 to each element
        #if N=9, max F = 100((max(Q)+1)/10) =  90%
        #if N=9, min F = 100((min(Q)+1)/10) =  10%
        f = river.resource.exceedance_probability(Q)
        self.assertEqual(f.min().values , 10. )
        self.assertEqual(f.max().values , 90. )

    def test_polynomial_fit(self):
        # Calculate a first order polynomial on an x=y line
        p, r2 = river.resource.polynomial_fit(np.arange(8), np.arange(8),1)
        # intercept should be 0
        self.assertAlmostEqual(p[0], 0.0, places=2 )
        # slope should be 1
        self.assertAlmostEqual(p[1], 1.0, places=2 )
        # r-squared should be perfect
        self.assertAlmostEqual(r2, 1.0, places=2 )


    def test_discharge_to_velocity(self):
        # Create arbitrary discharge between 0 and 8(N=9)
        Q = pd.Series(np.arange(9))
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10*np.arange(9),1)
        # Becuase the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(Q, p)
        self.assertAlmostEqual(np.sum(10*Q - V['V']), 0.00, places=2 )
        
    def test_velocity_to_power(self):
        # Calculate a first order polynomial on an DV_Curve x=y line 10 times greater than the Q values
        p, r2 = river.resource.polynomial_fit(np.arange(9), 10*np.arange(9),1)
        # Becuase the polynomial line fits perfect we should expect the V to equal 10*Q
        V = river.resource.discharge_to_velocity(pd.Series(np.arange(9)), p)
        # Calculate a first order polynomial on an VP_Curve x=y line 10 times greater than the V values
        p2, r22 = river.resource.polynomial_fit(np.arange(9), 10*np.arange(9),1)
        # Set cut in/out to exclude 1 bin on either end of V range
        cut_in  = V['V'][1]
        cut_out = V['V'].iloc[-2]  
        # Power should be 10x greater and exclude the ends of V
        P = river.resource.velocity_to_power(V['V'], p2, cut_in, cut_out)
        #Cut in power zero
        self.assertAlmostEqual(P['P'][0], 0.00, places=2 )
        #Cut out power zero
        self.assertAlmostEqual(P['P'].iloc[-1], 0.00, places=2 )
        # Middle 10x greater than velocity
        self.assertAlmostEqual((P['P'][1:-1] - 10*V['V'][1:-1] ).sum(), 0.00, places=2 )


    def test_energy_produced(self):
        # If power is always X then energy produced with be x*seconds 
        X=1
        seconds=1
        P = pd.Series(X*np.ones(10) )
        EP = river.resource.energy_produced(P, seconds)
        self.assertAlmostEqual(EP, X*seconds, places=1 )
        # for a normal distribution of Power EP = mean *seconds
        mu=5
        sigma=1
        power_dist = pd.Series(np.random.normal(mu, sigma, 10000))
        EP2 = river.resource.energy_produced(power_dist, seconds)
#        import ipdb; ipdb.set_trace()
        self.assertAlmostEqual(EP2, mu*seconds, places=1 )


    def test_plot_flow_duration_curve(self):
        filename = abspath(join(testdir, 'river_plot_flow_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
            
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.data['Q'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        
    def test_plot_power_duration_curve(self):
        filename = abspath(join(testdir, 'river_plot_power_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
        
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.results['P_control'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        
    def test_plot_velocity_duration_curve(self):
        filename = abspath(join(testdir, 'river_plot_velocity_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
        
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_velocity_duration_curve(self.results['V_control'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    
    def test_plot_discharge_timeseries(self):
        filename = abspath(join(testdir, 'river_plot_discharge_timeseries.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        river.graphics.plot_discharge_timeseries(self.data['Q'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        
    def test_plot_discharge_vs_velocity(self):
        filename = abspath(join(testdir, 'river_plot_discharge_vs_velocity.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        river.graphics.plot_discharge_vs_velocity(self.data['Q'], self.results['V_control'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    
    def test_plot_velocity_vs_power(self):
        filename = abspath(join(testdir, 'river_plot_velocity_vs_power.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        river.graphics.plot_velocity_vs_power(self.results['V_control'], self.results['P_control'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        

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


    def test_request_usgs_data_daily(self):
        data=river.io.request_usgs_data(station="15515500",
                            parameter='00060',
                            start_date='2009-08-01',
                            end_date='2009-08-10',
                            data_type='Daily')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual(data.shape, (10, 1))
    
   
    def test_request_usgs_data_instant(self):
        data=river.io.request_usgs_data(station="15515500",
                            parameter='00060',
                            start_date='2009-08-01',
                            end_date='2009-08-10',
                            data_type='Instantaneous')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        # Every 15 minutes or 4 times per hour
        self.assertEqual(data.shape, (10*24*4, 1))


if __name__ == '__main__':
    unittest.main() 

