from os.path import abspath, dirname, join, isfile, normpath, relpath
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
import scipy.interpolate as interp
import matplotlib.pylab as plt
import mhkit.river as river
import pandas as pd
import numpy as np
import unittest
import netCDF4
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, 'plots')
isdir = os.path.isdir(plotdir)
if not isdir: os.mkdir(plotdir)
datadir = normpath(join(testdir,'..','..','..','examples','data','river'))


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
        filename = abspath(join(plotdir, 'river_plot_flow_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
            
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.data['Q'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        

    def test_plot_power_duration_curve(self):
        filename = abspath(join(plotdir, 'river_plot_power_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
        
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_flow_duration_curve(self.results['P_control'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        

    def test_plot_velocity_duration_curve(self):
        filename = abspath(join(plotdir, 'river_plot_velocity_duration_curve.png'))
        if isfile(filename):
            os.remove(filename)
        
        f = river.resource.exceedance_probability(self.data.Q)
        plt.figure()
        river.graphics.plot_velocity_duration_curve(self.results['V_control'], f['F'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    

    def test_plot_discharge_timeseries(self):
        filename = abspath(join(plotdir, 'river_plot_discharge_timeseries.png'))
        if isfile(filename): os.remove(filename)
        
        plt.figure()
        river.graphics.plot_discharge_timeseries(self.data['Q'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        

    def test_plot_discharge_vs_velocity(self):
        filename = abspath(join(plotdir, 'river_plot_discharge_vs_velocity.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        river.graphics.plot_discharge_vs_velocity(self.data['Q'], self.results['V_control'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    

    def test_plot_velocity_vs_power(self):
        filename = abspath(join(plotdir, 'river_plot_velocity_vs_power.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        river.graphics.plot_velocity_vs_power(self.results['V_control'], self.results['P_control'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
        

    
if __name__ == '__main__':
    unittest.main() 

