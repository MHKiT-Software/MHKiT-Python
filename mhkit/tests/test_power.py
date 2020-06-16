import unittest
from os.path import abspath, dirname, join, isfile
import os
import numpy as np
import pandas as pd
import mhkit.power as power

class TestDevice(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.t = 600
        fs = 1000
        
        self.samples = np.linspace(0, self.t, int(fs*self.t), endpoint=False)
        self.frequency = 60
        self.freq_array = np.ones(len(self.samples))*60
        harmonics_int = np.arange(0,60*50,5)
        self.interharmonic = np.zeros(len(harmonics_int)) #since this is an idealized sin wave, the interharmonics should be zero
        self.harmonics_vals = np.zeros(len(harmonics_int))
        self.harmonics_vals[12]= 1.0  #setting 60th harmonic to amplitude of the signal
        self.harmonic_groups = self.harmonics_vals[0::12] #harmonic groups should be equal to every 12th harmonic in this idealized example
        self.thcd = 0.0 #Since this is an idealized sin wave, there should be no distortion 
        
        self.signal = np.sin(2 * np.pi * self.frequency * self.samples)
        
        self.current_data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        self.voltage_data = [[1,5,9],[2,6,10],[3,7,11],[4,8,12]]

    @classmethod
    def tearDownClass(self):
        pass


    def test_instfreq(self):
        um = pd.Series(self.signal,index = self.samples)
        
        freq = power.characteristics.instantaneous_frequency(um)
        for i in freq.values:
            self.assertAlmostEqual(i[0], self.frequency,1)
        # Should we test than len(freq) == len(um)-1?

    def test_harmonics(self):
        current = pd.Series(self.signal,index = self.samples)
        harmonics = power.quality.harmonics(current,1000,self.frequency)
        for i,j in zip(harmonics.values,self.harmonics_vals):
            
            self.assertAlmostEqual(i[0], j,1)
        hsg = power.quality.harmonic_subgroups(harmonics,self.frequency)
        for i,j in zip(hsg.values,self.harmonic_groups):
            
            self.assertAlmostEqual(i[0], j,1)

        TCHD = power.quality.total_harmonic_current_distortion(hsg,18.8) # had to just put a random rated current in here
        self.assertAlmostEqual(TCHD.values[0],self.thcd)
        #test interharmonics
        ih = power.quality.interharmonics(harmonics,self.frequency)
        for i,j in zip(ih.values,self.interharmonic):
            
            self.assertAlmostEqual(i[0], j,1)


        

    def test_dc_power_DataFrame(self):
        current = pd.DataFrame(self.current_data, columns=['A1', 'A2', 'A3'])
        voltage = pd.DataFrame(self.voltage_data, columns=['V1', 'V2', 'V3'])
        P = power.characteristics.dc_power(voltage, current)
        self.assertEqual(P.sum()['Gross'], (voltage.values * current.values).sum())
        
    def test_dc_power_Series(self):
        current = pd.DataFrame(self.current_data, columns=['A1', 'A2', 'A3'])
        voltage = pd.DataFrame(self.voltage_data, columns=['V1', 'V2', 'V3'])
        P = power.characteristics.dc_power(voltage['V1'], current['A1'])
        self.assertEqual(P.sum()['Gross'], sum( voltage['V1'] * current['A1']))

    def test_ac_power_three_phase(self):
        current = pd.DataFrame(self.current_data, columns=['A1', 'A2', 'A3'])
        voltage = pd.DataFrame(self.voltage_data, columns=['V1', 'V2', 'V3'])
        
        P1 = power.characteristics.ac_power_three_phase( voltage, current, 1, False)
        P1b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, False)
        P2 = power.characteristics.ac_power_three_phase( voltage, current,1, True)
        P2b = power.characteristics.ac_power_three_phase(voltage, current, 0.5, True)
        
        self.assertEqual(P1.sum()[0], 584)
        self.assertEqual(P1b.sum()[0], 584/2)
        self.assertAlmostEqual(P2.sum()[0], 1011.518, 2)
        self.assertAlmostEqual(P2b.sum()[0], 1011.518/2, 2)

if __name__ == '__main__':
    unittest.main() 
        
