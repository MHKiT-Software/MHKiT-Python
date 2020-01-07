import unittest
from os.path import abspath, dirname, join
import numpy as np
import pandas as pd
import mhkit.tidal as tidal

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
        eq, ca = tidal.device.circular(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)

    def test_ducted(self):
        eq, ca =tidal.device.ducted(self.diameter) 
        self.assertEqual(eq, self.diameter)
        self.assertEqual(ca, 4*np.pi*self.diameter**2.)
    
    def test_rectangular(self):
        eq, ca = tidal.device.rectangular(self.height, self.width)
        self.assertAlmostEqual(eq, 2.76, places=2)
        self.assertAlmostEqual(ca, self.height*self.width, places=2)

    def test_multiple_circular(self):
        eq, ca = tidal.device.multiple_circular(self.diameters)
        self.assertAlmostEqual(eq, 5.48, places=2)
        self.assertAlmostEqual(ca, 23.56, places=2)


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


    def test_principal_directions(self):    
        width_direction=10
        direction1, direction2 = tidal.resource.principal_flow_directions(self.data.d, width_direction)

        self.assertEqual(direction1,172.0) 
        self.assertEqual(round(direction2,1),round(352.3,1))                                                                                   



        #import ipdb;ipdb.set_trace()


if __name__ == '__main__':
    unittest.main() 

