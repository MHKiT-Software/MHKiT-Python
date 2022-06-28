from os.path import abspath, dirname, join, isfile, normpath, relpath
import matplotlib.pylab as plt
import mhkit.tidal as tidal
import pandas as pd
import numpy as np
import unittest
import os


testdir = dirname(abspath(__file__))
plotdir = join(testdir, 'plots')
isdir = os.path.isdir(plotdir)
if not isdir: os.mkdir(plotdir)
datadir = normpath(join(testdir,relpath('../../../examples/data/tidal')))


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass
        
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_load_noaa_data(self):
        file_name = join(datadir, 's08010.json')
        data, metadata = tidal.io.noaa.read_noaa_json(file_name)
        self.assertTrue(np.all(data.columns == ['s','d','b']) )
        self.assertEqual(data.shape, (18890, 3))

    def test_request_noaa_data(self):
        data, metadata = tidal.io.noaa.request_noaa_data(station='s08010', parameter='currents',
                                       start_date='20180101', end_date='20180102',
                                       proxy=None, write_json=None)
        self.assertTrue(np.all(data.columns == ['s','d','b']) )
        self.assertEqual(data.shape, (92, 3))


if __name__ == '__main__':
    unittest.main() 

