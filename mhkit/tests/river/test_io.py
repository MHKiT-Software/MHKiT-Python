
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


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        d3ddatadir = normpath(join(datadir,'d3d'))
        
        filename= 'turbineTest_map.nc'
        self.d3d_flume_data = netCDF4.Dataset(join(d3ddatadir,filename))
        
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_load_usgs_data_instantaneous(self):
        file_name = join(datadir, 'USGS_08313000_Jan2019_instantaneous.json')
        data = river.io.usgs.read_usgs_file(file_name)
        
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual(data.shape, (2972, 1)) # 4 data points are missing
        
    def test_load_usgs_data_daily(self):
        file_name = join(datadir, 'USGS_08313000_Jan2019_daily.json')
        data = river.io.usgs.read_usgs_file(file_name)

        expected_index = pd.date_range('2019-01-01', '2019-01-31', freq='D')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual((data.index == expected_index.tz_localize('UTC')).all(), True)
        self.assertEqual(data.shape, (31, 1))


    def test_request_usgs_data_daily(self):
        data=river.io.usgs.request_usgs_data(station="15515500",
                            parameter='00060',
                            start_date='2009-08-01',
                            end_date='2009-08-10',
                            data_type='Daily')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        self.assertEqual(data.shape, (10, 1))
    
   
    def test_request_usgs_data_instant(self):
        data=river.io.usgs.request_usgs_data(station="15515500",
                            parameter='00060',
                            start_date='2009-08-01',
                            end_date='2009-08-10',
                            data_type='Instantaneous')
        self.assertEqual(data.columns, ['Discharge, cubic feet per second'])
        # Every 15 minutes or 4 times per hour
        self.assertEqual(data.shape, (10*24*4, 1))


    def test_layer_data(self): 
        data=self.d3d_flume_data
        variable= 'ucx'
        layer=2 
        time_index= 3
        layer_data= river.io.d3d.get_layer_data(data, variable, layer, time_index)
        layer_compare = 2
        time_index_compare= 4
        layer_data_expected= river.io.d3d.get_layer_data(data,
                                                        variable, layer_compare,
                                                        time_index_compare)
       
        assert_array_almost_equal(layer_data.x,layer_data_expected.x, decimal = 2)
        assert_array_almost_equal(layer_data.y,layer_data_expected.y, decimal = 2)
        assert_array_almost_equal(layer_data.v,layer_data_expected.v, decimal= 2)
        
        
    def test_create_points(self):
        x=np.linspace(1, 3, num= 3)
        y=np.linspace(1, 3, num= 3)
        z=1 
        points= river.io.d3d.create_points(x,y,z)
        
        x=[1,2,3,1,2,3,1,2,3]
        y=[1,1,1,2,2,2,3,3,3]
        z=[1,1,1,1,1,1,1,1,1]
        
        points_array= np.array([ [x_i, y_i, z_i] for x_i, y_i, z_i in zip(x, y, z)]) 
        points_expected= pd.DataFrame(points_array, columns=('x','y','z'))
        assert_array_almost_equal(points, points_expected,decimal = 2)  
        
        
    def test_get_all_data_points(self): 
        data=self.d3d_flume_data
        variable= 'ucx'
        time_step= 3
        output = river.io.d3d.get_all_data_points(data, variable, time_step)
        size_output = np.size(output) 
        time_step_compair=4
        output_expected= river.io.d3d.get_all_data_points(data, variable, time_step_compair)
        size_output_expected= np.size(output_expected)
        self.assertEqual(size_output, size_output_expected)
 
    
    def test_unorm(self): 
        x=np.linspace(1, 3, num= 3)
        y=np.linspace(1, 3, num= 3)
        z=np.linspace(1, 3, num= 3)
        unorm = river.io.d3d.unorm(x,y,z)
        unorm_expected= [np.sqrt(1**2+1**2+1**2),np.sqrt(2**2+2**2+2**2), np.sqrt(3**2+3**2+3**2)]
        assert_array_almost_equal(unorm, unorm_expected, decimal = 2) 
    
    def test_turbulent_intensity(self): 
        data=self.d3d_flume_data
        time_step= -1
        x_test=np.linspace(1, 17, num= 10)
        y_test=np.linspace(3, 3, num= 10)
        z_test=np.linspace(1, 1, num= 10)
       
        test_points = np.array([ [x, y, z] for x, y, z in zip(x_test, y_test, z_test)])
        points= pd.DataFrame(test_points, columns=['x','y','z'])
        
        TI= river.io.d3d.turbulent_intensity(data, points, time_step)

        TI_vars= ['turkin1', 'ucx', 'ucy', 'ucz']
        TI_data_raw = {}
        for var in TI_vars:
            #get all data
            var_data_df = river.io.d3d.get_all_data_points(data, var,time_step)           
            TI_data_raw[var] = var_data_df 
            TI_data= points.copy(deep=True)
        
        for var in TI_vars:    
            TI_data[var] = interp.griddata(TI_data_raw[var][['x','y','z']],
                                TI_data_raw[var][var], points[['x','y','z']])
        
        u_mag=river.io.d3d.unorm(TI_data['ucx'],TI_data['ucy'], TI_data['ucz'])
        turbulent_intensity_expected= np.sqrt(2/3*TI_data['turkin1'])/u_mag
       
        
        assert_array_almost_equal(TI.turbulent_intensity, turbulent_intensity_expected, decimal = 2)     
       
if __name__ == '__main__':
    unittest.main() 

