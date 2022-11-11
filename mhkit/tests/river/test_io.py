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

    def test_get_all_time(self): 
        data= self.d3d_flume_data
        seconds_run = river.io.d3d.get_all_time(data)
        seconds_run_expected= np.ndarray(shape=(5,), buffer= np.array([0, 60, 120, 180, 240]), dtype=int)
        np.testing.assert_array_equal(seconds_run, seconds_run_expected)
        
    def test_convert_time(self): 
        data= self.d3d_flume_data
        time_index = 2
        seconds_run = river.io.d3d.index_to_seconds(data, time_index = time_index)
        seconds_run_expected = 120 
        self.assertEqual(seconds_run, seconds_run_expected)
        seconds_run = 60
        time_index= river.io.d3d.seconds_to_index(data, seconds_run = seconds_run)
        time_index_expected = 1
        self.assertEqual(time_index, time_index_expected)
        seconds_run = 62
        time_index= river.io.d3d.seconds_to_index(data, seconds_run = seconds_run)
        time_index_expected = 1
        output_expected= f'ERROR: invalid seconds_run. Closest seconds_run found {time_index_expected}'
        self.assertWarns(UserWarning)

    def test_layer_data(self): 
        data=self.d3d_flume_data
        variable = ['ucx', 's1']
        for var in variable:
            layer=2 
            time_index= 3
            layer_data= river.io.d3d.get_layer_data(data, var, layer, time_index)
            layer_compare = 2
            time_index_compare= 4
            layer_data_expected= river.io.d3d.get_layer_data(data,
                                                            var, layer_compare,
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
        
        x=np.linspace(1, 3, num= 3)
        y=2
        z=1 
        points= river.io.d3d.create_points(x,y,z)
        x=[1,2,3]
        y=[2,2,2]
        z=[1,1,1]
        points_array= np.array([ [x_i, y_i, z_i] for x_i, y_i, z_i in zip(x, y, z)]) 
        points_expected= pd.DataFrame(points_array, columns=('x','y','z'))
        assert_array_almost_equal(points, points_expected,decimal = 2)  
        
        x=3
        y=2
        z=1 
        points= river.io.d3d.create_points(x,y,z)
        output_expected='Can provide at most two arrays'
        self.assertWarns(UserWarning)
        
    def test_variable_interpolation(self):
        data=self.d3d_flume_data
        variables= ['ucx','turkin1']
        transformes_data= river.io.d3d.variable_interpolation(data, variables, points= 'faces', edges='nearest')
        self.assertEqual(np.size(transformes_data['ucx']), np.size(transformes_data['turkin1']))
        transformes_data= river.io.d3d.variable_interpolation(data, variables, points= 'cells', edges='nearest')
        self.assertEqual(np.size(transformes_data['ucx']), np.size(transformes_data['turkin1']))        
        x=np.linspace(1, 3, num= 3)
        y=np.linspace(1, 3, num= 3)
        waterdepth=1 
        points= river.io.d3d.create_points(x,y,waterdepth)
        transformes_data= river.io.d3d.variable_interpolation(data, variables, points= points)
        self.assertEqual(np.size(transformes_data['ucx']), np.size(transformes_data['turkin1']))
        
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
        time_index= -1
        x_test=np.linspace(1, 17, num= 10)
        y_test=np.linspace(3, 3, num= 10)
        waterdepth_test=np.linspace(1, 1, num= 10)
       
        test_points = np.array([ [x, y, waterdepth] for x, y, waterdepth in zip(x_test, y_test, waterdepth_test)])
        points= pd.DataFrame(test_points, columns=['x','y','waterdepth'])
        
        TI= river.io.d3d.turbulent_intensity(data, points, time_index)

        TI_vars= ['turkin1', 'ucx', 'ucy', 'ucz']
        TI_data_raw = {}
        for var in TI_vars:
            #get all data
            var_data_df = river.io.d3d.get_all_data_points(data, var,time_index)           
            TI_data_raw[var] = var_data_df 
            TI_data= points.copy(deep=True)
        
        for var in TI_vars:    
            TI_data[var] = interp.griddata(TI_data_raw[var][['x','y','waterdepth']],
                                TI_data_raw[var][var], points[['x','y','waterdepth']])
            idx= np.where(np.isnan(TI_data[var]))
        
            if len(idx[0]):
                for i in idx[0]: 
                    TI_data[var][i]= interp.griddata(TI_data_raw[var][['x','y','waterdepth']], 
                                TI_data_raw[var][var],
                                [points['x'][i],points['y'][i], points['waterdepth'][i]],
                                method='nearest')
        
        u_mag=river.io.d3d.unorm(TI_data['ucx'],TI_data['ucy'], TI_data['ucz'])
        turbulent_intensity_expected= (np.sqrt(2/3*TI_data['turkin1'])/u_mag)*100
       
        
        assert_array_almost_equal(TI.turbulent_intensity, turbulent_intensity_expected, decimal = 2)     
        
        TI = river.io.d3d.turbulent_intensity(data, points='faces')
        TI_size = np.size(TI['turbulent_intensity'])
        turkin1= river.io.d3d.get_all_data_points(data, 'turkin1',time_index)
        turkin1_size= np.size(turkin1['turkin1'])
        self.assertEqual(TI_size, turkin1_size)
        
        TI = river.io.d3d.turbulent_intensity(data, points='cells')
        TI_size = np.size(TI['turbulent_intensity'])
        ucx= river.io.d3d.get_all_data_points(data, 'ucx',time_index)
        ucx_size= np.size(ucx['ucx'])
        self.assertEqual(TI_size, ucx_size)
if __name__ == '__main__':
    unittest.main() 

