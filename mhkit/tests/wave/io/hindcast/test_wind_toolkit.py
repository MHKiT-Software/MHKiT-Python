from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from random import seed, randint
import matplotlib.pylab as plt
from datetime import datetime
import xarray.testing as xrt
import mhkit.wave.io.hindcast.wind_toolkit as wtk
from io import StringIO
import pandas as pd
import numpy as np
import contextlib
import unittest
import netCDF4
import inspect
import pickle
import time
import json
import sys
import os


testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,'..','..','..','..','..','examples','data','wave','wind_toolkit'))


class TestWINDToolkit(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.my = pd.read_csv(join(datadir,'wtk_multiyear.csv'),
            index_col = 'time_index',
            names = ['time_index','pressure_200m_0'],
            header = 0,
            dtype = {'pressure_200m_0':'float32'})
        self.my.index = pd.to_datetime(self.my.index)

        self.ml = pd.read_csv(join(datadir,'wtk_multiloc.csv'),
            index_col = 'time_index',
            names = ['time_index','windspeed_10m_0','windspeed_10m_1'],
            header = 0,
            dtype = {'windspeed_10m_0':'float32',
                     'windspeed_10m_1':'float32'})
        self.ml.index = pd.to_datetime(self.ml.index)

        self.mp = pd.read_csv(join(datadir,'wtk_multiparm.csv'),
            index_col = 'time_index',
            names = ['time_index','temperature_20m_0','temperature_40m_0'],
            header = 0,
            dtype = {'temperature_20m_0':'float32',
                     'temperature_40m_0':'float32'})
        self.mp.index = pd.to_datetime(self.mp.index)

        self.my_meta = pd.read_csv(join(datadir,'wtk_multiyear_meta.csv'),
            index_col = 0,
            names = ['latitude','longitude','country','state','county','timezone','elevation','offshore'],
            header = 0, 
            dtype = {'latitude':'float32','longitude':'float32',
                     'country':'str','state':'str','county':'str',
                     'timezone':'int16','elevation':'float32','offshore':'int16'})
        
        self.ml_meta = pd.read_csv(join(datadir,'wtk_multiloc_meta.csv'),
            index_col = 0,
            names = ['latitude','longitude','country','state','county','timezone','elevation','offshore'],
            header = 0, 
            dtype = {'latitude':'float32','longitude':'float32',
                     'country':'str','state':'str','county':'str',
                     'timezone':'int16','elevation':'float32','offshore':'int16'})
        
        self.mp_meta = pd.read_csv(join(datadir,'wtk_multiparm_meta.csv'),
            index_col = 0,
            names = ['latitude','longitude','country','state','county','timezone','elevation','offshore'],
            header = 0, 
            dtype = {'latitude':'float32','longitude':'float32',
                     'country':'str','state':'str','county':'str',
                     'timezone':'int16','elevation':'float32','offshore':'int16'})

    @classmethod
    def tearDownClass(self):
        pass

    ## WIND Toolkit data
    def test_multi_year(self):
        data_type = '1-hour'
        years = [2018,2019]
        lat_lon = (44.624076,-124.280097) # NW_Pacific
        parameters = 'pressure_200m'
        wtk_multiyear, meta = wtk.request_wtk_point_data(
                                        data_type, parameters,
                                        lat_lon, years)
        assert_frame_equal(self.my,wtk_multiyear)
        assert_frame_equal(self.my_meta,meta)


    def test_multi_loc(self):
        data_type = '1-hour'
        years = [2001]
        lat_lon = ((39.33,-67.21),(41.3,-75.9)) # Mid-Atlantic
        parameters = 'windspeed_10m'
        wtk_multiloc, meta = wtk.request_wtk_point_data(
                                        data_type, parameters,
                                        lat_lon, years)
        assert_frame_equal(self.ml,wtk_multiloc)
        assert_frame_equal(self.ml_meta,meta)


    def test_multi_parm(self):
        data_type = '1-hour'
        years = [2012]
        lat_lon = (17.2,-156.5) # Hawaii
        parameters = ['temperature_20m','temperature_40m']
        wtk_multiparm, meta = wtk.request_wtk_point_data(
                                        data_type, parameters,
                                        lat_lon, years)
        assert_frame_equal(self.mp,wtk_multiparm)
        assert_frame_equal(self.mp_meta,meta)
    
    # test region_selection function and catch for the preferred region
    def test_region(self):
        region = wtk.region_selection((41.9,-125.3), preferred_region='Offshore_CA')
        assert region=='Offshore_CA'
        
        region = wtk.region_selection((41.9,-125.3), preferred_region='NW_Pacific')
        assert region=='NW_Pacific'
        
        try:
            region = wtk.region_selection((41.9,-125.3))
        except TypeError:
            pass
        else:
            assert False, 'Check wind_toolkit.region_selection() method for catching regional overlap'
        
        region = wtk.region_selection((36.3,-122.3), preferred_region='')
        assert region=='Offshore_CA'
        
        region = wtk.region_selection((16.3,-155.3), preferred_region='')
        assert region=='Hawaii'
        
        region = wtk.region_selection((45.3,-126.3), preferred_region='')
        assert region=='NW_Pacific'
        
        region = wtk.region_selection((39.3,-70.3), preferred_region='')
        assert region=='Mid_Atlantic'
    
    # test the check for multiple region
    def test_multi_region(self):
        data_type = '1-hour'
        years = [2012]
        lat_lon = ((17.2,-156.5),(45.3,-126.3))
        parameters = ['temperature_20m']
        try:
            data, meta = wtk.request_wtk_point_data(
                                    data_type, parameters,
                                    lat_lon, years)
        except TypeError:
            pass
        else:
            assert False, 'Check wind_toolkit.region_selection() method for catching requests over multiple regions'

    # test plot_region()
    def test_plot_region(self):
        fig, ax1 = plt.subplots()
        ax1 = wtk.plot_region('Mid_Atlantic',ax=ax1)
        
        ax2 = wtk.plot_region('NW_Pacific')
    
    # test elevation_to_string()
    def test_elevation_to_string(self):

        parameter = 'windspeed'
        elevations = [20, 40, 60, 120, 180]
        parameter_list = wtk.elevation_to_string(parameter, elevations)
        assert parameter_list==['windspeed_20m','windspeed_40m','windspeed_60m',
                                'windspeed_120m','windspeed_180m']
    

if __name__ == '__main__':
    unittest.main()
