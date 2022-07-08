from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from random import seed, randint
import matplotlib.pylab as plt
from datetime import datetime
import xarray.testing as xrt
import mhkit.wave as wave
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
datadir = normpath(join(testdir,relpath('../../../../examples/data/wave')))


class TestWPTOhindcast(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.my_swh = pd.read_csv(join(datadir,'hindcast/multi_year_hindcast.csv'),index_col = 'time_index',
        names = ['time_index','significant_wave_height_0'],header = 0,
        dtype = {'significant_wave_height_0':'float32'})
        self.my_swh.index = pd.to_datetime(self.my_swh.index)

        self.ml = pd.read_csv(join(datadir,'hindcast/single_year_hindcast_multiloc.csv'),index_col = 'time_index',
        names = ['time_index','mean_absolute_period_0','mean_absolute_period_1'],
        header = 0, dtype = {'mean_absolute_period_0':'float32',
        'mean_absolute_period_1':'float32'})
        self.ml.index = pd.to_datetime(self.ml.index)

        self.mp = pd.read_csv(join(datadir,'hindcast/multiparm.csv'),index_col = 'time_index',
        names = ['time_index','energy_period_0','mean_zero-crossing_period_0'],
        header = 0, dtype = {'energy_period_0':'float32',
        'mean_zero-crossing_period_0':'float32'})
        self.mp.index = pd.to_datetime(self.mp.index)

        self.ml_meta = pd.read_csv(join(datadir,'hindcast/multiloc_meta.csv'),index_col = 0,
        names = [None,'water_depth','latitude','longitude','distance_to_shore','timezone'
        ,'jurisdiction'],header = 0, dtype = {'water_depth':'float32','latitude':'float32'
        ,'longitude':'float32','distance_to_shore':'float32','timezone':'int16'})

        self.my_meta = pd.read_csv(join(datadir,'hindcast/multi_year_meta.csv'),index_col = 0,
        names = [None,'water_depth','latitude','longitude','distance_to_shore','timezone'
        ,'jurisdiction'],header = 0, dtype = {'water_depth':'float32','latitude':'float32'
        ,'longitude':'float32','distance_to_shore':'float32','timezone':'int16'})

        self.mp_meta = pd.read_csv(join(datadir,'hindcast/multiparm_meta.csv'),index_col = 0,
        names = [None,'water_depth','latitude','longitude','distance_to_shore','timezone'
        ,'jurisdiction'],header = 0, dtype = {'water_depth':'float32','latitude':'float32'
        ,'longitude':'float32','distance_to_shore':'float32','timezone':'int16'})

        my_dir = pd.read_csv(join(datadir,'hindcast/multi_year_dir.csv'),header = 0,
        dtype={'87':'float32','58':'float32'})
        my_dir['time_index'] = pd.to_datetime(my_dir['time_index'])
        my_dir = my_dir.set_index(['time_index','frequency','direction'])
        self.my_dir = my_dir.to_xarray()

        self.my_dir_meta = pd.read_csv(join(datadir,'hindcast/multi_year_dir_meta.csv'),
        names = ['water_depth','latitude','longitude','distance_to_shore','timezone'
        ,'jurisdiction'],header = 0, dtype = {'water_depth':'float32','latitude':'float32'
        ,'longitude':'float32','distance_to_shore':'float32','timezone':'int16'})

    @classmethod
    def tearDownClass(self):
        pass

    ## WPTO hindcast data
    #   only run test for one version of python per to not spam the server
    #   yet keep coverage high on each test
    if float(sys.version[0:3]) == 3.7:
        def test_multi_year(self):
            data_type = '3-hour'
            years = [1990,1992]
            lat_lon = (44.624076,-124.280097)
            parameters = 'significant_wave_height'
            (wave_multiyear,
            meta) = (wave.io.hindcast
                    .request_wpto_point_data(data_type,parameters,
                                            lat_lon,years))
            assert_frame_equal(self.my_swh,wave_multiyear)
            assert_frame_equal(self.my_meta,meta)

    elif float(sys.version[0:3]) == 3.8:
        # wait five minute to ensure python 3.7 call is complete
        time.sleep(300)
        def test_multi_loc(self):
            data_type = '3-hour'
            years = [1995]
            lat_lon = ((44.624076,-124.280097),(43.489171,-125.152137))
            parameters = 'mean_absolute_period'
            wave_multiloc, meta=(wave.io.hindcast
                                .request_wpto_point_data(data_type,
                                              parameters,lat_lon,years))
            (dir_multiyear,
            meta_dir)=(wave.io.hindcast
                       .request_wpto_directional_spectrum(lat_lon,year='1995'))
            dir_multiyear = dir_multiyear.sel(time_index=slice(dir_multiyear.time_index[0],dir_multiyear.time_index[99]))
            dir_multiyear = dir_multiyear.rename_vars({87:'87',58:'58'})

            assert_frame_equal(self.ml,wave_multiloc)
            assert_frame_equal(self.ml_meta,meta)
            xrt.assert_allclose(self.my_dir,dir_multiyear)
            assert_frame_equal(self.my_dir_meta,meta_dir)

    elif float(sys.version[0:3]) == 3.9:
        # wait ten minutes to ensure python 3.7 and 3.8 call is complete
        time.sleep(500)

        def test_multi_parm(self):
            data_type = '1-hour'
            years = [1996]
            lat_lon = (44.624076,-124.280097)
            parameters = ['energy_period','mean_zero-crossing_period']
            wave_multiparm, meta= wave.io.hindcast.request_wpto_point_data(data_type,
            parameters,lat_lon,years)

            assert_frame_equal(self.mp,wave_multiparm)
            assert_frame_equal(self.mp_meta,meta)


if __name__ == '__main__':
    unittest.main()
