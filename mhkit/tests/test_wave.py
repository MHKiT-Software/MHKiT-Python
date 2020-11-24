from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from datetime import datetime
import mhkit.wave as wave
from io import StringIO
import pandas as pd
import numpy as np
import contextlib
import unittest
import inspect
import pickle
import json
import os

testdir = dirname(abspath(__file__))
datadir = normpath(join(testdir,relpath('../../examples/data/wave')))


class TestResourceSpectrum(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        omega = np.arange(0.1,3.5,0.01)
        self.f = omega/(2*np.pi)
        self.Hs = 2.5
        self.Tp = 8
        df = self.f[1] - self.f[0]
        Trep = 1/df
        self.t = np.arange(0, Trep, 0.05)
            
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_pierson_moskowitz_spectrum(self):
        S = wave.resource.pierson_moskowitz_spectrum(self.f,self.Tp)
        Tp0 = wave.resource.peak_period(S).iloc[0,0]
        
        error = np.abs(self.Tp - Tp0)/self.Tp
        
        self.assertLess(error, 0.01)
        
    def test_bretschneider_spectrum(self):
        S = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)
        Hm0 = wave.resource.significant_wave_height(S).iloc[0,0]
        Tp0 = wave.resource.peak_period(S).iloc[0,0]
        
        errorHm0 = np.abs(self.Tp - Tp0)/self.Tp
        errorTp0 = np.abs(self.Hs - Hm0)/self.Hs
        
        self.assertLess(errorHm0, 0.01)
        self.assertLess(errorTp0, 0.01)

    def test_surface_elevation_seed(self):
        S = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)

        sig = inspect.signature(wave.resource.surface_elevation)
        seednum = sig.parameters['seed'].default
        
        eta0 = wave.resource.surface_elevation(S, self.t)
        eta1 = wave.resource.surface_elevation(S, self.t, seed=seednum)                
        
        assert_frame_equal(eta0, eta1)        

    def test_surface_elevation_phasing(self):
        S = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)
        eta0 = wave.resource.surface_elevation(S, self.t)        
        sig = inspect.signature(wave.resource.surface_elevation)
        seednum = sig.parameters['seed'].default
        np.random.seed(seednum)
        phases = np.random.rand(len(S)) * 2 * np.pi
        eta1 = wave.resource.surface_elevation(S, self.t, phases=phases)

        assert_frame_equal(eta0, eta1)


    def test_surface_elevation_phases_np_and_pd(self):
        S0 = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)
        S1 = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs*1.1)
        S = pd.concat([S0, S1], axis=1)

        phases_np = np.random.rand(S.shape[0], S.shape[1]) * 2 * np.pi
        phases_pd = pd.DataFrame(phases_np, index=S.index, columns=S.columns)

        eta_np = wave.resource.surface_elevation(S, self.t, phases=phases_np)
        eta_pd = wave.resource.surface_elevation(S, self.t, phases=phases_pd)

        assert_frame_equal(eta_np, eta_pd)

    def test_surface_elevation_frequency_bins_np_and_pd(self):
        S0 = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)
        S1 = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs*1.1)
        S = pd.concat([S0, S1], axis=1)

        eta0 = wave.resource.surface_elevation(S, self.t)

        f_bins_np = np.array([np.diff(S.index)[0]]*len(S))
        f_bins_pd = pd.DataFrame(f_bins_np, index=S.index, columns=['df'])

        eta_np = wave.resource.surface_elevation(S, self.t, frequency_bins=f_bins_np)        
        eta_pd = wave.resource.surface_elevation(S, self.t, frequency_bins=f_bins_pd)

        assert_frame_equal(eta0, eta_np)        
        assert_frame_equal(eta_np, eta_pd)        

    def test_surface_elevation_moments(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        eta = wave.resource.surface_elevation(S, self.t)
        dt = self.t[1] - self.t[0]
        Sn = wave.resource.elevation_spectrum(eta, 1/dt, len(eta.values), 
                                              detrend=False, window='boxcar',
                                              noverlap=0)

        m0 = wave.resource.frequency_moment(S,0).m0.values[0]
        m0n = wave.resource.frequency_moment(Sn,0).m0.values[0]
        errorm0 = np.abs((m0 - m0n)/m0)

        self.assertLess(errorm0, 0.01)

        m1 = wave.resource.frequency_moment(S,1).m1.values[0]
        m1n = wave.resource.frequency_moment(Sn,1).m1.values[0]
        errorm1 = np.abs((m1 - m1n)/m1)

        self.assertLess(errorm1, 0.01)

    def test_surface_elevation_rmse(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        eta = wave.resource.surface_elevation(S, self.t)
        dt = self.t[1] - self.t[0]
        Sn = wave.resource.elevation_spectrum(eta, 1/dt, len(eta), 
                                              detrend=False, window='boxcar',
                                              noverlap=0)

        fSn = interp1d(Sn.index.values, Sn.values, axis=0)
        rmse = (S.values - fSn(S.index.values))**2
        rmse_sum = (np.sum(rmse)/len(rmse))**0.5

        self.assertLess(rmse_sum, 0.02)
    
    def test_jonswap_spectrum(self):
        S = wave.resource.jonswap_spectrum(self.f, self.Tp, self.Hs)
        Hm0 = wave.resource.significant_wave_height(S).iloc[0,0]
        Tp0 = wave.resource.peak_period(S).iloc[0,0]
        
        errorHm0 = np.abs(self.Tp - Tp0)/self.Tp
        errorTp0 = np.abs(self.Hs - Hm0)/self.Hs
        
        self.assertLess(errorHm0, 0.01)
        self.assertLess(errorTp0, 0.01)
    
    def test_plot_spectrum(self):            
        filename = abspath(join(testdir, 'wave_plot_spectrum.png'))
        if isfile(filename):
            os.remove(filename)
        
        S = wave.resource.pierson_moskowitz_spectrum(self.f,self.Tp)
        
        plt.figure()
        wave.graphics.plot_spectrum(S)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))

    def test_plot_chakrabarti(self):            
        filename = abspath(join(testdir, 'wave_plot_chakrabarti.png'))
        if isfile(filename):
            os.remove(filename)
        
        D = 5
        H = 10
        lambda_w = 200

        wave.graphics.plot_chakrabarti(H, lambda_w, D)
        plt.savefig(filename)

    def test_plot_chakrabarti_np(self):            
        filename = abspath(join(testdir, 'wave_plot_chakrabarti_np.png'))
        if isfile(filename):
            os.remove(filename)
        
        D = np.linspace(5, 15, 5)
        H = 10 * np.ones_like(D)
        lambda_w = 200 * np.ones_like(D)

        wave.graphics.plot_chakrabarti(H, lambda_w, D)
        plt.savefig(filename)
        
        self.assertTrue(isfile(filename))

    def test_plot_chakrabarti_pd(self):            
        filename = abspath(join(testdir, 'wave_plot_chakrabarti_pd.png'))
        if isfile(filename):
            os.remove(filename)
        
        D = np.linspace(5, 15, 5)
        H = 10 * np.ones_like(D)
        lambda_w = 200 * np.ones_like(D)
        df = pd.DataFrame([H.flatten(),lambda_w.flatten(),D.flatten()],
                         index=['H','lambda_w','D']).transpose()

        wave.graphics.plot_chakrabarti(df.H, df.lambda_w, df.D)
        plt.savefig(filename)
        
        self.assertTrue(isfile(filename))
        

class TestResourceMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        file_name = join(datadir, 'ValData1.json')
        with open(file_name, "r") as read_file:
            self.valdata1 = pd.DataFrame(json.load(read_file))
        
        self.valdata2 = {}

        file_name = join(datadir, 'ValData2_MC.json')
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2['MC'] = data
        for i in data.keys():
            # Calculate elevation spectra
            elevation = pd.DataFrame(data[i]['elevation'])
            elevation.index = elevation.index.astype(float)
            elevation.sort_index(inplace=True)
            sample_rate = data[i]['sample_rate']
            NFFT = data[i]['NFFT']
            self.valdata2['MC'][i]['S'] = wave.resource.elevation_spectrum(elevation, 
                         sample_rate, NFFT)

        file_name = join(datadir, 'ValData2_AH.json')
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2['AH'] = data
        for i in data.keys():
            # Calculate elevation spectra
            elevation = pd.DataFrame(data[i]['elevation'])
            elevation.index = elevation.index.astype(float)
            elevation.sort_index(inplace=True)
            sample_rate = data[i]['sample_rate']
            NFFT = data[i]['NFFT']
            self.valdata2['AH'][i]['S'] = wave.resource.elevation_spectrum(elevation, 
                         sample_rate, NFFT)
        
        file_name = join(datadir, 'ValData2_CDiP.json')       
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self.valdata2['CDiP'] = data
        for i in data.keys():
            temp = pd.Series(data[i]['S']).to_frame('S')
            temp.index = temp.index.astype(float)
            self.valdata2['CDiP'][i]['S'] = temp

            
    @classmethod
    def tearDownClass(self):
        pass

    def test_kfromw(self):
        for i in self.valdata1.columns:
            f = np.array(self.valdata1[i]['w'])/(2*np.pi)
            h = self.valdata1[i]['h']
            rho = self.valdata1[i]['rho']
            
            expected = self.valdata1[i]['k']
            calculated = wave.resource.wave_number(f, h, rho).loc[:,'k'].values
            error = ((expected-calculated)**2).sum() # SSE
            
            self.assertLess(error, 1e-6)
    
    def test_moments(self):
        for file_i in self.valdata2.keys(): # for each file MC, AH, CDiP
            datasets = self.valdata2[file_i]
            for s in datasets.keys(): # for each set
                data = datasets[s]
                for m in data['m'].keys():
                    expected = data['m'][m]
                    S = data['S']
                    if s == 'CDiP1' or s == 'CDiP6':
                        f_bins=pd.Series(data['freqBinWidth'])                       
                    else: 
                        f_bins = None

                    calculated = wave.resource.frequency_moment(S, int(m)
                                       ,frequency_bins=f_bins).iloc[0,0]
                    error = np.abs(expected-calculated)/expected
                    
                    self.assertLess(error, 0.01) 

    

    def test_metrics(self):
       for file_i in self.valdata2.keys(): # for each file MC, AH, CDiP
            datasets = self.valdata2[file_i]
            
            for s in datasets.keys(): # for each set
                
                
                data = datasets[s]
                S = data['S']
                if file_i == 'CDiP':
                    f_bins=pd.Series(data['freqBinWidth'])
                else: 
                    f_bins = None
                
                # Hm0
                expected = data['metrics']['Hm0']
                calculated = wave.resource.significant_wave_height(S,
                                        frequency_bins=f_bins).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('Hm0', expected, calculated, error)
                self.assertLess(error, 0.01) 

                # Te
                expected = data['metrics']['Te']
                calculated = wave.resource.energy_period(S,
                                        frequency_bins=f_bins).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('Te', expected, calculated, error)
                self.assertLess(error, 0.01) 
                
                # T0
                expected = data['metrics']['T0']
                calculated = wave.resource.average_zero_crossing_period(S,
                                         frequency_bins=f_bins).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('T0', expected, calculated, error)
                self.assertLess(error, 0.01) 

                # Tc
                expected = data['metrics']['Tc']
                calculated = wave.resource.average_crest_period(S,
                # Tc = Tavg**2
                                     frequency_bins=f_bins).iloc[0,0]**2 
                error = np.abs(expected-calculated)/expected
                #print('Tc', expected, calculated, error)
                self.assertLess(error, 0.01) 

                # Tm
                expected = np.sqrt(data['metrics']['Tm'])
                calculated = wave.resource.average_wave_period(S,
                                        frequency_bins=f_bins).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('Tm', expected, calculated, error)
                self.assertLess(error, 0.01) 
                
                # Tp
                expected = data['metrics']['Tp']
                calculated = wave.resource.peak_period(S).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('Tp', expected, calculated, error)
                self.assertLess(error, 0.001) 
                
                # e
                expected = data['metrics']['e']
                calculated = wave.resource.spectral_bandwidth(S,
                                        frequency_bins=f_bins).iloc[0,0]
                error = np.abs(expected-calculated)/expected
                #print('e', expected, calculated, error)
                self.assertLess(error, 0.001) 

                # v
                if file_i == 'CDiP': 
                    # this should be updated to run on other datasets
                    expected = data['metrics']['v']                    
                    calculated = wave.resource.spectral_width(S,
                                        frequency_bins=f_bins).iloc[0,0]
                    error = np.abs(expected-calculated)/expected

                       
                    self.assertLess(error, 0.01) 

                if file_i == 'MC':
                    expected = data['metrics']['v']
                    # testing that default uniform frequency bin widths works 
                    calculated = wave.resource.spectral_width(S).iloc[0,0] 
                    error = np.abs(expected-calculated)/expected

                       
                    self.assertLess(error, 0.01)

                
    def test_plot_elevation_timeseries(self):            
        filename = abspath(join(testdir, 'wave_plot_elevation_timeseries.png'))
        if isfile(filename):
            os.remove(filename)
        
        data = self.valdata2['MC']
        temp = pd.DataFrame(data[list(data.keys())[0]]['elevation'])
        temp.index = temp.index.astype(float)
        temp.sort_index(inplace=True)
        eta = temp.iloc[0:100,:]
        
        plt.figure()
        wave.graphics.plot_elevation_timeseries(eta)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))


class TestResourceContours(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        
        f_name= 'Hm0_Te_46022.json'
        self.Hm0Te = pd.read_json(join(datadir,f_name))
        

        with open(join(datadir, 'principal_component_analysis.pkl'), 'rb') as f:
            self.pca = pickle.load(f)                       

        
            
    @classmethod
    def tearDownClass(self):
        pass
        
    def test_environmental_contour(self):
       
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te['Hm0'] < 20]
        
        Hm0 = df.Hm0.values  
        Te = df.Te.values 
        
        dt_ss = (Hm0Te.index[2]-Hm0Te.index[1]).seconds  
        time_R = 100  
        
        Hm0_contour, Te_contour = wave.resource.environmental_contour(Hm0, Te, 
                                                    dt_ss, time_R)
        
        expected_contours = pd.read_csv(join(datadir,'Hm0_Te_contours_46022.csv'))
        assert_allclose(expected_contours.Hm0_contour.values, Hm0_contour, rtol=1e-3)
        
    def test__principal_component_analysis(self):
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te['Hm0'] < 20]
        
        Hm0 = df.Hm0.values  
        Te = df.Te.values 
        PCA = wave.resource._principal_component_analysis(Hm0,Te, bin_size=250)
        
        assert_allclose(PCA['principal_axes'], self.pca['principal_axes'])
        self.assertAlmostEqual(PCA['shift'], self.pca['shift'])
        self.assertAlmostEqual(PCA['x1_fit'], self.pca['x1_fit'])
        self.assertAlmostEqual(PCA['mu_fit'].slope, self.pca['mu_fit'].slope)
        self.assertAlmostEqual(PCA['mu_fit'].intercept, self.pca['mu_fit'].intercept)
        assert_allclose(PCA['sigma_fit']['x'], self.pca['sigma_fit']['x'])
        
    def test_plot_environmental_contour(self):
        filename = abspath(join(testdir, 'wave_plot_environmental_contour.png'))
        if isfile(filename):
            os.remove(filename)
        
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te['Hm0'] < 20]
        
        Hm0 = df.Hm0.values  
        Te = df.Te.values 
        
        dt_ss = (Hm0Te.index[2]-Hm0Te.index[1]).seconds  
        time_R = 100  
        
        Hm0_contour, Te_contour = wave.resource.environmental_contour(Hm0, Te, 
                                                    dt_ss, time_R)
        
        plt.figure()
        wave.graphics.plot_environmental_contour(Te, Hm0,
                                                 Te_contour, Hm0_contour,
                                                 data_label='NDBC 46022',
                                                 contour_label='100-year Contour',
                                                 x_label = 'Te [s]',
                                                 y_label = 'Hm0 [m]')
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))        

    def test_plot_environmental_contour_multiyear(self):
        filename = abspath(join(testdir, 
                       'wave_plot_environmental_contour_multiyear.png'))
        if isfile(filename):
            os.remove(filename)
        
        Hm0Te = self.Hm0Te
        df = Hm0Te[Hm0Te['Hm0'] < 20]
        
        Hm0 = df.Hm0.values  
        Te = df.Te.values 
        
        dt_ss = (Hm0Te.index[2]-Hm0Te.index[1]).seconds  

        time_R = np.array([100, 105, 110, 120, 150])
        
        Hm0_contour, Te_contour = wave.resource.environmental_contour(Hm0, Te, 
                                                    dt_ss, time_R)
        
        contour_label = [f'{year}-year Contour' for year in time_R]
        plt.figure()
        wave.graphics.plot_environmental_contour(Te, Hm0,
                                                 Te_contour, Hm0_contour,
                                                 data_label='NDBC 46022',
                                                 contour_label=contour_label,
                                                 x_label = 'Te [s]',
                                                 y_label = 'Hm0 [m]')
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))        
        
        
class TestPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(123)
        Hm0 = np.random.rayleigh(4, 100000)
        Te = np.random.normal(4.5, .8, 100000)
        P = np.random.normal(200, 40, 100000)
        J = np.random.normal(300, 10, 100000)
        
        self.data = pd.DataFrame({'Hm0': Hm0, 'Te': Te, 'P': P,'J': J})
        self.Hm0_bins = np.arange(0,19,0.5)
        self.Te_bins = np.arange(0,9,1)

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_capture_length(self):
        L = wave.performance.capture_length(self.data['P'], self.data['J'])
        L_stats = wave.performance.statistics(L)
        
        self.assertAlmostEqual(L_stats['mean'], 0.6676, 3)
        
    def test_capture_length_matrix(self):
        L = wave.performance.capture_length(self.data['P'], self.data['J'])
        LM = wave.performance.capture_length_matrix(self.data['Hm0'], self.data['Te'], 
                        L, 'std', self.Hm0_bins, self.Te_bins)
        
        self.assertEqual(LM.shape, (38,9))
        self.assertEqual(LM.isna().sum().sum(), 131)
        
    def test_wave_energy_flux_matrix(self):
        JM = wave.performance.wave_energy_flux_matrix(self.data['Hm0'], self.data['Te'], 
                        self.data['J'], 'mean', self.Hm0_bins, self.Te_bins)
        
        self.assertEqual(JM.shape, (38,9))
        self.assertEqual(JM.isna().sum().sum(), 131)
        
    def test_power_matrix(self):
        L = wave.performance.capture_length(self.data['P'], self.data['J'])
        LM = wave.performance.capture_length_matrix(self.data['Hm0'], self.data['Te'], 
                        L, 'mean', self.Hm0_bins, self.Te_bins)
        JM = wave.performance.wave_energy_flux_matrix(self.data['Hm0'], self.data['Te'], 
                        self.data['J'], 'mean', self.Hm0_bins, self.Te_bins)
        PM = wave.performance.power_matrix(LM, JM)
        
        self.assertEqual(PM.shape, (38,9))
        self.assertEqual(PM.isna().sum().sum(), 131)
        
    def test_mean_annual_energy_production(self):
        L = wave.performance.capture_length(self.data['P'], self.data['J'])
        maep = wave.performance.mean_annual_energy_production_timeseries(L, self.data['J'])

        self.assertAlmostEqual(maep, 1754020.077, 2)
    
        
    def test_plot_matrix(self):
        filename = abspath(join(testdir, 'wave_plot_matrix.png'))
        if isfile(filename):
            os.remove(filename)
        
        M = wave.performance.wave_energy_flux_matrix(self.data['Hm0'], self.data['Te'], 
                        self.data['J'], 'mean', self.Hm0_bins, self.Te_bins)
        
        plt.figure()
        wave.graphics.plot_matrix(M)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    
class TestIOndbc(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.expected_columns_metRT = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 
            'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
        self.expected_units_metRT = {'WDIR': 'degT', 'WSPD': 'm/s', 'GST': 'm/s', 
            'WVHT': 'm', 'DPD': 'sec', 'APD': 'sec', 'MWD': 'degT', 'PRES': 'hPa', 
            'ATMP': 'degC', 'WTMP': 'degC', 'DEWP': 'degC', 'VIS': 'nmi', 
            'PTDY': 'hPa', 'TIDE': 'ft'}
        
        self.expected_columns_metH = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 
            'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
        self.expected_units_metH = {'WDIR': 'degT', 'WSPD': 'm/s', 'GST': 'm/s', 
            'WVHT': 'm', 'DPD': 'sec', 'APD': 'sec', 'MWD': 'deg', 'PRES': 'hPa', 
            'ATMP': 'degC', 'WTMP': 'degC', 'DEWP': 'degC', 'VIS': 'nmi', 
            'TIDE': 'ft'}
        self.filenames=['46042w1996.txt.gz', 
                        '46029w1997.txt.gz', 
                        '46029w1998.txt.gz']
        self.swden = pd.read_csv(join(datadir,self.filenames[0]), sep=r'\s+', 
                                 compression='gzip')
        
    @classmethod
    def tearDownClass(self):
        pass
    
    ### Realtime data
    def test_ndbc_read_realtime_met(self):
        data, units = wave.io.ndbc.read_file(join(datadir, '46097.txt'))
        expected_index0 = datetime(2019,4,2,13,50)
        self.assertSetEqual(set(data.columns), set(self.expected_columns_metRT))
        self.assertEqual(data.index[0], expected_index0)
        self.assertEqual(data.shape, (6490, 14))
        self.assertEqual(units,self.expected_units_metRT)
            
    ### Historical data
    def test_ndbnc_read_historical_met(self):
        # QC'd monthly data, Aug 2019
        data, units = wave.io.ndbc.read_file(join(datadir, '46097h201908qc.txt'))
        expected_index0 = datetime(2019,8,1,0,0)
        self.assertSetEqual(set(data.columns), set(self.expected_columns_metH))
        self.assertEqual(data.index[0], expected_index0)
        self.assertEqual(data.shape, (4464, 13))
        self.assertEqual(units,self.expected_units_metH)
        
    ### Spectral data
    def test_ndbc_read_spectral(self):
        data, units = wave.io.ndbc.read_file(join(datadir, 'data.txt'))
        self.assertEqual(data.shape, (743, 47))
        self.assertEqual(units, None)
		
    def test_ndbc_available_data(self):
        data=wave.io.ndbc.available_data('swden', buoy_number='46029')
                
        cols = data.columns.tolist()
        exp_cols = ['id', 'year', 'filename']
        self.assertEqual(cols, exp_cols)                
                
        years = [int(year) for year in data.year.tolist()]
        exp_years=[*range(1996,1996+len(years))]
        self.assertEqual(years, exp_years)
        self.assertEqual(data.shape, (len(data), 3))

    def test__ndbc_parse_filenames(self):  
        filenames= pd.Series(self.filenames)
        buoys = wave.io.ndbc._parse_filenames('swden', filenames)
        years = buoys.year.tolist()
        numbers = buoys.id.tolist()
        fnames = buoys.filename.tolist()
        
        self.assertEqual(buoys.shape, (len(filenames),3))              
        self.assertListEqual(years, ['1996','1997','1998'])  
        self.assertListEqual(numbers, ['46042','46029','46029'])          
        self.assertListEqual(fnames, self.filenames)
        
    def test_ndbc_request_data(self):
        filenames= pd.Series(self.filenames[0])
        ndbc_data = wave.io.ndbc.request_data('swden', filenames)
        self.assertTrue(self.swden.equals(ndbc_data['1996']))

    def test_ndbc_request_data_from_dataframe(self):
        filenames= pd.DataFrame(pd.Series(data=self.filenames[0]))
        ndbc_data = wave.io.ndbc.request_data('swden', filenames)
        assert_frame_equal(self.swden, ndbc_data['1996'])

    def test_ndbc_request_data_filenames_length(self):
        with self.assertRaises(AssertionError):  
                               wave.io.ndbc.request_data('swden', pd.Series(dtype=float)) 

    def test_ndbc_to_datetime_index(self):
        dt = wave.io.ndbc.to_datetime_index('swden', self.swden)        
        self.assertEqual(type(dt.index), pd.DatetimeIndex)
        self.assertFalse({'YY','MM','DD','hh'}.issubset(dt.columns))       

    def test_ndbc_request_data_empty_file(self):
        temp_stdout = StringIO()
        # known empty file. If NDBC replaces, this test may fail. 
        filename = "42008h1984.txt.gz"  
        buoy_id='42008'
        year = '1984'
        with contextlib.redirect_stdout(temp_stdout):
            wave.io.ndbc.request_data('stdmet', pd.Series(filename))
        output = temp_stdout.getvalue().strip()
        msg = (f'The NDBC buoy {buoy_id} for year {year} with ' 
               f'filename {filename} is empty or missing '     
                'data. Please omit this file from your data '   
                'request in the future.')
        self.assertEqual(output, msg)

    def test_ndbc_request_multiple_files_with_empty_file(self):
        temp_stdout = StringIO()
        # known empty file. If NDBC replaces, this test may fail. 
        empty_file = '42008h1984.txt.gz'
        working_file = '46042h1996.txt.gz'
        filenames = pd.Series([empty_file, working_file])
        with contextlib.redirect_stdout(temp_stdout):
            ndbc_data =wave.io.ndbc.request_data('stdmet', filenames)        
        self.assertEqual(1, len(ndbc_data))              
        
    def test_ndbc_dates_to_datetime(self):
        dt = wave.io.ndbc.dates_to_datetime('swden', self.swden)
        self.assertEqual(datetime(1996, 1, 1, 1, 0), dt[1])
               
    def test_date_string_to_datetime(self):
        swden = self.swden.copy(deep=True)
        swden['mm'] = np.zeros(len(swden)).astype(int).astype(str)
        year_string='YY'
        year_fmt='%y'
        parse_columns = [year_string, 'MM', 'DD', 'hh', 'mm']
        df = wave.io.ndbc._date_string_to_datetime(swden, parse_columns, 
                                                   year_fmt) 
        dt = df['date']
        self.assertEqual(datetime(1996, 1, 1, 1, 0), dt[1])  
        
    def test_parameter_units(self):
        parameter='swden'
        units = wave.io.ndbc.parameter_units(parameter)
        self.assertEqual(units[parameter], '(m*m)/Hz')        

class TestWECSim(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass
            
    @classmethod
    def tearDownClass(self):
        pass

    ### WEC-Sim data, mo mooring
    def test_read_wecSim_no_mooring(self):
        ws_output = wave.io.wecsim.read_output(join(datadir, 'RM3_matlabWorkspace_structure.mat'))
        self.assertEqual(ws_output['wave'].elevation.name,'elevation')
        self.assertEqual(ws_output['bodies']['body1'].name,'float')
        self.assertEqual(ws_output['ptos'].name,'PTO1')        
        self.assertEqual(ws_output['constraints'].name,'Constraint1')
        self.assertEqual(len(ws_output['mooring']),0)
        self.assertEqual(len(ws_output['moorDyn']),0)
        self.assertEqual(len(ws_output['ptosim']),0)

    ### WEC-Sim data, with mooring
    def test_read_wecSim_with_mooring(self):
        ws_output = wave.io.wecsim.read_output(join(datadir, 'RM3MooringMatrix_matlabWorkspace_structure.mat'))
        self.assertEqual(ws_output['wave'].elevation.name,'elevation')
        self.assertEqual(ws_output['bodies']['body1'].name,'float')
        self.assertEqual(ws_output['ptos'].name,'PTO1')        
        self.assertEqual(ws_output['constraints'].name,'Constraint1')
        self.assertEqual(len(ws_output['mooring']),40001)
        self.assertEqual(len(ws_output['moorDyn']),0)
        self.assertEqual(len(ws_output['ptosim']),0)
        
    ### WEC-Sim data, with moorDyn
    def test_read_wecSim_with_moorDyn(self):
        ws_output = wave.io.wecsim.read_output(join(datadir, 'RM3MoorDyn_matlabWorkspace_structure.mat'))
        self.assertEqual(ws_output['wave'].elevation.name,'elevation')
        self.assertEqual(ws_output['bodies']['body1'].name,'float')
        self.assertEqual(ws_output['ptos'].name,'PTO1')        
        self.assertEqual(ws_output['constraints'].name,'Constraint1')
        self.assertEqual(len(ws_output['mooring']),40001)
        self.assertEqual(len(ws_output['moorDyn']),7)
        self.assertEqual(len(ws_output['ptosim']),0)

class TestWPTOhindcast(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.sy_swh = pd.read_csv(join(datadir,'single_year_hindcast.csv'),index_col = 'time_index',
        names = ['time_index','significant_wave_height_44.624076_-124.280097'],header = 0, 
        dtype = {'significant_wave_height_44.624076_-124.280097':'float32'})
        self.sy_swh.index = pd.to_datetime(self.sy_swh.index)

        self.my_swh = pd.read_csv(join(datadir,'multi_year_hindcast.csv'),index_col = 'time_index',
        names = ['time_index','significant_wave_height_44.624076_-124.280097'],header = 0, 
        dtype = {'significant_wave_height_44.624076_-124.280097':'float32'})
        self.my_swh.index = pd.to_datetime(self.my_swh.index)

        self.my2 = pd.read_csv(join(datadir,'multi_year_hindcast2.csv'),index_col = 'time_index',
        names = ['time_index','omni-directional_wave_power_44.624076_-124.280097'],header = 0, 
        dtype = {'omni-directional_wave_power_44.624076_-124.280097':'float32'})
        self.my2.index = pd.to_datetime(self.my2.index)

        self.sy_per = pd.read_csv(join(datadir,'single_year_hindcast_period.csv'),index_col = 'time_index',
        names = ['time_index','mean_absolute_period_44.624076_-124.280097'],header = 0, 
        dtype = {'mean_absolute_period_44.624076_-124.280097':'float32'})
        self.sy_per.index = pd.to_datetime(self.sy_per.index)

        self.ml = pd.read_csv(join(datadir,'single_year_hindcast_multiloc.csv'),index_col = 'time_index',
        names = ['time_index','mean_absolute_period_44.624076_-124.280097','mean_absolute_period_43.489171_-125.152137'],
        header = 0, dtype = {'mean_absolute_period_44.624076_-124.280097':'float32',
        'mean_absolute_period_43.489171_-125.152137':'float32'})
        self.ml.index = pd.to_datetime(self.ml.index)

        self.ml_meta = pd.read_csv(join(datadir,'multiloc_meta.csv'),index_col = 0,
        names = [None,'water_depth','latitude','longitude','distance_to_shore','timezone'
        ,'jurisdiction'],header = 0, dtype = {'water_depth':'float32','latitude':'float32'
        ,'longitude':'float32','distance_to_shore':'float32','timezone':'int16'})
        
            
    @classmethod
    def tearDownClass(self):
        pass

    ### WPTO hindcast data
    def test_single_year_sig_wave_height(self):
        single_year_waves = f'/nrel/US_wave/US_wave_1995.h5'
        lat_lon = (44.624076,-124.280097)
        parameters = 'significant_wave_height'

        wave_singleyear, meta = wave.io.wave_hindcast.read_US_wave_dataset(single_year_waves,parameters,lat_lon)
        assert_frame_equal(self.sy_swh,wave_singleyear)
        print(type(meta.jurisdiction.values))
        self.assertEqual(float(meta.water_depth),77.42949676513672)
        self.assertEqual(float(meta.latitude),44.624298095703125)
        self.assertEqual(float(meta.longitude),-124.27899932861328)
        self.assertEqual(float(meta.distance_to_shore),15622.17578125)
        self.assertEqual(float(meta.timezone),-8)
        self.assertEqual(meta.jurisdiction.values,"Federal")

    def test_single_year_mean_per(self):
        single_year_waves = f'/nrel/US_wave/US_wave_1995.h5'
        lat_lon = (44.624076,-124.280097)
        parameters = 'mean_absolute_period'

        wave_singleyear, meta = wave.io.wave_hindcast.read_US_wave_dataset(single_year_waves,parameters,lat_lon)
        assert_frame_equal(self.sy_per,wave_singleyear)

        self.assertEqual(float(meta.water_depth),77.42949676513672)
        self.assertEqual(float(meta.latitude),44.624298095703125)
        self.assertEqual(float(meta.longitude),-124.27899932861328)
        self.assertEqual(float(meta.distance_to_shore),15622.17578125)
        self.assertEqual(float(meta.timezone),-8)
        self.assertEqual(meta.jurisdiction.values,"Federal")

    def test_multi_year_sig_wave_height(self):
        multi_year_waves = f'/nrel/US_wave/US_wave_199*.h5'
        lat_lon = (44.624076,-124.280097) 
        parameters = 'significant_wave_height'

        wave_multiyear, meta = wave.io.wave_hindcast.read_US_wave_dataset(multi_year_waves,parameters,lat_lon)
        assert_frame_equal(self.my_swh,wave_multiyear)

        self.assertEqual(float(meta.water_depth),77.42949676513672)
        self.assertEqual(float(meta.latitude),44.624298095703125)
        self.assertEqual(float(meta.longitude),-124.27899932861328)
        self.assertEqual(float(meta.distance_to_shore),15622.17578125)
        self.assertEqual(float(meta.timezone),-8)
        self.assertEqual(meta.jurisdiction.values,"Federal")

    def test_multiyear_2(self):
        file = f'/nrel/US_wave/US_wave_*.h5' #specifying the years file of interest
        years = [1995,1996]
        parameter = 'omni-directional_wave_power'
        lat_lon = (44.624076,-124.280097) # setting lat/lon pair of interest
        odwp, meta= wave.io.wave_hindcast.read_US_wave_dataset(file,parameter,lat_lon,years=years)

        assert_frame_equal(self.my2,odwp)

        self.assertEqual(float(meta.water_depth),77.42949676513672)
        self.assertEqual(float(meta.latitude),44.624298095703125)
        self.assertEqual(float(meta.longitude),-124.27899932861328)
        self.assertEqual(float(meta.distance_to_shore),15622.17578125)
        self.assertEqual(float(meta.timezone),-8)
        self.assertEqual(meta.jurisdiction.values,"Federal")

    def test_multi_loc(self):
        single_year_waves = f'/nrel/US_wave/US_wave_1995.h5'
        lat_lon = ((44.624076,-124.280097),(43.489171,-125.152137)) 
        parameters = 'mean_absolute_period'

        wave_multiloc, meta= wave.io.wave_hindcast.read_US_wave_dataset(single_year_waves,
        parameters,lat_lon,hsds= True)

        assert_frame_equal(self.ml,wave_multiloc)
        assert_frame_equal(self.ml_meta,meta)
        

if __name__ == '__main__':
    unittest.main() 
