from os.path import abspath, dirname, join, isfile, normpath, relpath
from pandas.testing import assert_frame_equal
import xarray.testing as xrt
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
import netCDF4
import inspect
import pickle
import json
import sys
import os
import time
from random import seed, randint

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
        omega = np.arange(0.1,3.5,0.01)
        self.f = omega/(2*np.pi)
        self.Hs = 2.5
        self.Tp = 8
    
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
            k = wave.resource.wave_number(f, h, rho)
            calculated = k.loc[:,'k'].values
            error = ((expected-calculated)**2).sum() # SSE
            
            self.assertLess(error, 1e-6)

    def test_kfromw_one_freq(self):
        g = 9.81
        f = 0.1
        h = 1e9
        w = np.pi*2*f # deep water dispersion
        expected = w**2 / g
        calculated = wave.resource.wave_number(f=f, h=h, g=g).values[0][0]
        error = np.abs(expected-calculated)
        self.assertLess(error, 1e-6)
    
    def test_wave_length(self):
        k_list=[1,2,10,3]
        l_expected = (2.*np.pi/np.array(k_list)).tolist()
        
        k_df = pd.DataFrame(k_list,index = [1,2,3,4])
        k_series= k_df[0]
        k_array=np.array(k_list)
        
        for l in [k_list, k_df, k_series, k_array]:
            l_calculated = wave.resource.wave_length(l)            
            self.assertListEqual(l_expected,l_calculated.tolist())
        
        idx=0
        k_int = k_list[idx]
        l_calculated = wave.resource.wave_length(k_int)
        self.assertEqual(l_expected[idx],l_calculated)

    def test_depth_regime(self):
        expected = [True,True,False,True]
        l_list=[1,2,10,3]
        l_df = pd.DataFrame(l_list,index = [1,2,3,4])
        l_series= l_df[0]
        l_array=np.array(l_list)
        h = 10
        for l in [l_list, l_df, l_series, l_array]:
            calculated = wave.resource.depth_regime(l,h)            
            self.assertListEqual(expected,calculated.tolist())
        
        idx=0
        l_int = l_list[idx]
        calculated = wave.resource.depth_regime(l_int,h)
        self.assertEqual(expected[idx],calculated)
        

    def test_wave_celerity(self):
        # Depth regime ratio
        dr_ratio=2

        # small change in f will give similar value cg
        f=np.linspace(20.0001,20.0005,5)
        
        # Choose index to spike at. cg spike is inversly proportional to k
        k_idx=2
        k_tmp=[1, 1, 0.5, 1, 1]
        k = pd.DataFrame(k_tmp, index=f)
        
        # all shallow
        cg_shallow1 = wave.resource.wave_celerity(k, h=0.0001,depth_check=True)
        cg_shallow2 = wave.resource.wave_celerity(k, h=0.0001,depth_check=False)
        self.assertTrue(all(cg_shallow1.squeeze().values == 
                            cg_shallow2.squeeze().values))
        
        
        # all deep 
        cg = wave.resource.wave_celerity(k, h=1000,depth_check=True)
        self.assertTrue(all(np.pi*f/k.squeeze().values == cg.squeeze().values))
        
    def test_energy_flux_deep(self):
        # Dependent on mhkit.resource.BS spectrum
        S = wave.resource.bretschneider_spectrum(self.f,self.Tp,self.Hs)
        Te = wave.resource.energy_period(S)
        Hm0 = wave.resource.significant_wave_height(S)
        rho=1025
        g=9.80665
        coeff = rho*(g**2)/(64*np.pi)
        J = coeff*(Hm0.squeeze()**2)*Te.squeeze()
        
        h=-1 # not used when deep=True
        J_calc = wave.resource.energy_flux(S, h, deep=True)
        
        self.assertTrue(J_calc.squeeze() == J)


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

                # J
                if file_i != 'CDiP': 
                    for i,j in zip(data['h'],data['J']):
                        expected = data['J'][j]
                        calculated = wave.resource.energy_flux(S,i)
                        error = np.abs(expected-calculated.values)/expected
                        self.assertLess(error, 0.1)
                 

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
        self.assertAlmostEqual(PCA['x1_fit']['mu'], self.pca['x1_fit']['mu'])
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
        ndbc_data_file = join(datadir,'data.txt')
        [raw_ndbc_data, meta] = wave.io.ndbc.read_file(ndbc_data_file)
        self.S = raw_ndbc_data.T
        
        self.data = pd.DataFrame({'Hm0': Hm0, 'Te': Te, 'P': P,'J': J})
        self.Hm0_bins = np.arange(0,19,0.5)
        self.Te_bins = np.arange(0,9,1)
        self.expected_stats = ["mean","std","median","count","sum","min","max","freq"]

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

    def test_powerperformance_workflow(self):
        filename = abspath(join(testdir, 'Capture Length Matrix mean.png'))
        if isfile(filename):
            os.remove(filename)
        P = pd.Series(np.random.normal(200, 40, 743),index = self.S.columns)
        statistic = ['mean']
        savepath = testdir
        show_values = True
        h = 60
        expected = 401239.4822345051
        x = self.S.T
        CM,MAEP = wave.performance.power_performance_workflow(self.S, h, 
                        P, statistic, savepath=savepath, show_values=show_values)

        self.assertTrue(isfile(filename))
        self.assertEqual(list(CM.data_vars),self.expected_stats)

        error = (expected-MAEP)/expected # SSE
            
        self.assertLess(error, 1e-6)

    
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

    ### WPTO hindcast data
    # only run test for one version of python per to not spam the server
    # yet keep coverage high on each test
    if float(sys.version[0:3]) == 3.7:
        def test_multi_year(self):
            data_type = '3-hour'
            years = [1990,1992]
            lat_lon = (44.624076,-124.280097) 
            parameters = 'significant_wave_height'
            wave_multiyear, meta = wave.io.hindcast.request_wpto_point_data(data_type,parameters,lat_lon,years)
            assert_frame_equal(self.my_swh,wave_multiyear)
            assert_frame_equal(self.my_meta,meta)

    elif float(sys.version[0:3]) == 3.8:
        # wait five minute to ensure python 3.7 call is complete
        #time.sleep(300)
        def test_multi_loc(self):            
            data_type = '3-hour'
            years = [1995]
            lat_lon = ((44.624076,-124.280097),(43.489171,-125.152137)) 
            parameters = 'mean_absolute_period'
            wave_multiloc, meta= wave.io.hindcast.request_wpto_point_data(data_type,
            parameters,lat_lon,years)
            dir_multiyear, meta_dir = wave.io.hindcast.request_wpto_directional_spectrum(lat_lon,year='1995')
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

class TestSWAN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        swan_datadir = join(datadir,'swan')
        self.table_file = join(swan_datadir,'SWANOUT.DAT')
        self.swan_block_mat_file = join(swan_datadir,'SWANOUT.MAT')
        self.swan_block_txt_file = join(swan_datadir,'SWANOUTBlock.DAT')
        self.expected_table = pd.read_csv(self.table_file, sep='\s+', comment='%', 
                  names=['Xp', 'Yp', 'Hsig', 'Dir', 'RTpeak', 'TDir'])  
                  
    @classmethod
    def tearDownClass(self):
        pass

    def test_read_table(self):
        swan_table, swan_meta = wave.io.swan.read_table(self.table_file)
        assert_frame_equal(self.expected_table, swan_table)
        
    def test_read_block_mat(self):        
        swanBlockMat, metaDataMat = wave.io.swan.read_block(self.swan_block_mat_file )
        self.assertEqual(len(swanBlockMat), 4)
        self.assertAlmostEqual(self.expected_table['Hsig'].sum(), 
                               swanBlockMat['Hsig'].sum().sum(), places=1)
        
    def test_read_block_txt(self):        
        swanBlockTxt, metaData = wave.io.swan.read_block(self.swan_block_txt_file)
        self.assertEqual(len(swanBlockTxt), 4)
        sumSum = swanBlockTxt['Significant wave height'].sum().sum()
        self.assertAlmostEqual(self.expected_table['Hsig'].sum(), 
                               sumSum, places=-2)
                               
    def test_block_to_table(self):
        x=np.arange(5)
        y=np.arange(5,10)
        df = pd.DataFrame(np.random.rand(5,5), columns=x, index=y)
        dff = wave.io.swan.block_to_table(df)
        self.assertEqual(dff.shape, (len(x)*len(y), 3))
        self.assertTrue(all(dff.x.unique() == np.unique(x)))
        
    def test_dictionary_of_block_to_table(self):
        x=np.arange(5)
        y=np.arange(5,10)
        df = pd.DataFrame(np.random.rand(5,5), columns=x, index=y)
        keys = ['data1', 'data2']
        data = [df, df]
        dict_of_dfs = dict(zip(keys,data)) 
        dff = wave.io.swan.dictionary_of_block_to_table(dict_of_dfs) 
        self.assertEqual(dff.shape, (len(x)*len(y), 2+len(keys)))
        self.assertTrue(all(dff.x.unique() == np.unique(x)))
        for key in keys:
            self.assertTrue(key in dff.keys())
            
class TestIOcdip(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        b067_1996='http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/' + \
                   'archive/067p1/067p1_d04.nc'
        self.test_nc = netCDF4.Dataset(b067_1996)
       
        self.vars2D = [ 'waveEnergyDensity', 'waveMeanDirection', 
                        'waveA1Value', 'waveB1Value', 'waveA2Value', 
                        'waveB2Value', 'waveCheckFactor', 'waveSpread', 
                        'waveM2Value', 'waveN2Value'] 
        
    @classmethod
    def tearDownClass(self):
        pass
           
    def test_validate_date(self):
        date='2013-11-12'
        start_date = wave.io.cdip._validate_date(date)
        assert isinstance(start_date, datetime)        
        
        date='11-12-2012'
        self.assertRaises(ValueError, wave.io.cdip._validate_date, date)
        
    def test_request_netCDF_historic(self):
        station_number='067'
        nc = wave.io.cdip.request_netCDF(station_number, 'historic')
        isinstance(nc, netCDF4.Dataset)

    def test_request_netCDF_realtime(self):
        station_number='067'
        nc = wave.io.cdip.request_netCDF(station_number, 'realtime')
        isinstance(nc, netCDF4.Dataset)        

        
    def test_start_and_end_of_year(self):   
        year = 2020
        start_day, end_day = wave.io.cdip._start_and_end_of_year(year)
        
        assert isinstance(start_day, datetime)  
        assert isinstance(end_day, datetime)  
        
        expected_start = datetime(year,1,1)        
        expected_end = datetime(year,12,31)
        
        self.assertEqual(start_day, expected_start)
        self.assertEqual(end_day, expected_end)
        
    def test_dates_to_timestamp(self):   
    
        start_date='1996-10-02'
        end_date='1996-10-20'
    
        start_stamp, end_stamp = wave.io.cdip._dates_to_timestamp(self.test_nc, 
            start_date=start_date, end_date=end_date)
        
        start_dt =  datetime.utcfromtimestamp(start_stamp)
        end_dt =  datetime.utcfromtimestamp(end_stamp)
        
        self.assertTrue(start_dt.strftime('%Y-%m-%d') == start_date)
        self.assertTrue(end_dt.strftime('%Y-%m-%d') == end_date)
        
    def test_get_netcdf_variables_all2Dvars(self):
        data = wave.io.cdip.get_netcdf_variables(self.test_nc, 
            all_2D_variables=True)
        returned_keys = [key for key in data['data']['wave2D'].keys()]
        self.assertTrue( returned_keys == self.vars2D)
        
    def test_get_netcdf_variables_params(self):
        parameters =['waveHs', 'waveTp','notParam', 'waveMeanDirection']
        data = wave.io.cdip.get_netcdf_variables(self.test_nc, 
            parameters=parameters)        
        
        returned_keys_1D = [key for key in data['data']['wave'].keys()]
        returned_keys_2D = [key for key in data['data']['wave2D'].keys()]
        returned_keys_metadata = [key for key in data['metadata']['wave']]        

        self.assertTrue( returned_keys_1D == ['waveHs', 'waveTp'])
        self.assertTrue( returned_keys_2D == ['waveMeanDirection'])
        self.assertTrue( returned_keys_metadata == ['waveFrequency'])
        
        
    def test_get_netcdf_variables_time_slice(self):
        start_date='1996-10-01'
        end_date='1996-10-31'
                
        data = wave.io.cdip.get_netcdf_variables(self.test_nc,
                start_date=start_date, end_date=end_date,
                parameters='waveHs')        
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.assertTrue(data['data']['wave'].index[-1] < end_dt)
        self.assertTrue(data['data']['wave'].index[0] > start_dt)
        
        
    def test_request_parse_workflow_multiyear(self):
        station_number = '067'
        year1=2011
        year2=2013
        years = [year1, year2]
        parameters =['waveHs', 'waveMeanDirection', 'waveA1Value']
        data = wave.io.cdip.request_parse_workflow(station_number=station_number,
            years=years, parameters =parameters )
        
        expected_index0 = datetime(year1,1,1)   
        expected_index_final = datetime(year2,12,30) # last data on 30th
        
        wave1D = data['data']['wave']
        self.assertEqual(wave1D.index[0].floor('d').to_pydatetime(), expected_index0)

        self.assertEqual(wave1D.index[-1].floor('d').to_pydatetime(), expected_index_final) 
        
        for key,wave2D  in data['data']['wave2D'].items():
            self.assertEqual(wave2D.index[0].floor('d').to_pydatetime(), expected_index0)
            self.assertEqual(wave2D.index[-1].floor('d').to_pydatetime(), expected_index_final) 


    def test_plot_boxplot(self):            
        filename = abspath(join(testdir, 'wave_plot_boxplot.png'))
        if isfile(filename):
            os.remove(filename)
            
        station_number = '067'
        year = 2011
        data = wave.io.cdip.request_parse_workflow(station_number=station_number,years=year,
                       parameters =['waveHs'],
                       all_2D_variables=False)
                                 
        plt.figure()
        wave.graphics.plot_boxplot(data['data']['wave']['waveHs'])
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))            
        
        
    def test_plot_compendium(self):            
        filename = abspath(join(testdir, 'wave_plot_boxplot.png'))
        if isfile(filename):
            os.remove(filename)
            
        station_number = '067'
        year = 2011
        data = wave.io.cdip.request_parse_workflow(station_number=station_number,years=year,
                       parameters =['waveHs', 'waveTp', 'waveDp'],
                       all_2D_variables=False)
                                 
        plt.figure()
        wave.graphics.plot_compendium(data['data']['wave']['waveHs'], 
            data['data']['wave']['waveTp'], data['data']['wave']['waveDp'] )
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))  

class TestPlotResouceCharacterizations(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        f_name= 'Hm0_Te_46022.json'
        self.Hm0Te = pd.read_json(join(datadir,f_name)) 
    @classmethod
    def tearDownClass(self):
        pass
    def test_plot_avg_annual_energy_matrix(self):
    
        filename = abspath(join(testdir, 'avg_annual_scatter_table.png'))
        if isfile(filename):
            os.remove(filename)
        
        Hm0Te = self.Hm0Te
        Hm0Te.drop(Hm0Te[Hm0Te.Hm0 > 20].index, inplace=True)
        J = np.random.random(len(Hm0Te))*100 
        
        plt.figure()
        fig = wave.graphics.plot_avg_annual_energy_matrix(Hm0Te.Hm0, 
            Hm0Te.Te, J, Hm0_bin_size=0.5, Te_bin_size=1)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))  
        
    def test_plot_monthly_cumulative_distribution(self):
    
        filename = abspath(join(testdir, 'monthly_cumulative_distribution.png'))
        if isfile(filename):
            os.remove(filename)
            
        a = pd.date_range(start='1/1/2010',  periods=10000, freq='h')
        S = pd.Series(np.random.random(len(a)) , index=a)
        ax=wave.graphics.monthly_cumulative_distribution(S)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))        

if __name__ == '__main__':
    unittest.main() 
