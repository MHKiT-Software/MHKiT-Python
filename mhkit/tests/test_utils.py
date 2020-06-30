import unittest
import numpy as np
import pandas as pd
import mhkit.utils as utils
from pandas.testing import assert_frame_equal

class TestGenUtils(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.filepath = 'data/data_loads.csv'
        self.freq = 50 # Hz
        self.period = 600 # seconds

    def test_get_statistics(self):
        # load in file
        df = pd.read_csv(self.filepath)
        df.Timestamp = pd.to_datetime(df.Timestamp)
        df.set_index('Timestamp',inplace=True)
        # run function
        means,maxs,mins,stdevs = utils.get_statistics(df,self.freq,period=self.period)
        # check statistics
        self.assertAlmostEqual(means['uWind_80m'],7.773,2) # mean
        self.assertAlmostEqual(maxs['uWind_80m'],13.271,2) # max
        self.assertAlmostEqual(mins['uWind_80m'],3.221,2) # min
        self.assertAlmostEqual(stdevs['uWind_80m'],1.551,2) # standard deviation
        # check timestamp
        string_time = '2017-03-01 01:28:41'
        time = pd.to_datetime(string_time)
        self.assertTrue(means.index[0]==time)
        
    def test_unwrap_vector(self):
        # create array of test values and corresponding expected answers
        test = [-740,-400,-50,0,50,400,740]
        correct = [340,320,310,0,50,40,20]
        # get answers from function
        answer = utils.unwrap_vector(test)
        
        # check if answer is correct
        assert_frame_equal(pd.DataFrame(answer),pd.DataFrame(correct))

    def test_matlab_to_datetime(self):
        # store excel timestamp
        mat_time = 7.367554921296296e+05
        # corresponding datetime
        string_time = '2017-03-01 11:48:40'
        time = pd.to_datetime(string_time)
        # test function
        answer = utils.matlab_to_datetime(mat_time)
        answer2 = answer.round('s') # round to nearest second for comparison
        
        # check if answer is correct
        self.assertTrue(answer2 == time)

    def test_excel_to_datetime(self):
        # store excel timestamp
        excel_time = 4.279549212962963e+04
        # corresponding datetime
        string_time = '2017-03-01 11:48:40'
        time = pd.to_datetime(string_time)
        # test function
        answer = utils.excel2datetime(excel_time)
        answer2 = answer.round('s') # round to nearest second for comparison
        
        # check if answer is correct
        self.assertTrue(answer2 == time)


