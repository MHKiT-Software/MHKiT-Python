# import statements
import pandas as pd 
import numpy as np 
import _utilities as util
import _loadsKit as lk 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import datetime as dt

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac


# import matlab data and convert to pandas
matfile = loadmat('workspace/data_mooring/AzLoadCellData105.mat')
data = [[row.flat[0] for row in line] for line in matfile['AzLoadCellData105'][0][0]]
time = data[4][0][0][0]
L1 = data[4][0][1][0]
L3 = data[4][0][3][0]
dfraw = pd.DataFrame({'time':time,'L1':L1,'L3':L3})
dfraw.time = dfraw.time.apply(matlab2datetime) # convert to datetime

# resample to 10 min data 
dfmean = dfraw.resample('10min',on='time').mean()
dfmax = dfraw.resample('10min',on='time').max()
dfmin = dfraw.resample('10min',on='time').min()

# TODO: turn this into function
# TODO: add functionality about what to do with intermittent data

# get DELs by looping through 10 minute periods of data
step = 600*50
DELs = []
for d in range(int(len(dfraw)/step)):
    start = step * d
    end = (step*(d+1))-1
    DEL = lk.get_DEL(dfraw.loc[start:end,'L1'],3)
    DELs.append(DEL)

print('done')