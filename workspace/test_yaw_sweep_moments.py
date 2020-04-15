# import statements
import pandas as pd 
import numpy as np 
from datetime import timedelta
import _utilities as util
import random
from scipy.io import loadmat
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt 

########## generate data

# read in .mat file
matfile = loadmat('C:/Users/hivanov/Desktop/yaw_sweeps/T2_FastData_2019_10_02_07_07_03_50Hz.mat')
data = [[row.flat[0] for row in line] for line in matfile['data_out'][0][0]]
tbaseB1 = data[6][0][2][0]
tbaseB2 = data[7][0][2][0]

# INPUTS
yaw = data[12][0][2][0] # yaw variable
overhang = 943.88 # overhang moment [kN-m]
df = pd.DataFrame(columns=['B1','B2']) # create df containing gage info
df.B1 = tbaseB1
df.B2 = tbaseB2
heading = 271.6 # heading of positive bridge heading relative to true north
# unwrap yaw vector
yaw = util.unwrapvec(yaw)

######## PROCESSING

# bin data and get average bin values
edges = np.asarray(range(361))
yaw_avg = binned_statistic(yaw,yaw,statistic='mean',bins=edges)
df_avg = pd.DataFrame(columns=df.columns.values)
for i in range(len(df.columns.values)):
    vals = binned_statistic(yaw,df.iloc[:,i],statistic='mean',bins=edges)
    df_avg.iloc[:,i] = vals.statistic

# find offset
offsets = df_avg.mean()

# create df of average values with zero mean
df_avg_0 = df_avg - offsets 

# find slope
slopes = overhang/((df_avg_0.max()+df_avg_0.min().abs())/2)

# scaled moments
df_moments = (df-offsets) * slopes

print('done')