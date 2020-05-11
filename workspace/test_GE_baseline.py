# import statements
import pandas as pd 
import numpy as np 
from mhkit import utils
from mhkit import loadsKit as lk 
import matplotlib.pyplot as plt 
import os
from scipy.io import loadmat

# import txt file containing header and unit information
headers = pd.read_csv('workspace/data_GE/headers.txt',sep='\t',header=None)

# pre-allocate lists
means = []
maxs = []
mins = []
stdev = []
DELlist = []

# create tuple for fatigue
var_dict = [
    ('LSSDW_Tq',4),
    ('LSSDW_My',4),
    ('LSSDW_Mz',4),
    ('TTTq',4),
    ('TT_ForeAft',4),
    ('TT_SideSide',4),
    ('TB_ForeAft',4),
    ('TB_SideSide',4),
    ('BL1_FlapMom',10),
    ('BL1_EdgeMom',10)
]

# import mat files and calculate stats for each one
pathOut = 'workspace/data_GE'
for f in os.listdir(pathOut):
    if f.endswith('.mat'):
        # import mat file
        matfile = loadmat(pathOut+'/'+f)
        print('import complete: '+ f)
        # extract data from dict and turn into dataframe
        data = [[row.flat[0] for row in line] for line in matfile['data_new']]
        df = pd.DataFrame(data,columns=headers[0])
        # get stats
        means.append(df.mean().values.tolist())
        maxs.append(df.max().values.tolist())
        mins.append(df.min().values.tolist())
        stdev.append(df.std().values.tolist())

        # start fatigue calc
        dfDEL = lk.get_DEL_channels(df,var_dict)
        DELlist.extend(dfDEL.values.tolist()) 


dfmeans = pd.DataFrame(means,columns=df.columns.values)
dfmaxs = pd.DataFrame(maxs,columns=df.columns.values)
dfmins = pd.DataFrame(mins,columns=df.columns.values)
dfstd = pd.DataFrame(stdev,columns=df.columns.values)
dfDELs = pd.DataFrame(np.squeeze(DELlist),columns=dfDEL.columns.values)

plt.scatter(dfmeans.uWind_80m,dfDELs.LSSDW_My)
plt.xlim(0,25)
plt.grid()
plt.show()


print('done')