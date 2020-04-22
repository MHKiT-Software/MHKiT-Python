# import statements
import pandas as pd 
import numpy as np 
import _utilities as util
import _loadsKit as lk 
import matplotlib.pyplot as plt 
import mhkit

# import peetz 1min average stats from database
df = pd.read_csv('workspace/data_peetz/peetzSlow.csv')
df.time = pd.to_datetime(df.time)
df.set_index('time',inplace=True)

# generate statistics
sMean,sMax,sMin,sStdev = util.get_stats(df,1,period=600)

# make plots of stats
lk.statplotter(sMean.t3_wind_speed,sMean['t3_active_power'],sMax['t3_active_power'],sMin['t3_active_power'],ylabel='Power',xlabel='Wind speed',title='Raw Stats')

# bin data
bmeans,bmstd = lk.bin_stats(sMean.t3_wind_speed,sMean.t3_active_power,20,bwidth=1)
bmax,_ = lk.bin_stats(sMean.t3_wind_speed,sMax.t3_active_power,20,bwidth=1)
bmin,_ = lk.bin_stats(sMean.t3_wind_speed,sMin.t3_active_power,20,bwidth=1)

# make plot of binned data
x = np.linspace(0.5,18.5,19)
plt.plot(x,bmeans)
plt.errorbar(x,bmeans.loc[:,0],yerr=bmstd.loc[:,0],fmt='none',capsize=4,ecolor='black')
plt.plot(x,bmax)
plt.plot(x,bmin)
plt.grid(alpha=0.5)
plt.title('Binned Stats')
plt.xlabel('Wind speed')
plt.ylabel('Power')
plt.show()

# apply slope and offset to tower base bending
df.t3_tb_bending_1 = (df.t3_tb_bending_1 - 1.7553e-04)*4.2143e07
df.t3_tb_bending_2 = (df.t3_tb_bending_2 - 1.4968e-04)*4.4831e07

# get DELs by looping through 10 minute periods of data
# step = 600
# DELs = []
# statcheck = []
# begin = int(df.index.values[df['time'].dt.round('1s')==sMean.index[0]])
# ending = int(df.index.values[df['time'].dt.round('1s')==sMean.index[-1]])
# for d in range(int(np.floor(ending/step))):
#     start = 600 * d
#     end = (600*(d+1))-1
#     DEL = lk.get_DEL(df.loc[start:end,'t3_tb_bending_1'],3)
#     DELs.append(DEL)

# # bind DELs by wind speed
# ws = sMean.t3_wind_speed.dropna()
# DELs = pd.Series(DELs)
# bDELs,bDstd = lk.bin_stats(ws.iloc[0:-2],DELs,20,bwidth=1)

# # make plot of binned data
# x = np.linspace(0.5,18.5,19)
# plt.plot(x,bDELs)
# #plt.errorbar(x,bmeans.loc[:,0],yerr=bmstd.loc[:,0],fmt='none',capsize=4,ecolor='black')
# #plt.plot(x,bmax)
# #plt.plot(x,bmin)
# plt.grid(alpha=0.5)
# plt.title('Binned Stats')
# plt.xlabel('Wind speed')
# plt.ylabel('DELs')
# plt.show()


print('done')