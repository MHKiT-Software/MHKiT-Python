# import statements
import pandas as pd 
import numpy as np 
import _utilities as util
import _loadsKit as lk 
import matplotlib.pyplot as plt 


# import peetz 1min average stats from database
df = pd.read_csv('workspace/data_peetz/data.csv')
df.time = pd.to_datetime(df.time)

# generate statistics
sMean = util.get_stats(df,'time','mean',600,1/60)
sMax = util.get_stats(df,'time','max',600,1/60)
sMin = util.get_stats(df,'time','min',600,1/60)

# make plots of stats
util.statplotter(sMean.t3_wind_speed,sMean['t3_active_power'],sMax['t3_active_power'],sMin['t3_active_power'],ylabel='Power',xlabel='Wind speed')

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
plt.show()

print('done')