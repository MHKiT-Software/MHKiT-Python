# import statements
import pandas as pd 
import numpy as np 
import _utilities as util
from mhkit import loadsKit as lk 
import matplotlib.pyplot as plt 
import os
import nptdms # 3rd party module used to read TDMS files
from scipy import signal

# inputs
pathOut = 'workspace/data_peetz/fastdata'

# preallocate
dfmeans = []
DELlist = []
time = []
# start loop
for f in os.listdir(pathOut):
    if f.endswith('.tdms'):
        # import tdms file
        data = nptdms.TdmsFile(pathOut+'/'+f)
        print('import complete: '+ f)
        # convert to a pandas dataframe
        df = data.as_dataframe()
        # fix column names to be lowercase with no spaces
        df.columns = df.columns.str.replace("/'FastData'/",'')
        df.columns = df.columns.str.replace("'",'')
        df.columns = df.columns.str.replace(" ",'_')
        df.columns = map(str.lower, df.columns)
        # convert time from excel format & rename column
        df['ms_excel_timestamp'] = pd.to_datetime('1899-12-30')+pd.to_timedelta(df['ms_excel_timestamp'],'D')
        df.rename(columns={'ms_excel_timestamp':'time'},inplace=True)
        # remove unwanted columns
        df = df.drop(columns=['scan_errors','late_scans','labview_timestamp'])
        df = df[df.columns.drop(list(df.filter(regex='droop_turns')))]
        df = df[df.columns.drop(list(df.filter(regex='analog')))]

        # start processing
        df.set_index('time',inplace=True)
        sMean,sMax,sMin,sStd = util.get_stats(df,50,period=600)
        dfmeans.extend(sMean.values.tolist())
        time.extend(sMean.index.values)
        
        # get DELs
        DEL = lk.get_DEL(df.t3_tb_bending_1,3)
        DELlist.append(DEL) 
                
        print('done')

dfmeans = pd.DataFrame(dfmeans,columns=df.columns.values,index=time)


# calculate PSD of last time series
f, pxx = signal.welch(df.t3_tb_bending_1,fs=50,window='hamming',nperseg=3750,return_onesided=True)

# plot the PSD
plt.semilogy(f,pxx)
plt.xlim([0,3])
plt.grid(alpha=0.7)
plt.show()

print('all done')