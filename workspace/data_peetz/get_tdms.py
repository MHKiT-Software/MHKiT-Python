# import
import nptdms
import pandas as pd 
import os

# path where data resides
pathOut = 'workspace/data_peetz/tdmsfiles/'

# define start and end days
folders = os.listdir(pathOut)

dfall = pd.DataFrame() 

for day in folders:
    for f in os.listdir(pathOut+day):
        if f.endswith(".tdms"):
            # import tdms file
            data = nptdms.TdmsFile(pathOut+day+'/'+f)
            print('import complete: '+ f)

            # convert to a pandas dataframe
            df = data.as_dataframe()

            # fix column names to be lowercase with no spaces
            df.columns = df.columns.str.replace("/'SlowData'/",'')
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
    
    # add data to main dataframe
    
    dfall = pd.concat([dfall,df],axis=0)        

dfall.to_csv('workspace/data_peetz/peetzSlow.csv',index=False)

print('done')