
import mhkit 

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import calendar

stn = '100'
startdate = "04/01/2012" # MM/DD/YYYY
enddate = "04/30/2012"

##################################
# Import Data as NetCDF from THREDDS URL
##################################

# Comment out the URL that you are not using

# CDIP Archived Dataset URL
data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'

# CDIP Realtime Dataset URL
# data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'

##################################
# Open Remote Dataset from CDIP THREDDS Server
##################################
nc = netCDF4.Dataset(data_url)

# Read Buoy Variables
ncTime = nc.variables['sstTime'][:]
timeall = [datetime.datetime.fromtimestamp(t) for t in ncTime] # Convert ncTime variable to datetime stamps
Hs = nc.variables['waveHs']
Tp = nc.variables['waveTp']
Dp = nc.variables['waveDp'] 


# Create a variable of the Buoy Name and Month Name, to use in plot title
buoyname = nc.variables['metaStationName'][:]
# buoytitle = " ".join(buoyname[:-40])
buoytitle = buoyname[:-40].data.tostring()

month_name = calendar.month_name[int(startdate[0:2])]
year_num = (startdate[6:10])

##################################
# Local Indexing Functions
##################################
# Find nearest value in numpy array
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# Convert from human-format to UNIX timestamp
def getUnixTimestamp(humanTime,dateFormat):
    unixTimestamp = int(time.mktime(datetime.datetime.strptime(humanTime, dateFormat).timetuple()))
    return unixTimestamp


##################################
# Time Index Values
##################################
unixstart = getUnixTimestamp(startdate,"%m/%d/%Y") 
neareststart = find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date

unixend = getUnixTimestamp(enddate,"%m/%d/%Y")
future = find_nearest(ncTime, unixend)  # Find the closest unix timestamp
futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date



##################################
# Plot Wave Time-Series
##################################
# Crete figure and specify subplot orientation (3 rows, 1 column), shared x-axis, and figure size
f, (pHs, pTp, pDp) = plt.subplots(3, 1, sharex=True, figsize=(15,10)) 


# Create 3 stacked subplots for three PARAMETERS (Hs, Tp, Dp)
pHs.plot(timeall[nearIndex:futureIndex],Hs[nearIndex:futureIndex],'b')
pTp.plot(timeall[nearIndex:futureIndex],Tp[nearIndex:futureIndex],'b')
pDp.scatter(timeall[nearIndex:futureIndex],Dp[nearIndex:futureIndex],color='blue',s=5) # Plot Dp variable as a scatterplot, rather than line

# Set Titles
plt.suptitle(buoytitle, fontsize=30, y=0.99)
plt.title(month_name + " " + year_num, fontsize=20, y=3.45)

# Set tick parameters
pHs.set_xticklabels(['1','6','11','16','21','26','31']) 
pHs.tick_params(axis='y', which='major', labelsize=12, right='off')
pHs.tick_params(axis='x', which='major', labelsize=12, top='off')

# Set x-axis tick interval to every 5 days
days = DayLocator(interval=5) 
daysFmt = DateFormatter('%d')
plt.gca().xaxis.set_major_locator(days)
plt.gca().xaxis.set_major_formatter(daysFmt)

# Label x-axis
plt.xlabel('Day', fontsize=18)

# Make a second y-axis for the Hs plot, to show values in both meters and feet
pHs2 = pHs.twinx()

# Set y-axis limits for each plot
pHs.set_ylim(0,8)
pHs2.set_ylim(0,25)
pTp.set_ylim(0,28)
pDp.set_ylim(0,360)

# Label each y-axis
pHs.set_ylabel('Hs, m', fontsize=18)
pHs2.set_ylabel('Hs, ft', fontsize=18)
pTp.set_ylabel('Tp, s', fontsize=18)
pDp.set_ylabel('Dp, deg', fontsize=18)

# Plot dashed gridlines
pHs.grid(b=True, which='major', color='b', linestyle='--')
pTp.grid(b=True, which='major', color='b', linestyle='--')
pDp.grid(b=True, which='major', color='b', linestyle='--')



