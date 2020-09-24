import mhkit 
import matplotlib
import matplotlib.pyplot as plt

stn = '100'
startdate = "04/01/2012" # MM/DD/YYYY
enddate = "04/30/2012"

[data, buoytitle] = mhkit.wave.io.cdip.request_data(stn,startdate,enddate)
data.head()

##################################
# Plot Wave Time-Series
##################################
# Create figure and specify subplot orientation (3 rows, 1 column), shared x-axis, and figure size
f, (pHs, pTp, pDp) = plt.subplots(3, 1, sharex=True, figsize=(15,10)) 

# Create 3 stacked subplots for three PARAMETERS (Hs, Tp, Dp)
pHs.plot(data.index,data.Hs,'b')
pTp.plot(data.index,data.Tp,'b')
pDp.scatter(data.index,data.Dp,color='blue',s=5) # Plot Dp variable as a scatterplot, rather than line

# Significant Wave Height, Hs
pHs.tick_params(axis='x', which='major', labelsize=12, top='off')
pHs.set_xticklabels(['3','8','13','18','23','28']) 
pHs.set_ylim(0,8)
pHs.tick_params(axis='y', which='major', labelsize=12, right='off')
pHs.set_ylabel('Hs, m', fontsize=18)
pHs.grid(b=True, which='major', color='b', linestyle='--')
# Make a second y-axis for the Hs plot, to show values in both meters and feet
pHs2 = pHs.twinx()
pHs2.set_ylim(0,25)
pHs2.set_ylabel('Hs, ft', fontsize=18)

# Peak Period, Tp
pTp.set_ylim(0,28)
pTp.set_ylabel('Tp, s', fontsize=18)
pTp.grid(b=True, which='major', color='b', linestyle='--')

# Direction, Dp 
pDp.set_ylim(0,360)
pDp.set_ylabel('Dp, deg', fontsize=18)
pDp.grid(b=True, which='major', color='b', linestyle='--')
pDp.set_xlabel('Day', fontsize=18)

# Set x-axis tick interval to every 5 days
days = matplotlib.dates.DayLocator(interval=5) 
daysFmt = matplotlib.dates.DateFormatter('%d')
plt.gca().xaxis.set_major_locator(days)
plt.gca().xaxis.set_major_formatter(daysFmt)

# Set Titles
month_name = data.index.month_name()[-1]
year_num = data.index.year[-1]
plt.suptitle(buoytitle, fontsize=30) 
plt.title(month_name + " " + str(year_num), fontsize=20)


