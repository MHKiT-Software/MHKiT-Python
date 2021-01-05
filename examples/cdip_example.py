from mhkit.wave.io import cdip
import matplotlib.pyplot as plt
import mhkit 

########################################################################
# Boxplot Exampe
########################################################################
# Load Historic data
station_number = '067'
year_date = '2011'

data = cdip.request_historic(station_number,year=year_date)

# Plot Boxplot
mhkit.wave.graphics.plot_boxplot(data, data.name)
#Kelley FIX 
# Boxplot scatter plot of means not properly alligned
# Boxplot doens't matach CDIP example
import ipdb; ipdb.set_trace()
########################################################################
# Compendium Example
########################################################################

#########################
# Load Historic data
#########################
stn = '100'
start_date = "04/01/2012" # MM/DD/YYYY
end_date = "04/30/2012"

# stn = '179'
# start_date = "04/01/2019" # MM/DD/YYYY
# end_date = "04/30/2019"

# stn = '187'
# start_date = "08/01/2018" # MM/DD/YYYY
# end_date = "08/31/2018"

# stn = '213'
# start_date = "08/01/2018" # MM/DD/YYYY
# end_date = "08/31/2018"

data = cdip.request_historic(stn,start_date= start_date,end_date=start_date)
# data.head()

#########################
# Realtime data
#########################
# stn = '187'
# start_date = "05/01/2020" # MM/DD/YYYY
# end_date = "05/31/2020"

# [data, buoytitle] = mhkit.wave.io.cdip.request_data(stn,start_date,end_date,data_type='Realtime')
# data.head()

#########################
# Plot data Compendium
#########################
mhkit.wave.graphics.plot_compendium(data, data.name)

import ipdb; ipdb.set_trace()


