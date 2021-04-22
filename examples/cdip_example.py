from mhkit.wave.io import cdip
import matplotlib.pyplot as plt
import mhkit 


station_number='100'
data_type='historic'
nc = cdip.request_netCDF(station_number, data_type)
data = cdip.get_netcdf_variables(nc, )
import ipdb; ipdb.set_trace() 

parameters =['waveHs', 'waveTp', 'waveMeanDirection']







########################################################################
# Boxplot Exampe
########################################################################
# Load Historic data
station_number = '067'
year = 2011

#data = cdip.parse_data(station_number=station_number,years=year,
#                                   all_2D_variables=False)

data = cdip.parse_data(station_number=station_number,years=year,
                       parameters =['waveHs', 'waveTp','notParam', 'waveMeanDirection'],
                                   all_2D_variables=False)
import ipdb; ipdb.set_trace()                                   
# Plot Boxplot
#mhkit.wave.graphics.plot_boxplot(data.waveHs, data.name)
#Kelley FIX 
# Boxplot scatter plot of means not properly alligned
# Boxplot doens't matach CDIP example
#import ipdb; ipdb.set_trace()
########################################################################
# Compendium Example
########################################################################

#########################
# Load Historic data
#########################
stn = '100'
start_date = "2012-04-01" 
end_date = "2012-04-30"

# stn = '179'
# start_date = "04/01/2019" 
# end_date = "04/30/2019"

# stn = '187'
# start_date = "08/01/2018" 
# end_date = "08/31/2018"

# stn = '213'
# start_date = "08/01/2018" 
# end_date = "08/31/2018"

#data, metadata = cdip.request_data(stn,start_date=start_date,end_date=end_date)
# data.head()

#########################
# Realtime data
#########################
stn = '433'
start_date = "2020-11-01" 
end_date = "2020-11-30"

data, metadata = cdip.request_data(stn,start_date=start_date,
                                   end_date=end_date,data_type='Realtime')
#import ipdb; ipdb.set_trace()

#########################
# Plot data Compendium
#########################
mhkit.wave.graphics.plot_compendium(data.waveHs, data.waveTp, 
                                    data.waveDp, data.name)

import ipdb; ipdb.set_trace()


