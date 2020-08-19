from mhkit.wave.PCA import principal_component_analysis, getContours
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


# Hs={}
# Te={}
# for year in ndbc_data:
    # year_data = ndbc_data[year]
    # Hs[year] = significant_wave_height(year_data.T)
    # Te[year] = energy_period(year_data.T)

  
# Hs_list = [ v for k,v in Hs.items()]; 
# Te_list = [ v for k,v in Te.items()]; 

# Hs= pd.concat(Hs_list ,axis=0)
# Te= pd.concat(Te_list ,axis=0)

# Hs['Te'] = Te.Te
# Hs.dropna(inplace=True)


df_raw = pd.read_pickle('46022HsTe.pkl')

# Remove Outliers
abs_z_scores = np.abs(stats.zscore(df_raw))
N_standard_deviations = 10
filtered_entries = (abs_z_scores < N_standard_deviations).all(axis=1)
df = df_raw[filtered_entries]
df = df.sort_index()
df = df[:'20121231']


PCA  = principal_component_analysis(df.Hm0.values, df.Te.values)

# Declare required parameters
time_SS = 1.  # Sea state duration (hrs)
time_R = 100  # Return periods (yrs) of interest
Hs_Return, T_Return = getContours(time_SS, time_R, PCA , nb_steps=1000)




# """
# Display a plot of the 100-year return contour, full sea state samples
# and contour samples
# """
plt.figure()
plt.plot(df.Te, df.Hm0, 'bo', alpha=0.1, label='NDBC data')
plt.plot(T_Return, Hs_Return, 'k-', label='100 year contour')
#plt.plot(self.T_SampleFSS, self.Hs_SampleFSS, 'ro', label='full sea state samples')
#plt.plot(self.T_SampleCA, self.Hs_SampleCA, 'y^', label='contour approach samples')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.xlabel('Energy period, $T_e$ [s]')
plt.ylabel('Sig. wave height, $H_s$ [m]')
plt.show()
import ipdb; ipdb.set_trace()

#for year in noaa_data_years:
#    print(np.diff([float(freq) for freq in noaa_data_years[year].columns.values]))
#Combine yearly data
#prep_data(noaa_data_years)

import WDRT.ESSC as ESSC
# Create buoy object, in this case for Station #44020
buoy46022 = ESSC.Buoy('46022', 'NDBC')
## Read data from ndbc.noaa.gov
#buoy46022.fetchFromWeb()
#buoy46022.saveAsH5('NDBC46022.h5')
buoy46022.loadFromH5('NDBC46022.h5')


import ipdb; ipdb.set_trace()