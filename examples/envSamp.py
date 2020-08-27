from mhkit.wave.resource import significant_wave_height, energy_period, \
                                principal_component_analysis,  \
                                environmental_contour
import matplotlib.pyplot as plt
from mhkit.wave.io import ndbc
from scipy import stats
import pandas as pd
import numpy as np

def get_Hm0_Te(buoy_number):
    '''
    For the selected buoy number this function will request all available
    historical wave spectral density data. Then the script will modify 
    the data and calculate Hm0 and Te and save the dataframe as a pickle
    '''
    # Set the parameter to be spectral wave density
    spectral_wave_density='swden'

    # Get the NDBC data units
    units = ndbc.parameter_units(spectral_wave_density)

    # Find available parameter data for NDBC buoy_number
    available_data= ndbc.available_data(spectral_wave_density, 
                                        buoy_number)

    # Get dictionary of parameter data by year
    filenames= available_data['filename']
    ndbc_data = ndbc.request_data(spectral_wave_density, filenames)


    # Create a Datetime Index and remove NOAA date columns for each year
    for year in ndbc_data:
       print(year)
       year_data = ndbc_data[year]
       year_data['date'], ndbc_date_cols = ndbc.dates_to_datetime(spectral_wave_density, 
                                                                  year_data, 
                                                                  return_date_cols=True)
       year_data = year_data.drop(ndbc_date_cols, axis=1)
       year_data = year_data.set_index('date')
       # Convert columns to float now that the ndbc_date_cols (type=str) are gone
       year_data.columns = year_data.columns.astype(float)     
       ndbc_data[year] = year_data


    #=======================================================================
    # Calculate Hm0 and Te for each year
    #=======================================================================
    Hs={}
    Te={}
    for year in ndbc_data:
        year_data = ndbc_data[year]
        Hs[year] = significant_wave_height(year_data.T)
        Te[year] = energy_period(year_data.T)

      
    Hs_list = [ v for k,v in Hs.items()]; 
    Te_list = [ v for k,v in Te.items()]; 

    Hs= pd.concat(Hs_list ,axis=0)
    Te= pd.concat(Te_list ,axis=0)

    Hs['Te'] = Te.Te
    Hs.dropna(inplace=True)

    # Save the data locally
    Hs.to_pickle(f'{buoy_number}Hm0Te.pkl')
    return Hs

#=======================================================================
# Get Spectral Wave Density & process data
#=======================================================================
buoy_number='46022'
try:
    df_raw = pd.read_pickle(f'{buoy_number}Hm0Te.pkl')
except:
   df_raw = get_Hm0_Te(buoy_number)

# Remove Outliers
abs_z_scores = np.abs(stats.zscore(df_raw))
N_standard_deviations = 4
filtered_entries = (abs_z_scores < N_standard_deviations).all(axis=1)
df = df_raw[filtered_entries]
df = df.sort_index()
df = df[:'20121231']


PCA  = principal_component_analysis(df.Hm0.values, df.Te.values)

# Declare required parameters
time_SS = 1.  # Sea state duration (hrs)
time_R = 100  # Return periods (yrs) of interest
Hs_Return, T_Return = environmental_contour(time_SS, time_R, PCA , nb_steps=1000)




# """
# Display a plot of the 100-year return contour, full sea state samples
# and contour samples
# """

plt.figure()

plt.plot(df.Te, df.Hm0, 'bo', alpha=0.1, label='NDBC data')
plt.plot(T_Return, Hs_Return, 'k-', label='100 year contour')

plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.xlabel('Energy period, $T_e$ [s]')
plt.ylabel('Sig. wave height, $H_s$ [m]')
plt.show()
import ipdb; ipdb.set_trace()



















