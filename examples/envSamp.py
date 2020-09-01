from mhkit.wave.resource import significant_wave_height, energy_period, \
                                environmental_contour
from mhkit.wave.graphics import plot_environmental_contour
import matplotlib.pyplot as plt
from mhkit.wave.io import ndbc
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import json

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
    #Hs.to_hdf(f'data/wave/Hm0_Te_{buoy_number}.h5',  key='df')
    #Hs.to_pickle(f'data/wave/Hm0_Te_{buoy_number}.pkl')
    
    Hs.to_json(f'data/wave/Hm0_Te_{buoy_number}.json')
    return Hs

#=======================================================================
# Get Spectral Wave Density & process data
#=======================================================================
buoy_number='46022'
try:
    #df_raw = pd.read_hdf(f'data/wave/Hm0_Te_{buoy_number}.h5', 'df')
    #df_raw = pd.read_pickle(f'data/wave/Hm0_Te_{buoy_number}.pkl' )
    df_raw = pd.read_json(f'data/wave/Hm0_Te_{buoy_number}.json' )
except:
    df_raw = get_Hm0_Te(buoy_number)

# Remove Outliers
df = df_raw[df_raw['Hm0'] < 20]

# Sea state duration (hrs)
# Delta time of sea-states 
dt_ss = (df.index[2]-df.index[1]).seconds  
# Return periods (yrs) of interest
time_R = 100  
Hm0_contour, Te_contour,PCA = environmental_contour(df.Hm0.values, df.Te.values, 
                                                dt_ss, time_R,return_PCA=True)

contours_46022_Hm0Te = pd.DataFrame.from_records(Hm0_contour.reshape(-1,1), 
                                                 columns=['Hm0_contour'] )
                                                  
contours_46022_Hm0Te['Te_contour'] =  Te_contour
 
contours_46022_Hm0Te.to_csv('data/wave/Hm0_Te_contours_46022.csv', index=False)


def save_dict(obj, name ):
    with open(f'data/wave/{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)
save_dict(PCA,'principal_component_analysis')
# PCA['principal_axes'] = PCA['principal_axes'].tolist()
# PCA['sigma_fit']['jac'] = PCA['sigma_fit']['jac'].tolist()
# PCA['sigma_fit']['x'] = PCA['sigma_fit']['x'].tolist()
# PCA['sigma_fit']['success'] = str(PCA['sigma_fit']['success'])
# with open('data/wave/principal_component_analysis.json', 'w') as fp: json.dump(PCA, fp)

plot_environmental_contour(df.Hm0.values, 
                           df.Te.values, 
						   Hm0_contour, 
						   Te_contour, 
                           data_label='NDBC 46022', 
						   contour_label='100 Year Contour',
						   ax=None)
plt.show()

import ipdb; ipdb.set_trace()



















