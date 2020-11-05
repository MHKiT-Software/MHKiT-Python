import pandas as pd
import numpy as np
from rex import WaveX, MultiYearWaveX

def read_US_wave_dataset(wave_path, parameter, lat_lon, tree=None, 
                                 unscale=True, str_decode=True, hsds=True):
    
        """
        Reads data from the WPTO wave hindcast data hosted on AWS. 

        Note: To access the WPTO hindcast data, you will need to configure h5pyd for data access on HSDS. 
        To get your own API key, visit https://developer.nrel.gov/signup/. 

        To configure h5phd type 
        hsconfigure
        and enter at the prompt:
        hs_endpoint = https://developer.nrel.gov/api/hsds
        hs_username = None
        hs_password = None
        hs_api_key = {your key}
        Parameters
        ----------
        wave_path : string
            Path to US_Wave .h5 files
            Available formats:
                /nrel/US_wave/US_wave$_{year}.h5
                /nrel/US_wave/virtual_buoy/US_virtual_buoy_{year}.h5
        parameter: string
            dataset parameter to be downloaded
            spatial dataset options: directionality_coefficient, energy_period, maximum_energy_direction
                mean_absolute_period, mean_zero-crossing_period, omni-directional_wave_power, peak_period
                significant_wave_height, spectral_width, water_depth 
            virtual buoy options: directional_wave_spectrum, directionality_coefficient
                energy_period, maximum_energy_direction, mean_absolute_period, mean_wave_direction
                mean_zero-crossing_period, omni_directional_wave_power, peak_period, significant_wave_height
                spectral_width, water_depth
        lat_lon: tuple or list of tuples
            latitude longitude pairs at which to extract data 
        tree : str | cKDTree (optional)
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool (optional)
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool (optional)
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool (optional)
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        ---------
        data: pandas DataFrame 
            Data indexed by datetime with columns named according to header row 
        """
        
        assert isinstance(parameter, str), 'parameter must be of type string'
        assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'

        waveKwargs = {'tree':tree,'unscale':unscale,'str_decode':str_decode, 'hsds':hsds}
        
        
        if isinstance(wave_path,list) or '*' in wave_path:
            rex_accessor = MultiYearWaveX
        else: 
            rex_accessor = WaveX
            
        with rex_accessor(wave_path, **waveKwargs) as waves:
            data = waves.get_lat_lon_df(parameter,lat_lon)
            col = data.columns[0]
            meta = waves.meta.loc[col,:]

        
        return data, meta