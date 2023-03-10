import numpy as np
import pandas as pd
from rex import MultiYearWaveX, WaveX
import sys
from time import sleep



def region_selection(lat_lon):
    '''
    Returns the name of the predefined region in which the given coordinates reside.
    Can be used to check if the passed lat/lon pair is within the WPTO hindcast dataset. 

    Parameters
    ----------
    lat_lon : list or tuple
        Latitude and longitude coordinates as flaots or integers
    
    Returns
    -------
    region : string
        Name of predefined region for given coordinates
    '''
    assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'
    assert isinstance(lat_lon[0], (float,int)), 'lat_lon  values must be of type float or int'
    assert isinstance(lat_lon[1], (float,int)), 'lat_lon  values must be of type float or int'

    rDict = {
        'Hawaii':{      'lat':[15.0,27.000002], 'lon':[-164.0,-151.0] },
        'West_Coast':{  'lat':[30.0906, 48.8641], 'lon':[-130.072, -116.899] },
        'Atlantic':{    'lat':[24.382, 44.8247], 'lon':[-81.552, -65.721] },
    }

    region_search = lambda x: all( ( True if rDict[x][dk][0] <= d <= rDict[x][dk][1] else False
                                    for dk, d in {'lat':lat_lon[0],'lon':lat_lon[1]}.items() ) )
    region = [key for key in rDict if region_search(key)]
    
    if len(region)==0:
        print('ERROR: coordinates out of bounds')
    else:
        return region[0]



def request_wpto_point_data(data_type, parameter, lat_lon, years, tree=None, 
                                 unscale=True, str_decode=True,hsds=True):
    
        """ 
        Returns data from the WPTO wave hindcast hosted on AWS at the specified latitude and longitude point(s), 
        or the closest available point(s).
        Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more information about the dataset and available 
        locations and years. 

        Note: To access the WPTO hindcast data, you will need to configure h5pyd for data access on HSDS. 
        Please see the WPTO_hindcast_example notebook for more information.  

        Parameters
        ----------
        data_type : string
            Data set type of interest
            Options: '3-hour' '1-hour'
        parameter: string or list of strings
            Dataset parameter to be downloaded
            3-hour dataset options: 'directionality_coefficient', 'energy_period', 'maximum_energy_direction'
                'mean_absolute_period', 'mean_zero-crossing_period', 'omni-directional_wave_power', 'peak_period'
                'significant_wave_height', 'spectral_width', 'water_depth' 
            1-hour dataset options: 'directionality_coefficient', 'energy_period', 'maximum_energy_direction'
                'mean_absolute_period', 'mean_zero-crossing_period', 'omni-directional_wave_power', 'peak_period'
                'significant_wave_height', 'spectral_width', 'water_depth', 'maximim_energy_direction',
                'mean_wave_direction', 'frequency_bin_edges'
        lat_lon: tuple or list of tuples
            Latitude longitude pairs at which to extract data 
        years : list 
            Year(s) to be accessed. The years 1979-2010 available. Examples: [1996] or [2004,2006,2007]
        tree : str | cKDTree (optional)
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, default = None
        unscale : bool (optional)
            Boolean flag to automatically unscale variables on extraction
            Default = True
        str_decode : bool (optional)
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            Default = True
        hsds : bool (optional)
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS. Setting to False will indicate to look for files on 
            local machine, not AWS. Default = True

        Returns
        ---------
        data: DataFrame 
            Data indexed by datetime with columns named for parameter and cooresponding metadata index 
        meta: DataFrame 
            Location metadata for the requested data location   
        """
        
        assert isinstance(parameter, (str, list)), 'parameter must be of type string or list'
        assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'
        assert isinstance(data_type, str), 'data_type must be a string'
        assert isinstance(years,list), 'years must be a list'
        assert isinstance(tree,(str,type(None))), 'tree must be a string'
        assert isinstance(unscale,bool), 'unscale must be bool type'
        assert isinstance(str_decode,bool), 'str_decode must be bool type'
        assert isinstance(hsds,bool), 'hsds must be bool type'

        if 'directional_wave_spectrum' in parameter:
            sys.exit('This function does not support directional_wave_spectrum output')

        # check for multiple region selection
        if isinstance(lat_lon[0], float):
            region = region_selection(lat_lon)
        else:
            reglist = []
            for loc in lat_lon:
                reglist.append(region_selection(loc))
            if reglist.count(reglist[0]) == len(lat_lon):
                region = reglist[0]
            else:
                sys.exit('Coordinates must be within the same region!')

        if data_type == '3-hour':
            wave_path = f'/nrel/US_wave/'+region+'/'+region+'_wave_*.h5'
        elif data_type == '1-hour':
            wave_path = f'/nrel/US_wave/virtual_buoy/'+region+'/'+region+'_virtual_buoy_*.h5'
        else:
            print(f'ERROR: invalid data_type')
            pass
        waveKwargs = {'tree':tree,'unscale':unscale,'str_decode':str_decode, 'hsds':hsds,
            'years':years}
        data_list = []
        
        with MultiYearWaveX(wave_path, **waveKwargs) as rex_waves:
            if isinstance(parameter, list):
                for p in parameter:
                    temp_data = rex_waves.get_lat_lon_df(p,lat_lon)
                    col = temp_data.columns[:]
                    for i,c in zip(range(len(col)),col):
                        temp = f'{p}_{i}'
                        temp_data = temp_data.rename(columns={c:temp})

                    data_list.append(temp_data)
                data= pd.concat(data_list, axis=1)
                
            else:
                data = rex_waves.get_lat_lon_df(parameter,lat_lon)
                col = data.columns[:]

                for i,c in zip(range(len(col)),col):
                    temp = f'{parameter}_{i}'
                    data = data.rename(columns={c:temp})

            meta = rex_waves.meta.loc[col,:]
            meta = meta.reset_index(drop=True)    
        return data, meta

def request_wpto_directional_spectrum(lat_lon, year, tree=None, 
                                 unscale=True, str_decode=True,hsds=True):
    
    """ 
    Returns directional spectra data from the WPTO wave hindcast hosted on AWS at the specified latitude and longitude point(s), 
    or the closest available point(s).
    Visit https://registry.opendata.aws/wpto-pds-us-wave/ for more information about the dataset and available 
    locations and years. 

    Note: To access the WPTO hindcast data, you will need to configure h5pyd for data access on HSDS. 
    Please see the WPTO_hindcast_example notebook for more information.  

    Parameters
    ----------
    lat_lon: tuple or list of tuples
        Latitude longitude pairs at which to extract data 
    year : string 
        Year to be accessed. The years 1979-2010 available. Only one year can be requested at a time. 
    tree : str | cKDTree (optional)
        cKDTree or path to .pkl file containing pre-computed tree
        of lat, lon coordinates, default = None
    unscale : bool (optional)
        Boolean flag to automatically unscale variables on extraction
        Default = True
    str_decode : bool (optional)
        Boolean flag to decode the bytestring meta data into normal
        strings. Setting this to False will speed up the meta data read.
                Default = True
    hsds : bool (optional)
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS. Setting to False will indicate to look for files on 
        local machine, not AWS. Default = True

    Returns
    ---------
    data: xarray 
        Coordinates as datetime, frequency, and direction for data at specified location(s)
    meta: DataFrame 
        Location metadata for the requested data location   
    """
    assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'
    assert isinstance(year,str), 'years must be a string'
    assert isinstance(tree,(str,type(None))), 'tree must be a sring'
    assert isinstance(unscale,bool), 'unscale must be bool type'
    assert isinstance(str_decode,bool), 'str_decode must be bool type'
    assert isinstance(hsds,bool), 'hsds must be bool type'

    # check for multiple region selection
    if isinstance(lat_lon[0], float):
        region = region_selection(lat_lon)
    else:
        reglist = []
        for loc in lat_lon:
            reglist.append(region_selection(loc))
        if reglist.count(reglist[0]) == len(lat_lon):
            region = reglist[0]
        else:
            sys.exit('Coordinates must be within the same region!')

    wave_path = f'/nrel/US_wave/virtual_buoy/'+region+'/'+region+'_virtual_buoy_'+year+'.h5'
    parameter = 'directional_wave_spectrum'
    waveKwargs = {'tree':tree,'unscale':unscale,'str_decode':str_decode, 'hsds':hsds}
        
    with WaveX(wave_path, **waveKwargs) as rex_waves:
        # Get graphical identifier
        gid = rex_waves.lat_lon_gid(lat_lon)  
        
        # Setup index and columns
        if isinstance(gid, (int, np.integer)):
            columns = [gid]
        else:
            columns = gid   
        time_index = rex_waves.time_index
        frequency = rex_waves['frequency']
        direction = rex_waves['direction']
        index = pd.MultiIndex.from_product(
            [time_index, frequency, direction],
            names=['time_index', 'frequency', 'direction']
        )                    

        # Create bins for multiple smaller API dataset requests
        N=6
        length = len(rex_waves)        
        quotient=length//N
        remainder=length%N    
        bins = [i*quotient for i in range(N+1) ]
        bins[-1]+= remainder        
        index_bins = (np.array(bins)*len(frequency)*len(direction)).tolist()
        
        # Request multiple datasets and add to dictionary
        datas={}
        for i in range(len(bins)-1):
            idx=index[index_bins[i]:index_bins[i+1]] 

            # Request with exponential back off wait time
            sleep_time = 2
            num_retries = 4
            for x in range(0, num_retries):  
                try:
                    data_array = rex_waves[parameter, bins[i]:bins[i+1], :, :, gid]
                    str_error = None

                except Exception as e:
                    str_error = str(e)

                if str_error:
                    sleep(sleep_time)
                    sleep_time *= 2  
                else:
                    break
        
            ax1 = np.product(data_array.shape[:3])
            ax2 = data_array.shape[-1] if len(data_array.shape) == 4 else 1
            data_array = data_array.reshape(ax1, ax2)        
            
            df = pd.DataFrame(data_array, columns=columns, index=idx)
            df.name = parameter
            datas[i]=df

        # Append each request into an xarray
        data_raw=datas[0]
        for i in list(datas.keys())[1:]:
            data_raw =  pd.concat([data_raw,datas[i]])                
        data = data_raw.to_xarray().to_array().drop('variable').squeeze()
        data['time_index'] = pd.to_datetime(data.time_index)

        # Get metadata
        meta = rex_waves.meta.loc[columns,:]
        meta = meta.reset_index(drop=True)  

    return data, meta