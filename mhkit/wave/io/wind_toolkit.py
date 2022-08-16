import pandas as pd
from rex import MultiYearWindX


def region_selection(lat_lon, preferred_region):
    '''
    Returns the name of the predefined region in which the given coordinates reside.
    Can be used to check if the passed lat/lon pair is within the WIND Toolkit hindcast dataset. 

    Parameters
    ----------
    lat_lon : list or tuple
        Latitude and longitude coordinates as floats or integers
        
    preferred_region : string
        Latitude and longitude coordinates as floats or integers
    
    Returns
    -------
    region : string
        Name of predefined region for given coordinates
    '''
    assert isinstance(lat_lon, (list,tuple)), 'lat_lon must be of type list or tuple'
    assert isinstance(lat_lon[0], (float,int)), 'lat_lon values must be of type float or int'
    assert isinstance(lat_lon[1], (float,int)), 'lat_lon values must be of type float or int'
    assert isinstance(preferred_region, str), 'preferred_region must be of type string'
    
    rDict = {
        'CA_NWP_overlap':{'lat':[41.213, 42.642], 'lon':[-129.090, -121.672]},
        'Offshore_CA':{   'lat':[31.932,42.642], 'lon':[-129.090,-115.806] },
        # 'Great Lakes':{ 'lat':[15.0,27.000002], 'lon':[-164.0,-151.0] },
        'Hawaii':{        'lat':[15.565,26.221], 'lon':[-164.451,-151.278] },
        'NW_Pacific':{    'lat':[41.213, 49.579], 'lon':[-130.831, -121.672] },
        'Mid_Atlantic':{  'lat':[37.273, 42.211], 'lon':[-76.427, -64.800] },
    }

    region_search = lambda x: all( ( True if rDict[x][dk][0] <= d <= rDict[x][dk][1] else False
                                    for dk, d in {'lat':lat_lon[0],'lon':lat_lon[1]}.items() ) )
    region = [key for key in rDict if region_search(key)]
    
    if region[0] == 'CA_NWP_overlap':
        if preferred_region == 'Offshore_CA':
            region[0] = 'Offshore_CA'
        elif preferred_region == 'NW_Pacific':
            region[0] = 'NW_Pacific'
        else:
            raise TypeError(f"Preferred_region ({preferred_region}) must be 'Offshore_CA' or 'NW_Pacific' when lat_lon {lat_lon} falls in the overlap region")
        
    if len(region)==0:
        raise TypeError(f'Coordinates {lat_lon} out of bounds. Must be within {rDict}')
    else:
        return region[0]



def request_wtk_point_data(time_interval, parameter, lat_lon, years, preferred_region='', 
                           tree=None, unscale=True, str_decode=True,hsds=True):
    
        """ 
        Returns data from the WIND Toolkit offshore wind hindcast hosted on AWS at the specified latitude and longitude point(s), 
        or the closest available point(s).
        Visit https://registry.opendata.aws/nrel-pds-wtk/ for more information about the dataset and available 
        locations and years. 

        Note: To access the WIND Toolkit hindcast data, you will need to configure h5pyd for data access on HSDS. 
        Please see the WTK_hindcast_example notebook for more information.  

        Parameters
        ----------
        time_interval : string
            Data set type of interest
            Options: '1-hour' '5-minute'
        parameter: string or list of strings
            Dataset parameter to be downloaded
            5-minute dataset options: 
                'windspeed_10m', 'windspeed_40m', 'windspeed_60m', 'windspeed_80m',
                'windspeed_100m', 'windspeed_120m', 'windspeed_140m', 'windspeed_160m', windspeed_200m', 
                'winddirection_10m', 'winddirection_40m', 'winddirection_60m', 'winddirection_80m',
                'winddirection_100m', 'winddirection_120m', 'winddirection_140m', 'winddirection_160m', 'winddirection_200m',
            1-hour dataset options: 'time_index', 'coordinates', 'inversemoninobukhovlength_2m', 
                'precipitationrate_0m', 'relativehumidity_2m', 
                'temperature_2m', 'temperature_10m', 
                'temperature_40m', 'temperature_60m', 'temperature_80m', 'temperature_100m', 
                'temperature_120m', 'temperature_140m', 'temperature_160m', 'temperature_200m', 
                'pressure_0m', 'pressure_100m', 'pressure_200m',
                'DIF', 'GHI', 'DNI', 'status'
                
                
                TODO : also available??
                ['', 'friction_velocity_2m', '', 'meta', 
                 'roughness_length', 'surface_sea_temperature'
                 '', '', '', 
                 '', '', '', '', 
                 '', 'temperature_180m', '', 'temperature_20m', 
                 '', '', '', '', 
                 '', 
                 '', '', '', '', 
                 '', 'winddirection_180m', '', 'winddirection_20m', 
                 '', '', '', 
                 '', '', '', '', '', 
                 'windspeed_180m', '', 'windspeed_20m', '', '', 
                 '']
                
        lat_lon: tuple or list of tuples
            Latitude longitude pairs at which to extract data 
        years : list 
            Year(s) to be accessed. The years 2000-2019 available (up to 2020 
            for Mid-Atlantic). Examples: [2015] or [2004,2006,2007]
        preferred_region : string (optional)
            Region that the lat_lon belongs to ('Offshore_CA' or 'NW_Pacific').
            Required when a lat_lon point falls in both the Offshore California
            and NW Pacific regions. Overlap region defined by 
            latitude = (41.213, 42.642) and longitude = (-129.090, -121.672).
            Default = ''
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
        assert isinstance(time_interval, str), 'time_interval must be a string'
        assert isinstance(years,list), 'years must be a list'
        assert isinstance(preferred_region, str), 'preferred_region must be a string'
        assert isinstance(tree,(str,type(None))), 'tree must be a string'
        assert isinstance(unscale,bool), 'unscale must be bool type'
        assert isinstance(str_decode,bool), 'str_decode must be bool type'
        assert isinstance(hsds,bool), 'hsds must be bool type'

        # check for multiple region selection
        if isinstance(lat_lon[0], float):
            region = region_selection(lat_lon, preferred_region)
        else:
            reglist = []
            for loc in lat_lon:
                reglist.append(region_selection(loc))
            if reglist.count(reglist[0]) == len(lat_lon):
                region = reglist[0]
            else:
                raise TypeError('Coordinates must be within the same region!')
        
        if time_interval == '1-hour':
            wind_path = f'/nrel/wtk/'+region.lower()+'/'+region+'_*.h5'
        elif time_interval == '5-minute':
            wind_path = f'/nrel/wtk/'+region.lower()+'-5min/'+region+'_*.h5'
        else:
            raise TypeError(f"Invalid time_interval '{time_interval}', must be '1-hour' or '5-minute'")
        windKwargs = {'tree':tree,'unscale':unscale,'str_decode':str_decode, 'hsds':hsds,
            'years':years}
        data_list = []
        
        with MultiYearWindX(wind_path, **windKwargs) as rex_wind:
            if isinstance(parameter, list):
                for p in parameter:
                    temp_data = rex_wind.get_lat_lon_df(p,lat_lon)
                    col = temp_data.columns[:]
                    for i,c in zip(range(len(col)),col):
                        temp = f'{p}_{i}'
                        temp_data = temp_data.rename(columns={c:temp})

                    data_list.append(temp_data)
                data= pd.concat(data_list, axis=1)
                
            else:
                data = rex_wind.get_lat_lon_df(parameter,lat_lon)
                col = data.columns[:]

                for i,c in zip(range(len(col)),col):
                    temp = f'{parameter}_{i}'
                    data = data.rename(columns={c:temp})

            meta = rex_wind.meta.loc[col,:]
            meta = meta.reset_index(drop=True)    
        return data, meta
