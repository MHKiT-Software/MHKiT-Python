from os.path import abspath, dirname, join, isfile, normpath, relpath
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4


def get_layer_data(data, variable, layer_index= -1 , time_index=-1):
    '''
    Get variable data from netcdf4 object at specified layer and timestep. 

    Parameters
    ----------
    data : netcdf4 object 
       After running a Delft3D model the output netcdf file can be read into this code.  
    variable : string
        Delft3d outputs many vairables that can be called. The full list can be found using "data.variables.keys()" in the consol. 
    layer_index: int
         A positve interger to pull out a layer from the dataset. 0 being closest to the surface. Defalt is the bottom layer "-1."
    time_index: int 
        A positive interger to pull the time step from the dataset. 0 being closest to time 0. Defalt is last time step -1.  

    Returns
    -------
    layer_data:DataFrame
        DataFrame of "x" and "y" location on specified layer with the variable values "v"

    '''
    assert isinstance(time_index, int), 'time_index  must be a int'
    assert isinstance(layer_index, int), 'layer_index  must be a int'
    assert type(data)== netCDF4._netCDF4.Dataset, 'data must be netCDF4 object'
    assert variable in data.variables.keys(), 'variable not recognized'
    coords = str(data.variables[variable].coordinates).split()
    var=data.variables[variable][:]
    max_time_index= len(var)
    assert time_index <= max_time_index, f'time_index must be less than the max time index {max_time_index}'
    assert time_index >= -1, 'time_index must be greater than or equal to -1'
    max_layer= len(var[0][0])
    assert layer_index <= max_layer, f'layer must be less than the max layer {max_layer}'
    assert layer_index >= -1, 'layer must be greater than or equal to -1'
    
    x=np.ma.getdata(data.variables[coords[0]][:], False) 
    y=np.ma.getdata(data.variables[coords[1]][:], False)
 
    v= np.ma.getdata(var[time_index,:,layer_index], False)
    
    layer= np.array([ [x_i, y_i, z_i] for x_i, y_i, z_i in zip(x, y, v)]) 
    layer_data = pd.DataFrame(layer, columns=[ 'x', 'y', 'v'])
    
    return layer_data


def create_points(x, y, z):
    '''
    Turns x, y, and z into a DataFrame of points to interpolate over,
    Must be provided at least two points and at most one array 
    
    Parameters
    ----------
   
    x: float, array or int 
        x values to create points.
    y: float, array or int
        y values to create points.
    z: float, array or int
        z values to create points.

    Returns
    -------
    points: DateFrame 
        DataFrame of x, y and z points 

    '''
    assert isinstance(x, (int, float, np.ndarray)), 'x must be a int, float or array'
    assert isinstance(y, (int, float, np.ndarray)), 'y must be a int, float or array'
    assert isinstance(z, (int, float, np.ndarray)), 'z must be a int, float or array'
    
    directions = {0:{'name':  'x',
                     'values': x},
                  1:{'name':  'y',
                     'values': y},
                  2:{'name':  'z',
                     'values': z}}

    for i in directions:
        try:
            len(directions[i]['values'])
        except:
            directions[i]['values'] = np.array([directions[i]['values']])  
            
        N= len(directions[i]['values'])
        if N== 1 :
            directions[i]['type']= 'point'
        elif N > 1 :
            directions[i]['type']= 'vector'
        else:
            raise Exception(f'length of direction {directions[i]["name"]} was neagative or zero')
    
    # Check how many times point is in "types" 
    types= [directions[i]['type'] for i in directions]
    N_points = types.count('point')
    if N_points >= 2: 
        #  treat_as centerline 
        lens = np.array([np.size(d)  for d in directions])
        max_len_idx = lens.argmax()
        not_max_idxs= [i for i in directions.keys()]
        del not_max_idxs[max_len_idx]

        for not_max in not_max_idxs:     
            N= len(directions[max_len_idx]['values'])
            vals =np.ones(N)*directions[not_max]['values']
            directions[not_max]['values'] = np.array(vals)
                    
        x_new = directions[0]['values']
        y_new = directions[1]['values']
        z_new = directions[2]['values']
            
        request= np.array([ [x_i, y_i, z_i] for x_i, y_i, z_i in zip(x_new, y_new, z_new)]) 
        points= pd.DataFrame(request, columns=[ 'x', 'y', 'z'])
    elif N_points == 1: 
        # treat as plane
        #find index of point 
        idx_point = types.index('point')
        max_idxs= [i for i in directions.keys()]
        del max_idxs[idx_point]
        #find vectors 
        XX, YY = np.meshgrid(directions[max_idxs[0]]['values'], directions[max_idxs[1]]['values'] )
        N_X=np.shape(XX)[1]#or take len of original vectors 
        N_Y=np.shape(YY)[0]
        ZZ= np.ones((N_Y,N_X))*directions[idx_point]['values'] #make sure is the same shape as XX , YY 
     
        request= np.array([ [x_i, y_i, z_i] for x_i, y_i, z_i in zip(XX.ravel(),
                            YY.ravel() , ZZ.ravel())]) 
        columns=[ directions[max_idxs[0]]['name'],  
                 directions[max_idxs[1]]['name'],  directions[idx_point]['name']]
        
        points= pd.DataFrame(request, columns=columns)
    else: 
        raise Exception('Can provide at most two vectors')

    return points 


def grid_data(data,variables, points='cells'):
    '''
    Convert multiple variables from the Delft3d onto the same points. 

    Parameters
    ----------
    data : netcdf4 object 
        After running a Delft3D model the output netcdf file can be read into this code.
    variables: string array 
        Name of variables to interpolate, e.g. turkin1, ucx, ucy and ucz. The full list can be found using "data.variables.keys()" in the console.
    points : string, DataFrame  
        Point to interpoate data onto. 
          'cells' : interpolates all data into velocity coordinat system (Default)
          'faces': interpolates all dada into TKE coordinate system 
          DataFrame of x, y, and z coordinates: Interpolates data onto user povided points 
  
    Returns
    -------
    transformed_data : DataFrame  
        Variables on specified grid points saved under the input varable names and the 
        x,y and z cordinates of those points 

    '''
    #assert points == 'cells' or points=='faces' or type(points) == pd.core.frame.DataFrame, 'points must be cells or faces or DataFrame'
    assert type(data)== netCDF4._netCDF4.Dataset, 'data must be nerCDF4 object'

    data_raw = {}
    for var in variables:
        #get all data
        var_data_df = get_all_data_points(data, var,time_index=-1)           
        data_raw[var] = var_data_df 
    if type(points) == pd.DataFrame:  
        print('points provided')
    elif points=='faces':
        points = data_raw['ucx'][['x','y','z']]
    elif points=='cells':
        points = data_raw['turkin1'][['x','y','z']]
    
    transformed_data= points.copy(deep=True)
    
    for var in variables :    
        transformed_data[var] = interp.griddata(data_raw[var][['x','y','z']],
                                        data_raw[var][var], points[['x','y','z']])

    return transformed_data


def get_all_data_points(data, variable, time_index= -1):  
    '''
    Get data points from all layers in netcdf file generated from Delft3D using get_layer_data function. 

    Parameters
    ----------
    data : netcdf4 object 
        After running a Delft3D model the output netcdf file can be read into this code.   
    variable : string
        Delft3d outputs many vairables that can be called. The full list can be found using "data.variables.keys()" in the consol. 
    time_index : int
        A positive interger to pull the time step from the dataset. Defalt is late time step -1.  
        
    Returns
    -------
    all_data: DataFrame 
        Dataframe of x, y, z, and variable. 

    '''  
    assert isinstance(time_index, int), 'time_index  must be a int'
    assert type(data)== netCDF4._netCDF4.Dataset, 'data must be nerCDF4 object'
    assert variable in data.variables.keys(), 'varaiable not reconized'

    max_time_index = len(data.variables[variable][:])
    assert time_index <= max_time_index, f'time_index must be less than the max time index {max_time_index}'
    assert time_index >= -1, 'time_index must be greater than or equal to -1'

    cords_to_layers= {'laydim': data.variables['LayCoord_cc'][:],
                       'wdim': data.variables['LayCoord_w'][:]}
    lay_element= 2 
    layer_dim =  [v.name for v in data[variable].get_dims()][lay_element]
    
    try:    
        cord_sys= cords_to_layers[layer_dim]
    except: 
        raise Exception('coordinates not recognized')
    else: 
        Layer_percentages= np.ma.getdata(cord_sys, False) 
        
        
    bottom_depth=np.ma.getdata(data.variables['waterdepth'][time_index, :], False)
    if layer_dim == 'wdim': 
        #interpolate 
        coords = str(data.variables['waterdepth'].coordinates).split()
        x_laydim=np.ma.getdata(data.variables[coords[0]][:], False) 
        y_laydim=np.ma.getdata(data.variables[coords[1]][:], False)
        points_laydim = np.array([ [x, y] for x, y in zip(x_laydim, y_laydim)])
        
        coords_request = str(data.variables[variable].coordinates).split()
        x_wdim=np.ma.getdata(data.variables[coords_request[0]][:], False) 
        y_wdim=np.ma.getdata(data.variables[coords_request[1]][:], False)
        points_wdim=np.array([ [x, y] for x, y in zip(x_wdim, y_wdim)])
        
        bottom_depth_wdim = interp.griddata(points_laydim, bottom_depth, points_wdim)
        
        idx= np.where(np.isnan(bottom_depth_wdim))
        
        for i in idx: 
            bottom_depth_wdim[i]= interp.griddata(points_laydim, bottom_depth,
                                                  points_wdim[i], method='nearest')
        
        
        
    x_all=[]
    y_all=[]
    z_all=[]
    v_all=[]
    
    N_layers = range(len(Layer_percentages))
    for layer in N_layers:
        layer_data= get_layer_data(data, variable, layer, time_index)
        if layer_dim == 'wdim': 
            z = [bottom_depth_wdim*Layer_percentages[layer]]
        else: 
            z = [bottom_depth*Layer_percentages[layer]]
        x_all=np.append(x_all, layer_data.x)
        y_all=np.append(y_all, layer_data.y)
        z_all=np.append(z_all, z)
        v_all=np.append(v_all, layer_data.v)
    
    known_points = np.array([ [x, y, z, v] for x, y, z, v in zip(x_all, y_all, z_all, v_all)])
    
    all_data= pd.DataFrame(known_points, columns=['x','y','z',f'{variable}'])
    
    return all_data


def unorm(x, y ,z):
    '''
    Calculates the root mean squared value given three arrays. 

    Parameters
    ----------
    x: array 
        One input for the root mean squared calculation.(eq. x velocity) 
    y: array
        One input for the root mean squared calculation.(eq. y velocity) 
    z: array
        One input for the root mean squared calculation.(eq. z velocity) 

    Returns
    -------
    unorm : array 
        root mean squared output 
    '''

    assert isinstance(x,(np.ndarray, np.float64, pd.Series)), 'x must be an array'
    assert isinstance(y,(np.ndarray, np.float64, pd.Series)), 'y must be an array'
    assert isinstance(z,(np.ndarray, np.float64, pd.Series)), 'z must be an array'
    
    if len(x) == len(y) & len (y) ==len (z) :
        xyz = np.array([x,y,z]) 
        unorm = np.linalg.norm(xyz, axis= 0)
    else:
        raise Exception ('lengths of arrays do not mathch')
    return unorm




def turbulent_intensity(data, points='cells', time_index= -1, intermediate_values = False ):

    '''
    Calculated the turbulent intesity for a given data set for the specified points.  
    Assumes variable names: turkin1, ucx, ucy and ucz.

    Parameters
    ----------
    data : netcdf4 object 
        After running a Delft3D model the output netcdf file can be read into this code. 
    points : string, DataFrame  
        Point to interpoate data onto. 
          'cells' : interpolates all data into velocity coordinat system (Default)
          'faces': interpolates all dada into TKE coordinate system 
          DataFrame of x, y, and z corrdinates: Interpolates data onto user povided points 
    time_step : float 
        A positive interger to pull the time step from the dataset. Defalt is late time step -1.  
    intermediate_values: boolean
        A true or fase boolean that if true will return ucx, uxy, uxz,and turkine1 vaues in Dataframe 
        if fause will only return x,y,z, and turbulent intesity values. Faules is the default value      
        
    Returns
    -------
    TI_data : Dataframe
        If intermediate_values is true all values are output 
        if intermediate_values is equal to fale only turbulent _intesity and x, y, and z varibles are output 
            turbulen_intesity: turbulent kinetic energy divided by the root mean squared velocity
            turkin1: turbulent kinetic energy 
            ucx: velocity in the x direction 
            ucy: velocity in the y direction 
            ucx: velocity in the z direction 
            x: position in the x direstion 
            y: position in the y direction 
            z: position in the z direction 

    '''
    # assert isinstance(points, (str, pd.core.frame.DataFrame)),  'points must be cells or faces or DataFrame'
    # assert points == 'cells' or points=='faces' or type(points) == pd.core.frame.DataFrame, 'points must be cells or faces or DataFrame'
    assert isinstance(time_index, int), 'time_index  must be a int'
    assert type(data)== netCDF4._netCDF4.Dataset, 'data must be nerCDF4 object'
    assert 'turkin1' in data.variables.keys(), 'Varaiable Turkine 1 not present in Data'
    assert 'ucx' in data.variables.keys(),'Varaiable ucx 1 not present in Data'
    assert 'ucy' in data.variables.keys(),'Varaiable ucy 1 not present in Data'
    assert 'ucz' in data.variables.keys(),'Varaiable ucz 1 not present in Data'

    
    TI_vars= ['turkin1', 'ucx', 'ucy', 'ucz']
    TI_data_raw = {}
    for var in TI_vars:
        #get all data
        var_data_df = get_all_data_points(data, var ,time_index)           
        TI_data_raw[var] = var_data_df 
    if type(points) == pd.DataFrame:  
        print('points provided')
    elif points=='faces':
        points = TI_data_raw['ucx'][['x','y','z']]
    elif points=='cells':
        points = TI_data_raw['turkin1'][['x','y','z']]
    
    TI_data = points.copy(deep=True)

    for var in TI_vars:    
        TI_data[var] = interp.griddata(TI_data_raw[var][['x','y','z']],
                                        TI_data_raw[var][var], points[['x','y','z']])
        idx= np.where(np.isnan(TI_data[var]))
        
        if len(idx[0]):
            for i in idx[0]: 
                TI_data[var][i]= interp.griddata(TI_data_raw[var][['x','y','z']], 
                                             TI_data_raw[var][var], [points['x'][i],points['y'][i],points['z'][i]], method='nearest')
            

    u_mag=unorm(np.array(TI_data['ucx']),np.array(TI_data['ucy']), np.array(TI_data['ucz']))
    TI_data['turbulent_intensity']= np.sqrt(2/3*TI_data['turkin1'])/u_mag
    
    if intermediate_values == False:
        TI_data= TI_data.drop(TI_vars, axis = 1)

    return TI_data