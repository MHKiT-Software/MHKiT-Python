from os.path import abspath, dirname, join, isfile, normpath, relpath
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import pandas as pd




def _get_layer_data(data, variable, layer = -1 , time_step=-1):
    '''
    Get variable data from netcdf4 object at specified layer and timestep. 

    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string
        variable to call.
    Layer: float
        Delft3D layer. layer must be a positve interger 
    TS: float 
        time step. Defalt is late tiem step -1  


    Returns
    -------
    x,y,v: float
        "x" and "y" location on specified layer with the variables values "v"

    '''
    coords = str(data.variables[variable].coordinates).split()
    var=data.variables[variable][:]
    max_layer= len(var[0][0])
    itime= time_step
    x=np.ma.getdata(data.variables[coords[0]][:], False) 
    y=np.ma.getdata(data.variables[coords[1]][:], False)
    if layer > max_layer:
        v=[0]*len(x)
    else:
        v= np.ma.getdata(var[itime,:,layer], False)
    return x,y,v



def create_points( x, y, z):

    '''
    Turns x, y, and z into a DataFrame of points to interpolate over,
    Must be provided at least two points and at most one array 
    
    Parameters
    ----------
   
    x: float, array or int 
        x values to create points
    y: float, array or int
        y values to create points
    z: float, array or int
        z values to create points

    Returns
    -------
    points: DateFrame 
        DataFrame of x, y and z points 

    '''
    # assert (instance( x,(int, float, array),' x must be a int, float or array'))
    # assert (instance( y,(int, float, array),' x must be a int, float or array'))
    # assert (instance( z,(int, float, array),' x must be a int, float or array'))
    
    directions = {0: {'name' : 'x',
                     'values': x},
                 1:{'name' : 'y',
                    'values' : y},
                 2:{'name' : 'z',
                    'values' : z}}

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
    
    types= [directions[i]['type'] for i in directions]
    #check how many times point is in "types" 
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
    elif N_points ==1: 
        # treat as plane
        #find inded of point 
        idx_point = types.index('point')
        max_idxs= [i for i in directions.keys()]
        del max_idxs[idx_point]
        #find vectors 
        # 
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
 


def get_all_data_points(data, variable, time_step):  
    '''
    Get data points from all layers in netcdf file generated from Delft3D  

    Parameters
    ----------
    data : netcdf object 
        d3d netcdf object 
    variable : string
        string to call variable 
    time_step : float 
        time step

    Returns
    -------
    all_data: DataFrame 
        Data frame of x , y, z, and variable 

    '''  
#    assert instance (variable, ['turbkin1', 'ucx', 'ucy', 'ucz','s1',
#    's0', 'waterdepth', 'numlimdt', 'taus', 'unorm', 'u0', 'q1', 'viu', 'diu',
#    'ucxa', 'ucya', 'rho', 'turkin1', 'vicwwu', 'tureps1', 'czs'], 'variable not reconized')
    
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
        
        
    bottom_depth=np.ma.getdata(data.variables['waterdepth'][time_step, :], False)# add Time_step
    
    x_all=[]
    y_all=[]
    z_all=[]
    v_all=[]
    
    N_layers = range(len(Layer_percentages))
    for layer in N_layers:
        x,y,v= _get_layer_data(data, variable, layer, time_step)
        z = [bottom_depth*Layer_percentages[layer]]
        x_all=np.append(x_all, x)
        y_all=np.append(y_all, y)
        z_all=np.append(z_all, z)
        v_all=np.append(v_all, v)
    
    known_points = np.array([ [x, y, z, v] for x, y, z, v in zip(x_all, y_all, z_all, v_all)])
    
    all_data= pd.DataFrame(known_points, columns=['x','y','z',f'{variable}'])
    
    return all_data

def interpolate_data(data_points, values , request_points):
   
    '''
    interpolate for "requested_points" vales over the known "data_points" wiuth corresponding "values" 
    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    points: DataFrame 
        Data Frame of x, y, z, points 
    request_points: DataFrame 
        x, y and z, locations of points calculate the values 

    Returns
    -------
    v_new: array 
        interpolated values for locations in request_points 

    '''
    v_new = interp.griddata(data_points, values, request_points)

    return v_new

def unorm(x, y ,z):
    '''
    Calculates the root mean squared value given three arrays 

    Parameters
    ----------
    x : Array 
       one input for the root mean squared calculation.(eq. x velocity) 
    y : Array
        one input for the root mean squared calculation.(eq. y velocity) 
    z : Array
        one input for the root mean squared calculation.(eq. z velocity) 

    Returns
    -------
    unorm : array 
        root mean squared output 

    '''
    assert(isinstance(x,np.array),'x input not reconized')
    if len(x) == len(y) & len (y) ==len (z) :
        unorm=np.sqrt(x**2 + y**2 + z**2)
    else:
        raise Exception ('lengths of arrays do not mathch')
    
    return unorm


def turbulent_intensity(data, points='cells'):

    '''
    Calculated the turbulent intesity for a given data set for the specified points 

    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf4 object, assumes variable names: turkin1, ucx, ucy and ucz
    points : string, DataFrame  
        'cells', 'faces', or DataFrame of x, y, and z corrdinates. 
        Default 'Cells' uses velocity coordinate system. 'faces' will
        use the turbulence coordinate system

    Returns
    -------
    turbulent_intensity : array   
        turbulent kinetic energy devided by the root mean squared velocity

    '''


    
    TI_vars= ['turkin1', 'ucx', 'ucy', 'ucz']
    TI_data_raw = {}
    for var in TI_vars:
        #get all data
        var_data_df = get_all_data_points(data, var,time_step=-1)           
        TI_data_raw[var] = var_data_df 
    if type(points) == pd.DataFrame:  
        print('points provided')
    elif points=='faces':
        points = TI_data_raw['ucx'][['x','y','z']]
    elif points=='cells':
        points = TI_data_raw['turkin1'][['x','y','z']]
    
    TI_data= points.copy(deep=True)
    
    for var in TI_vars:    
        TI_data[var] = interpolate_data(TI_data_raw[var][['x','y','z']],
                                        TI_data_raw[var][var], points[['x','y','z']])

    #calculate turbulent intensity 
    u_mag=unorm(TI_data['ucx'],TI_data['ucy'], TI_data['ucz'])
    turbulent_intensity= np.sqrt(2/3*TI_data['turkin1'])/u_mag
                                 
    return turbulent_intensity
    
exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data/river/d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))


x =  np.linspace (0.1, 17.9, num=1000)
y = np.linspace (1.1, 4.9, num=1000)
z=  1#np.linspace (0.2, 1.8, num=100) 
    
#request= np.array([ [x, y, z] for x, y, z in zip(x_new, y_new, z_new)]) 
#request_points= pd.DataFrame(request, columns=[ 'x', 'y', 'z'])
#points=request_points
   
variables= ['ucx'] # , 'turkin1', 'ucx', 'ucy', 'ucz'] 

#check that variable is in data set 

var_data_df=get_all_data_points(data, variables[0],time_step=-1)

var_data_df['TI' ] = turbulent_intensity(data,points='cells')
points = create_points(var_data_df.x, var_data_df.y, z)
points[variables[0]]=interpolate_data(var_data_df[['x','y','z']], var_data_df[variables[0]],
            points[['x','y','z']])   

# Plots 
var_data_df.dropna(inplace=True)

contour_plot= plt.tricontourf(var_data_df.x,var_data_df.y,var_data_df.ucx) 
units= data.variables['unorm'].units
cname='Turbulent Intesity'
cbar= plt.colorbar(contour_plot)
cbar.set_label(f'{cname} [{units}]')
plt.xlabel('x (m)')# user input 
plt.ylabel('y (m)')
plt.title('Cells')
#         plt.show()