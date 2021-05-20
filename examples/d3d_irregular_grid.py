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
        print ('layer out of range')
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
 # TODO loop between Velocity and turbulence data 
    Layer_percentages= np.ma.getdata(data.variables['LayCoord_cc'][:], False) # velocity data 
   # Layer_percentages= np.ma.getdata(data.variables['LayCoord_w'][:], False) # turbulent data 
    
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
        interpolared values for locations in request_points 

    '''
    v_new = interp.griddata(data_points, values, request_points)

    return v_new


# def plot(points, values, data, variable) :
    
#     if  treat_as == 'centerline'
#         plt.plot(x, y)
#         #plt.title(f'Depth {depth}')
#         units= data.variables[variable].units
#         cname=data.variables[variable].long_name
#         plt.xlabel('x (m)')
#         #plt.ylabel(f'{cname} [{units}]')
#         plt.show()
#     elif treat_as == 'plane'
#         x=np.array (points['x'])
#         yy=np.array (points['y'])
#         df=squeez(df)
        
#         plt.tricontourf(xx,yy,df)
#         cbar=plt.colorbar()
#         units= data.variables[variable].units
#         cname=data.variables[variable].long_name
#         cbar.set_label(f'{cname} [{units}]')
#         plt.xlabel('x (m)')# user input 
#         plt.ylabel('y (m)')
#         plt.show()
#     else 

    
exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data/river/d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))


x =  np.linspace (0.1, 17.9, num=100)
y = np.linspace (1.1, 4.9, num=100)
z=  1#np.linspace (0.2, 1.8, num=100) 
    
#request= np.array([ [x, y, z] for x, y, z in zip(x_new, y_new, z_new)]) 
#request_points= pd.DataFrame(request, columns=[ 'x', 'y', 'z'])
#points=request_points
   
variables= ['ucx']# , 'turkin1"]
#
var_data_df=get_all_data_points(data, variables[0],time_step=-1)
points = create_points(x, y, z)
points['ucx']=interpolate_data(var_data_df[['x','y','z']], var_data_df[['ucx']],
            points[['x','y','z']])  

points.dropna(inplace=True)
plt.tricontourf(points.x,points.y,points.ucx)