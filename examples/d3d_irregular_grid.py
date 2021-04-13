from os.path import abspath, dirname, join, isfile, normpath, relpath
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import pandas as pd

exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data/river/d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))

def get_variable(data,variable,layer = -1 ,TS=-1):
    '''
    get variable data from netcdf4 object

    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string
        variable to call.
    TS: float 
        time step. Defalt is late tiem step -1  
    Layer: float
        Delft3D layer. layer must be a positve interger 

    Returns
    -------
    None.

    '''
    coords = str(data.variables[variable].coordinates).split()
    var=data.variables[variable][:]
    max_layer= len(var[0][0])
    itime= TS 
    x=np.ma.getdata(data.variables[coords[0]][:], False) 
    y=np.ma.getdata(data.variables[coords[1]][:], False)
    if layer > max_layer:
        print ('layer out of range')
        z=[0]*len(x)
    else:
        z= np.ma.getdata(var[itime,:,layer], False)
    return x,y,z



def get_variable_points(data, variable,x,y,z, time_step=-1):

    '''
    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string 
        variable to call.
    depth : float
        depth 
    time_step : float 
        time step The default is -1.
    Center_line : float
        location of center line. The default is 3.

    Returns
    -------
    points 

    '''
    try:
        x = np.array([x])
    except:
        pass
    try:
        y = np.array([y])
    except:
        pass
    try:
        z = np.array([z])
    except:
        pass
    
    assert(sum([len(x) == 1, len(y)==1, len(z)==1]) >= 2), ('must provide'/
           f'at least 2 points. got x={x}, y={y}, z={z}') 
    
    if not points:
        assert(all([x,y,z])), 'Must specify either x,y,& z or provide points'
    
    
    # x = vector; y,z are floats
    if ((len(x) != len(y)) or (len(x) != len(z)) or (len(y) != len(z))):
         
        # Now find greatest length
        length = 0
        directions = {0: {'name' : 'x',
                         'values': x},
                      1:{'name' : 'y',
                         'values' : y},
                      2:{'name' : 'z',
                         'values' : z}}
        directions_vals = [directions[n]['values'] for n in directions]
        lens = np.array([np.size(d)  for d in directions])
        max_len_idx = lens.argmax()
        not_max_idxs= [i for i in directions.keys()]
        del not_max_idxs[max_len_idx]
        
            
        for not_max in not_maxs:           
            vals = [directions[not_max]['values'] for i in range(lens[max_len])]
            directions[not_max]['values'] = np.array(vals)
        
    

 




def get_all_data_points(data, variable,time_step):
    '''
    get data points from all layers in netcdf file generated from Delft3D  

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

        
    
    # variable= 'ucx'
    # time_step= -1
    # Center_line=3
    # depth= 3

    Layer_percentages= np.ma.getdata(data.variables['LayCoord_cc'][:], False) # velocity data 
   # Layer_percentages= np.ma.getdata(data.variables['LayCoord_w'][:], False) # turbulent data 
    
    bottom_depth=np.ma.getdata(data.variables['waterdepth'][time_step, :], False)# add Time_step
    
    x_all=[]
    y_all=[]
    z_all=[]
    v_all=[]
    
    N_layers = range(len(Layer_percentages))
    
    for layer in N_layers:
        x,y,v= get_variable(data, variable, layer, time_step)
        z = [bottom_depth*Layer_percentages[layer]]
        x_all=np.append(x_all, x)
        y_all=np.append(y_all, y)
        z_all=np.append(z_all, z)
        v_all=np.append(v_all, v)
    
    known_points = np.array([ [x, y, z, v] for x, y, z, v in zip(x_all, y_all, z_all, v_all)])
    
    all_data= pd.DataFrame(known_points, columns=['x','y','z',f'{variable}'])
    
    return all_data

def plot_data(data, variable, x=None , y=None, z=None, points=None , time_step = -1):
   
    '''
    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string 
        variable to call.
    depth : float
        depth 
    time_step : float 
        time step The default is -1.
    Center_line : float
        location of center line. The default is 3.

    Returns
    -------
    None.

    '''
    all_data = get_all_data_points(data, variable, time_step)
    
   # if points:
    v_new = interp.griddata(all_data[['x','y','z']], all_data[f'{variable}'], points)
        
      #  if len(v_new)> 1
            
        
        
    # if not points: 
    #     get_variable_points(data, variable, x, y, z, time_step)
    #     v_new = interp.griddata(known_points, v_all, points)
     
    

    


    x_new= points[:,0]
   

   # v_new = interp.griddata(points, v_all, request)

    
   # returnt Point 
    
   #ploting centerl line 
    plt.plot(x_new, v_new)
    #plt.title(f'Depth {depth}')
    units= data.variables[variable].units
    cname=data.variables[variable].long_name
    plt.xlabel('x (m)')
    #plt.ylabel(f'{cname} [{units}]')
    plt.show() 
    
    #plot contor plot 
    


# End loop section 
x_new = np.linspace (0, 18)
y_new = [3] *len(x_new)
z_new = [1]*len(x_new)
    
request= np.array([ [x, y, z] for x, y, z in zip(x_new, y_new, z_new)]) 
   
variables= [ 'ucx']# , 'turkin1"]
for var in variables:
    plot_data(data,var, points=request)  # data, variable, (x, y, z) or (points), timestep 