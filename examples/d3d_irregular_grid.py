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

def plot_centerline(data, variable, depth, time_step = -1, Center_line=3):
   
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
    
    for i in range(len(Layer_percentages)):
        layer= i 
        x,y,v= get_variable(data, variable, layer, time_step)
        z = [bottom_depth*Layer_percentages[i]]
        x_all=np.append(x_all, x)
        y_all=np.append(y_all, y)
        z_all=np.append(z_all, z)
        v_all=np.append(v_all, v)
    
    points = np.array([ [x_all, y_all, z_all] for x_all, y_all, z_all in zip(x_all, y_all, z_all)])
    
    x_new = np.linspace (0, round(max(x_all)))
    y_new = [Center_line] *len(x_new)
    z_new = [depth]*len(x_new)
    
    request= np.array([ [x_new, y_new, z_new] for x_new, y_new, z_new in zip(x_new, y_new, z_new)])
    
    v_new = interp.griddata(points, v_all, request)
    
    
    plt.plot(x_new, v_new)
    plt.title(f'Depth {depth}')
    units= data.variables[variable].units
    cname=data.variables[variable].long_name
    plt.xlabel('x (m)')
    plt.ylabel(f'{cname} [{units}]')
    plt.show() 
    
variables= [ 'ucx']# , 'turkin1"]
for var in variables:
    plot_centerline(data,var,1, -1, 3)  # data, variable, depth, time step, center_line 