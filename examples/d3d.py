import netCDF4
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np
from os.path import abspath, dirname, join, isfile, normpath, relpath


exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data/river/d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))

def plot_variable(data,variable, layer, TS=-1):

    '''
    plots velocity for d3d map file 

    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string
        variable to call. 
    TS: float 
        time step. Defalt is late tiem step -1  
    Layer: float
        Delft3D layer.      
        
    Returns
    -------
    ax: figure 
        contour of velocity

    '''
    x,y,z= get_variable(data,variable,layer, TS)

    
    plt.tricontourf(x,y,z)
    plt.title(f'Layer {layer}')
    cbar=plt.colorbar()
    units= data.variables[variable].units
    cname=data.variables[variable].long_name
    cbar.set_label(f'{cname} [{units}]')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()



def get_variable(data,variable,layer,TS=-1):
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
        Delft3D layer.     

    Returns
    -------
    None.

    '''
    coords = str(data.variables[variable].coordinates).split()
    var=data.variables[variable][:]
    itime= TS 
    x=data.variables[coords[0]][:]
    y=data.variables[coords[1]][:]
    z= var[itime,:,layer]
    return x,y,z 
#==============================================================================
vars= ['ucx', 'turkin1']
for var in vars: 
   plot_variable(data,var, 4)
#plot_variable(data, 'ucx', 'velocity', 3)

# interpolate turbulence data onto velocity grid 
#x,y,z= get_variable(data,'ucx')
#x2,y2,z2= get_variable(data,'turkin1')

#z3 = interp.griddata(np.array([x,y]).T, z,(x2, y2), method='cubic')
#plt.tricontourf(x2,y2,z3)
#plt.show()

# Turblent intensity

# x,y,vx= get_variable(data,'ucx',2)
# x,y,vy= get_variable(data,'ucy',2)
# x,y,vz= get_variable(data,'ucz',2)
# x,y,TK= get_variable(data,'turkin1',2)

# mv= (vx**2+vy**2+vz**2)**0.5
# TI= (2/3*TK)**0.5/mv 



#Centerline Plot 

def get_centerline(data,variable,layer, TS=-1):
   x,y,z= get_variable(data, variable, layer,TS)
   
   

