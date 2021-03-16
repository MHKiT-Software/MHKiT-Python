from os.path import abspath, dirname, join, isfile, normpath, relpath
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import pandas as pd


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


exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data/river/d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))

vars= ['ucx', 'turkin1']
#for var in vars: 
#   plot_variable(data,var, 4)
   
#plt.show()
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
def plot_centerline(data,variable,layer, TS=-1,CT=2.95):
    '''
    Parameters
    ----------
    data : netcdf4 object 
        d3d netcdf file 
    variable : string
        variable to call.
    Layer: float
        Delft3D layer.      
    TS: float 
        time step. Defalt is late tiem step -1  
    CT : float, optional
        centerline location. The default is 3.
    Returns
    -------
    None.

    '''
    x,y,z= get_variable(data, variable, layer,TS)
    
    x = np.ma.getdata(x, False)
    y = np.ma.getdata(y, False)
    z = np.ma.getdata(z, False)
    
    df = pd.DataFrame(x, columns=['x'])
    df['y'] = y
    df['z'] = z
    
    y_unique= np.unique(y)
    x_unique=np.unique(x)
      
    #import ipdb; ipdb.set_trace()
    if  any(CT==y_unique):
        Yidx=len(np.unique(y))//2
        idx= y_unique[Yidx]
        CTL = np.where(y== idx)
        zCT= z[CTL]
        
           
        from scipy import interpolate  
    else: 
        imax = np.searchsorted(y_unique, CT)
        imin=imax-1

        if imax== 0 or imin == y_unique[-1] :
            print('error')
        else:
        
            ymin = y_unique[imin]
            ymax = y_unique[imax]
            
            xmin= x_unique[imin]
            xmax= x_unique[imax]
        
            #idx_cl_max = np.where(y == ymax)[0]
            #idx_cl_min = np.where(y == ymin)[0]
        
            #var_max = np.ma.getdata(z[idx_cl_max], False)
            #var_min = np.ma.getdata(z[idx_cl_min], False)
    
            z_mins = df[df.y==ymin].z
            z_maxs = df[df.y==ymax].z
    
            zCT=[]
            Y=[ymin, ymax]
            X=[xmin, xmax]
            for z_min, z_max in zip(z_mins, z_maxs):
            
                Z=[z_min, z_max]
                Zi = interpolate.intep2d(X,Y,Z, )
                zCT.append(Zi)
                
            zCT=np.array(zCT)
    import ipdb; ipdb.set_trace()
    
    z_mins = df[df.y==ymin].z
    z_maxs = df[df.y==ymax].z   
    
    plt.plot(x_unique,zCT)
    # plt.title(f'Layer {layer}')
    # units= data.variables[variable].units
    # cname=data.variables[variable].long_name
    # plt.xlabel('x (m)')
    # plt.ylabel(f'{cname} [{units}]')
    # plt.show()
    
vars= [ 'turkin1']
for var in vars:
    plot_centerline(data,var,2) 