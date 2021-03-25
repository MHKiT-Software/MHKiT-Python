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
   
plt.show()
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
def plot_centerline(data,variable,layer, TS=-1, center_line=3):
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
    center_line : float, optional
        centerline location. The default is 3.
    Returns
    -------
    None.

    '''
    lower_layer= np.floor(layer)

    upper_layer= lower_layer + 1 

    if layer == lower_layer:
        x,y,v= get_variable(data, variable, layer,TS)
            
        x = np.ma.getdata(x, False)
        y = np.ma.getdata(y, False)
        v = np.ma.getdata(v, False)
                
        df = pd.DataFrame(x, columns=['x'])
        df['y'] = y
        df['v'] = v
                
        y_unique= np.unique(y)
        x_unique= np.unique(x)
                  
        #import ipdb; ipdb.set_trace()
        if  any(center_line==y_unique):
            Yidx=len(y_unique)//2
            idx= y_unique[Yidx]
            center_line_index = np.where(y== idx)
            x_plot = x[center_line_index]
            v_plot = v[center_line_index]
                    
         
        else: 
            imax = np.searchsorted(y_unique, center_line)
            imin=imax-1
            
            if imax== 0 or imin == y_unique[-1] :
                print('error')
            else:

                ymin = y_unique[imin]
                ymax = y_unique[imax]
                
                y_mins = df[df.y==ymin].y
                y_maxs = df[df.y==ymax].y
                Y=np.concatenate((y_mins.values, y_maxs.values))
                
                                
                x_mins = df[df.y==ymin].x
                x_maxs = df[df.y==ymax].x
                X=np.concatenate((x_mins.values, x_maxs.values))
                    
                
                v_mins = df[df.y==ymin].v
                v_maxs = df[df.y==ymax].v
                V=np.concatenate((v_mins.values, v_maxs.values))
        
                points= np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), V.reshape(-1,1)), axis=1)
                points= pd.DataFrame(points, columns=['X','Y','V'])
                points_sorted= points.sort_values(by= 'X')
                points_sorted = points_sorted.reset_index(drop=True)
                
                X_interpolated_centerline= []
                V_interpolated_centerline= []
                
        
                for i in range(len(points_sorted)-1):
                    X = np.interp(center_line, (points_sorted.Y[i],points_sorted.Y[i+1]), (points_sorted.X[i],points_sorted.X[i+1]))
                    V = np.interp(center_line, (points_sorted.Y[i],points_sorted.Y[i+1]), (points_sorted.V[i],points_sorted.V[i+1]))      
                    X_interpolated_centerline.append(X)
                    V_interpolated_centerline.append(V)
                x_plot = X_interpolated_centerline
                v_plot = V_interpolated_centerline
            
        
        plt.plot(x_plot,v_plot)
        plt.title(f'Layer {layer}')
        units= data.variables[variable].units
        cname=data.variables[variable].long_name
        plt.xlabel('x (m)')
        plt.ylabel(f'{cname} [{units}]')
        plt.show()
    else:
        print('error')
        
    
variables= [ 'ucx', 'turkin1']
for var in variables:
    plot_centerline(data,var,2, -1, 3) 