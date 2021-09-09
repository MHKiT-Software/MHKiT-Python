from os.path import abspath, dirname, join, normpath, relpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
from mhkit.river.io import d3d 
import scipy.interpolate as interp

# File location and load data
exdir= dirname(abspath(__file__))
datadir = normpath(join(exdir,relpath('data\\river\\d3d')))
filename= 'turbineTest_map.nc'
data = netCDF4.Dataset(join(datadir,filename))

variables= ['turkin1'] # , 'turkin1', 'ucx', 'ucy', 'ucz'] 

# All data in  NetCDF File
var_data_df= d3d.get_all_data_points(data, variables[0],time_step=-1)

#raw data cordinates 
x_raw= np.unique(var_data_df.x)
y_raw= np.unique(var_data_df.y)
z_raw= np.unique(var_data_df.z)


# Centerline points to request data at
xmin=var_data_df.x.max()
xmax=var_data_df.x.min()

ymin=var_data_df.y.max()
ymax=var_data_df.y.min()

zmin=var_data_df.z.max()
zmax=var_data_df.z.min()

x = np.linspace(xmin, xmax, num=100)
y = np.mean([ymin,ymax])
z = np.mean([zmin,zmax])

cline_points = d3d.create_points(x, y, z)

# Contour points to request data 
y_contour = np.linspace(ymin, ymax, num=40)
contour_points = d3d.create_points(x, y_contour, z) 


# Interpolate NetCDF Data onto Centerline
cline_variable = interp.griddata(var_data_df[['x','y','z']], 
                     var_data_df[variables[0]],
                     cline_points[['x','y','z']])
                     
# Interpolate NetCDF Data onto contour
contour_variable = interp.griddata(var_data_df[['x','y','z']], 
                     var_data_df[variables[0]],
                     contour_points[['x','y','z']])
contour_points['contour_variable']= contour_variable

 
time_step= -1
TI= d3d.turbulent_intensity(data, contour_points,  time_step)

max_plot_v= 0.2
min_plot_v=-0.2
# plot layered data 
layer=2
[x_layer,y_layer,value_layer] = d3d.get_layer_data(data, variables[0], layer , time_step=-1)
  
layer2=3
[x_layer2,y_layer2,value_layer2] = d3d.get_layer_data(data, variables[0], layer2 , time_step=-1)     
# Plot Centerline
Type='Centerline'
plt.plot(x, cline_variable)
plt.xlabel('x (m)')
plt.ylabel(f'{data.variables[variables[0]].long_name} [{data.variables[variables[0]].units}]')
plt.title(f'{Type} {data.variables[variables[0]].long_name}')
plt.savefig('cline.png')

#plot Contour TI 
Type= 'Contour'
plt.figure()
contour_plot = plt.tricontourf(contour_points.x,contour_points.y,TI ,vmin=min_plot_v,vmax=max_plot_v,levels=np.linspace(min_plot_v,max_plot_v,10))
plt.xlabel('x (m)')
plt.ylabel('y(m)')
plt.title(f'{Type} {data.variables[variables[0]].long_name}')
cbar= plt.colorbar(contour_plot,boundaries=np.linspace(min_plot_v,max_plot_v,5))
cbar.set_label(f'{data.variables[variables[0]].long_name} [{data.variables[variables[0]].units}]')
#plt.clim(0,0.08)
plt.savefig('contour.png')

# #Plot contour
# Type= 'Contour'
# plt.figure()
# contour_plot = plt.tricontourf(contour_points.x,contour_points.y,contour_points.contour_variable,vmin=min_plot_v,vmax=max_plot_v,levels=np.linspace(min_plot_v,max_plot_v,10))
# plt.xlabel('x (m)')
# plt.ylabel('y(m)')
# plt.title(f'{Type} {data.variables[variables[0]].long_name}')
# cbar= plt.colorbar(contour_plot,boundaries=np.linspace(min_plot_v,max_plot_v,5))
# cbar.set_label(f'{data.variables[variables[0]].long_name} [{data.variables[variables[0]].units}]')
# #plt.clim(0,0.08)
# plt.savefig('contour.png')



# #Plot Layer 
# Type= 'Contour'
# plt.figure()
# contour_plot = plt.tricontourf(x_layer,y_layer,value_layer, vmin=min_plot_v,vmax=max_plot_v,levels=np.linspace(min_plot_v,max_plot_v,10))
# plt.xlabel('x (m)')
# plt.ylabel('y(m)')
# plt.title(f'{Type} Layer: {layer} {data.variables[variables[0]].long_name}')
# cbar= plt.colorbar(contour_plot,boundaries=np.linspace(min_plot_v,max_plot_v,5))
# cbar.set_label(f'{data.variables[variables[0]].long_name} [{data.variables[variables[0]].units}]')
# #plt.clim(0,0.08)
# plt.savefig(f'contour{layer}.png')
 
# Type= 'Contour'
# plt.figure()
# contour_plot = plt.tricontourf(x_layer2,y_layer2,value_layer2,vmin=min_plot_v,vmax=max_plot_v,levels=np.linspace(min_plot_v,max_plot_v,10))
# plt.xlabel('x (m)')
# plt.ylabel('y(m)')
# plt.title(f'{Type} Layer: {layer2} {data.variables[variables[0]].long_name}')
# cbar= plt.colorbar(contour_plot,boundaries=np.linspace(min_plot_v,max_plot_v,5))
# cbar.set_label(f'{data.variables[variables[0]].long_name} [{data.variables[variables[0]].units}]')
# #plt.clim(0,0.08)
# plt.savefig(f'contour{layer}.png')
 
plt.show()