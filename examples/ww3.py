import mhkit.wave.io.ww3 as mio
import datetime
import xarray as xr
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs


date = datetime.datetime(2011,1,1)
fn = mio.request_data(date,23,['Tper'])
# fn = '2011-01-01.nc'
ds = xr.open_dataset(fn).squeeze().drop('depth')
fig = plt.figure(figsize=(16/2,10/1.5/2))
# ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
p = ds['Tper'].mean('time').plot.contourf(x='longitude',
                                          y='latitude',)
                                        #   ax=ax) 
plt.title('Average wave period on {}'.format(date.strftime('%Y-%m-%d')))
plt.show()