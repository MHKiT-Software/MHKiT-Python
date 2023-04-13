import time
import os
import xarray as xr
import matplotlib.pyplot as plt
# # import cartopy.crs as ccrs
# import mhkit.wave.io.ww3 as mio
# import datetime



# from erddapClient import ERDDAP_Server

# remote = ERDDAP_Server('https://coastwatch.pfeg.noaa.gov/erddap')
# searchResults = remote.search(searchFor="WW3")
# ww3 = searchResults[0]

# print("Search results:", searchResults)
# print(ww3)

# import ipdb; ipdb.set_trace()

path = './ww3.nc'

if os.path.exists(path):
    ds = xr.open_dataset('ww3.nc')
else:
    from erddapClient import ERDDAP_Griddap
    start_time = time.perf_counter()

    # remoteNDBC = ERDDAP_Server('https://coastwatch.pfeg.noaa.gov/erddap')
    remote = ERDDAP_Griddap('https://coastwatch.pfeg.noaa.gov/erddap', 'NWW3_Global_Best')
    # print("remoteNDBC:", remote)
    # print(remote.dimensions)
    # print(remote.dimensions['depth'].data)
    # print("variables:", remote.variables.keys())
    # print("summary:", remote.info['summary'])
    # print("start:", remote.getAttribute('time_coverage_start'))


    ds = (
        remote.setResultVariables(['Tper']).setSubset(
            time=slice('2011-01-01','2011-01-02'),
            depth=0,
            latitude=slice(-77.5,77.5),
            longitude=slice(0, 359.5)
        )
        .getData(filetype='nc')
    )


    print("Writing to file" )
    fb = open(path,'wb')
    fb.write(ds)
    fb.close()

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")


fig = plt.figure(figsize=(16/2,10/1.5/2))
mean_Tper = ds['Tper'].mean('time')
mean_Tper = mean_Tper.isel(depth=0)
mean_Tper = mean_Tper.transpose('latitude', 'longitude')
mean_Tper.plot.contourf(x='longitude', y='latitude')
plt.show()

# date = datetime.datetime(2011,1,1)
# fn = mio.request_data(date, 23, ['Tper'])
# # fn = '2011-01-01.nc'
# ds = xr.open_dataset(fn).squeeze().drop('depth')
# fig = plt.figure(figsize=(16/2,10/1.5/2))
# # ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
# p = ds['Tper'].mean('time').plot.contourf(x='longitude',
#                                           y='latitude',)
#                                         #   ax=ax) 
# plt.title('Average wave period on {}'.format(date.strftime('%Y-%m-%d')))
# plt.show()