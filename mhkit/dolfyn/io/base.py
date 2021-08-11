import numpy as np
import xarray as xr
import six
import json
import os
import warnings


def _find_userdata(filename, userdata=True):
    """
    Reads a userdata.json file and returns the data it contains as a
    dictionary.
    """
    # This function finds the file to read
    if userdata:
        for basefile in [filename.rsplit('.', 1)[0],
                         filename]:
            jsonfile = basefile + '.userdata.json'
            if os.path.isfile(jsonfile):
                return _read_userdata(jsonfile)

    elif isinstance(userdata, (six.string_types)) or hasattr(userdata, 'read'):
        return _read_userdata(userdata)
    return {}


def _read_userdata(fname):
    # This one actually does the read
    with open(fname) as data_file:
        data = json.load(data_file)
    for nm in ['body2head_rotmat', 'body2head_vec']:
        if nm in data:
            new_name = 'inst' + nm[4:]
            warnings.warn(
                f'{nm} has been deprecated, please change this to {new_name} \
                    in {fname}.')
            data[new_name] = data.pop(nm)
    if 'inst2head_rotmat' in data and \
       data['inst2head_rotmat'] in ['identity', 'eye', 1, 1.]:
        data['inst2head_rotmat'] = np.eye(3)
    for nm in ['inst2head_rotmat', 'inst2head_vec']:
        if nm in data:
            data[nm] = np.array(data[nm])
    if 'coord_sys' in data:
        raise Exception("The instrument coordinate system "
                        "('coord_sys') should not be specified in "
                        "the .userdata.json file, remove this and "
                        "read the file again.")
    return data


def _handle_nan(data):
    """
    Finds nan's that cause issues in running the rotation algorithms
    and deletes them. 
    
    """
    nan = np.zeros(data['coords']['time'].shape, dtype=bool)
    l = data['coords']['time'].size
    
    if any(np.isnan(data['coords']['time'])):
        nan += np.isnan(data['coords']['time'])
    
    var = ['accel', 'angrt', 'mag']
    for key in data['data_vars']:
        if any(val in key for val in var):
            shp = data['data_vars'][key].shape
            if shp[-1]==l:
                if len(shp)==1:
                    if any(np.isnan(data['data_vars'][key])):
                        nan += np.isnan(data['data_vars'][key])
                elif len(shp)==2:
                    if any(np.isnan(data['data_vars'][key][-1])):
                        nan += np.isnan(data['data_vars'][key][-1])

    if nan.sum()>0:
        data['coords']['time'] = data['coords']['time'][~nan]
        for key in data['data_vars']:
            if data['data_vars'][key].shape[-1]==l:
                data['data_vars'][key] = data['data_vars'][key][...,~nan]
    return data


def _create_dataset(data):
    """
    Creates an xarray dataset from dictionary created from binary
    readers
    
    """
    ds = xr.Dataset()
    beam = list(range(1,data['data_vars']['vel'].shape[0]+1))
    tag = ['_b5', '_echo', '_bt', '_gps']
    # 'dir' coordinates get reset in _set_coords()
    for key in data['data_vars']:
        # orientation matrices
        if 'mat' in key:
            try: # orientmat (inst2earth)
                if any(val in key for val in tag):
                    tg = [val for val in tag if val in key]
                    tg = tg[0]
                else:
                    tg = ''
                ds[key] = xr.DataArray(data['data_vars'][key],
                                       coords={'inst':['X','Y','Z'],
                                               'earth':['E','N','U'],
                                               'time'+tg:data['coords']['time'+tg]},
                                       dims=['inst','earth','time'+tg])
            except: # the other 2 (beam2inst & inst2head)
                ds[key] = xr.DataArray(data['data_vars'][key],
                                       coords={'x':beam,
                                               'x*':beam},
                                       dims=['x','x*'])
        # quaternion units never change
        elif 'quat' in key: 
            if any(val in key for val in tag):
                tg = [val for val in tag if val in key]
                tg = tg[0]
            else:
                tg = ''
            ds[key] = xr.DataArray(data['data_vars'][key],
                                   coords={'q':['w','x','y','z'],
                                           'time'+tg:data['coords']['time'+tg]},
                                   dims=['q','time'+tg])
        else:
            ds[key] = xr.DataArray(data['data_vars'][key])
            try: # not all variables have units
                ds[key].attrs['units'] = data['units'][key]
            except: # make sure ones with tags get units
                if any(val in key for val in tag):
                    if 'echo' not in key:
                        ds[key].attrs['units'] = data['units'][key[:-3]]
                    else:
                        ds[key].attrs['units'] = data['units'][key[:-5]]
                else:
                    pass
            
            shp = data['data_vars'][key].shape
            vshp = data['data_vars']['vel'].shape
            l = len(shp)
            if l==1: # 1D variables
                if any(val in key for val in tag):
                    tg = [val for val in tag if val in key]
                    tg = tg[0]
                else:
                    tg = ''
                ds[key] = ds[key].rename({'dim_0':'time'+tg})
                ds[key] = ds[key].assign_coords({'time'+tg:data['coords']['time'+tg]})
                
            elif l==2: # 2D variables
                if key=='echo':
                    ds[key] = ds[key].rename({'dim_0':'range_echo',
                                              'dim_1':'time_echo'})
                    ds[key] = ds[key].assign_coords({'range_echo':data['coords']['range_echo'],
                                                     'time_echo':data['coords']['time_echo']})
                # 3- & 4-beam instrument vector data, bottom tracking
                elif shp[0]==vshp[0] and not any(val in key for val in tag[:2]):
                    if 'bt' in key and 'time_bt' in data['coords']: # b/c rdi time
                        tg = '_bt'
                    else:
                        tg = ''
                    ds[key] = ds[key].rename({'dim_0':'dir',
                                              'dim_1':'time'+tg})
                    ds[key] = ds[key].assign_coords({'dir':beam,
                                                     'time'+tg:data['coords']['time'+tg]})
                # 4-beam instrument IMU data
                elif shp[0]==vshp[0]-1:
                    if not any(val in key for val in tag):
                        tg = ''
                    else:
                        tg = [val for val in tag if val in key]
                        tg = tg[0]
                        
                    ds[key] = ds[key].rename({'dim_0':'dirIMU',
                                              'dim_1':'time'+tg})
                    ds[key] = ds[key].assign_coords({'dirIMU':[1,2,3],
                                                     'time'+tg:data['coords']['time'+tg]})                            
                
                # b5 and echo tagged variables
                elif any(val in key for val in tag[:2]):
                    tg = [val for val in tag if val in key]
                    tg = tg[0]

                    ds[key] = ds[key].rename({'dim_0':'range'+tg,
                                              'dim_1':'time'+tg})
                    ds[key] = ds[key].assign_coords({'range'+tg:data['coords']['range'+tg],
                                                     'time'+tg:data['coords']['time'+tg]})
                else:
                    warnings.warn('Variable not included in dataset: {}'
                                  .format(key))

            elif l==3: # 3D variables
                if not any(val in key for val in tag):
                    ds[key] = ds[key].rename({'dim_0':'dir',
                                              'dim_1':'range',
                                              'dim_2':'time'})
                    ds[key] = ds[key].assign_coords({'dir':beam,
                                                     'range':data['coords']['range'],
                                                     'time':data['coords']['time']})
                elif 'b5' in key:
                    ds[key] = ds[key][0] # xarray can't handle coords of length 1
                    ds[key] = ds[key].rename({'dim_1':'range_b5',
                                              'dim_2':'time_b5'})
                    ds[key] = ds[key].assign_coords({'range_b5':data['coords']['range_b5'],
                                                     'time_b5':data['coords']['time_b5']})
                else:
                    warnings.warn('Variable not included in dataset: {}'
                                  .format(key))
    # coordinate units
    r_list = [r for r in ds.coords if 'range' in r]
    for ky in r_list:
        ds[ky].attrs['units'] = 'm'
    
    t_list = [t for t in ds.coords if 'time' in t]
    for ky in t_list:
        ds[ky].attrs['description'] = 'seconds since 1970-01-01T00:00:00'
        
    ds.attrs = data['attrs']

    return ds