import numpy as np
import xarray as xr
import json
import os
import warnings


def _abspath(fname):
    return os.path.abspath(os.path.expanduser(fname))


def _get_filetype(fname):
    """
    Detects whether the file is a Nortek, Signature (Nortek), or RDI
    file by reading the first few bytes of the file.

    Returns
    =======
       None - Doesn't match any known pattern
       'signature' - for Nortek signature files
       'nortek' - for Nortek (Vec, AWAC) files
       'RDI' - for RDI files
       '<GIT-LFS pointer> - if the file looks like a GIT-LFS pointer.
    """

    with open(fname, 'rb') as rdr:
        bytes = rdr.read(40)
    code = bytes[:2].hex()
    if code in ['7f79', '7f7f']:
        return 'RDI'
    elif code in ['a50a']:
        return 'signature'
    elif code in ['a505']:
        # AWAC
        return 'nortek'
    elif bytes == b'version https://git-lfs.github.com/spec/':
        return '<GIT-LFS pointer>'
    else:
        return None


def _find_userdata(filename, userdata=True):
    # This function finds the file to read
    if userdata:
        for basefile in [filename.rsplit('.', 1)[0],
                         filename]:
            jsonfile = basefile + '.userdata.json'
            if os.path.isfile(jsonfile):
                return _read_userdata(jsonfile)

    elif isinstance(userdata, (str, )) or hasattr(userdata, 'read'):
        return _read_userdata(userdata)
    return {}


def _read_userdata(fname):
    """
    Reads a userdata.json file and returns the data it contains as a
    dictionary.
    """
    with open(fname) as data_file:
        data = json.load(data_file)
    for nm in ['body2head_rotmat', 'body2head_vec']:
        if nm in data:
            new_name = 'inst' + nm[4:]
            warnings.warn(
                f'{nm} has been deprecated, please change this to {new_name} \
                    in {fname}.')
            data[new_name] = data.pop(nm)
    if 'inst2head_rotmat' in data:
        if data['inst2head_rotmat'] in ['identity', 'eye', 1, 1.]:
            data['inst2head_rotmat'] = np.eye(3)
        else:
            data['inst2head_rotmat'] = np.array(data['inst2head_rotmat'])
    if 'inst2head_vec' in data and type(data['inst2head_vec']) != list:
        data['inst2head_vec'] = list(data['inst2head_vec'])

    return data


def _handle_nan(data):
    """
    Finds trailing nan's that cause issues in running the rotation 
    algorithms and deletes them.
    """
    nan = np.zeros(data['coords']['time'].shape, dtype=bool)
    l = data['coords']['time'].size

    if any(np.isnan(data['coords']['time'])):
        nan += np.isnan(data['coords']['time'])

    # Required for motion-correction algorithm
    var = ['accel', 'angrt', 'mag']
    for key in data['data_vars']:
        if any(val in key for val in var):
            shp = data['data_vars'][key].shape
            if shp[-1] == l:
                if len(shp) == 1:
                    if any(np.isnan(data['data_vars'][key])):
                        nan += np.isnan(data['data_vars'][key])
                elif len(shp) == 2:
                    if any(np.isnan(data['data_vars'][key][-1])):
                        nan += np.isnan(data['data_vars'][key][-1])
    trailing = np.cumsum(nan)[-1]

    if trailing > 0:
        data['coords']['time'] = data['coords']['time'][:-trailing]
        for key in data['data_vars']:
            if data['data_vars'][key].shape[-1] == l:
                data['data_vars'][key] = data['data_vars'][key][..., :-trailing]

    return data


def _create_dataset(data):
    """Creates an xarray dataset from dictionary created from binary
    readers.
    Direction 'dir' coordinates are set in `set_coords`
    """
    ds = xr.Dataset()
    tag = ['_avg', '_b5', '_echo', '_bt', '_gps', '_ast', '_sl']

    FoR = {}
    try:
        beams = data['attrs']['n_beams']
    except:
        beams = data['attrs']['n_beams_avg']
    n_beams = max(min(beams, 4), 3)
    beams = np.arange(1, n_beams+1, dtype=np.int32)
    FoR['beam'] = xr.DataArray(beams, dims=['beam'], name='beam', attrs={
                               'units': '1', 'long_name': 'Beam Reference Frame'})
    FoR['dir'] = xr.DataArray(beams, dims=['dir'], name='dir', attrs={
                              'units': '1', 'long_name': 'Reference Frame'})

    for key in data['data_vars']:
        # orientation matrices
        if 'mat' in key:
            if 'inst' in key:  # beam2inst & inst2head orientation matrices
                ds[key] = xr.DataArray(data['data_vars'][key],
                                       coords={'x1': beams, 'x2': beams},
                                       dims=['x1', 'x2'],
                                       attrs={'units': '1',
                                              'long_name': 'Rotation Matrix'})
            elif 'orientmat' in key:  # earth2inst orientation matrix
                if any(val in key for val in tag):
                    tg = '_' + key.rsplit('_')[-1]
                else:
                    tg = ''
                earth = xr.DataArray(['E', 'N', 'U'], dims=['earth'], name='earth', attrs={
                    'units': '1', 'long_name': 'Earth Reference Frame'})
                inst = xr.DataArray(['X', 'Y', 'Z'], dims=['inst'], name='inst', attrs={
                    'units': '1', 'long_name': 'Instrument Reference Frame'})
                time = data['coords']['time'+tg]
                ds[key] = xr.DataArray(data['data_vars'][key],
                                       coords={'earth': earth,
                                               'inst': inst, 'time'+tg: time},
                                       dims=['earth', 'inst', 'time'+tg],
                                       attrs={'units': data['units']['orientmat'],
                                              'long_name': data['long_name']['orientmat']})

        # quaternion units never change
        elif 'quaternions' in key:
            if any(val in key for val in tag):
                tg = '_' + key.rsplit('_')[-1]
            else:
                tg = ''
            q = xr.DataArray(['w', 'x', 'y', 'z'], dims=['q'], name='q', attrs={
                             'units': '1', 'long_name': 'Quaternion Vector Components'})
            time = data['coords']['time'+tg]
            ds[key] = xr.DataArray(data['data_vars'][key],
                                   coords={'q': q,
                                           'time'+tg: time},
                                   dims=['q', 'time'+tg],
                                   attrs={'units': data['units']['quaternions'],
                                          'long_name': data['long_name']['quaternions']})
        else:
            # Assign each variable to a dataArray
            ds[key] = xr.DataArray(data['data_vars'][key])
            # Assign metadata to each dataArray
            for md in ['units', 'long_name', 'standard_name']:
                if key in data[md]:
                    ds[key].attrs[md] = data[md][key]
                try:  # make sure ones with tags get units
                    tg = '_' + key.rsplit('_')[-1]
                    if any(val in key for val in tag):
                        ds[key].attrs[md] = data[md][key[:-len(tg)]]
                except:
                    pass

            # Fill in dimensions and coordinates for each dataArray
            shp = data['data_vars'][key].shape
            l = len(shp)
            if l == 1:  # 1D variables
                if any(val in key for val in tag):
                    tg = '_' + key.rsplit('_')[-1]
                else:
                    tg = ''
                ds[key] = ds[key].rename({'dim_0': 'time'+tg})
                ds[key] = ds[key].assign_coords(
                    {'time'+tg: data['coords']['time'+tg]})

            elif l == 2:  # 2D variables
                if key == 'echo':
                    ds[key] = ds[key].rename({'dim_0': 'range_echo',
                                              'dim_1': 'time_echo'})
                    ds[key] = ds[key].assign_coords({'range_echo': data['coords']['range_echo'],
                                                     'time_echo': data['coords']['time_echo']})
                # ADV/ADCP instrument vector data, bottom tracking
                elif shp[0] == n_beams and not any(val in key for val in tag[:3]):
                    if 'bt' in key and 'time_bt' in data['coords']:
                        tg = '_bt'
                    else:
                        tg = ''
                    if any(key.rsplit('_')[0] in s for s in ['amp', 'corr', 'dist', 'prcnt_gd']):
                        dim0 = 'beam'
                    else:
                        dim0 = 'dir'
                    ds[key] = ds[key].rename({'dim_0': dim0,
                                              'dim_1': 'time'+tg})
                    ds[key] = ds[key].assign_coords({dim0: FoR[dim0],
                                                     'time'+tg: data['coords']['time'+tg]})
                # ADCP IMU data
                elif shp[0] == 3:
                    if not any(val in key for val in tag):
                        tg = ''
                    else:
                        tg = [val for val in tag if val in key]
                        tg = tg[0]
                    dirIMU = xr.DataArray([1, 2, 3], dims=['dirIMU'], name='dirIMU', attrs={
                        'units': '1', 'long_name': 'Reference Frame'})
                    ds[key] = ds[key].rename({'dim_0': 'dirIMU',
                                              'dim_1': 'time'+tg})
                    ds[key] = ds[key].assign_coords({'dirIMU': dirIMU,
                                                     'time'+tg: data['coords']['time'+tg]})

                ds[key].attrs['coverage_content_type'] = 'physicalMeasurement'

            elif l == 3:  # 3D variables
                if 'vel' in key:
                    dim0 = 'dir'
                else:  # amp, corr, prcnt_gd, status
                    dim0 = 'beam'

                if not any(val in key for val in tag) or ('_avg' in key):
                    if '_avg' in key:
                        tg = '_avg'
                    else:
                        tg = ''
                    ds[key] = ds[key].rename({'dim_0': dim0,
                                              'dim_1': 'range'+tg,
                                              'dim_2': 'time'+tg})
                    ds[key] = ds[key].assign_coords({dim0: FoR[dim0],
                                                     'range'+tg: data['coords']['range'+tg],
                                                     'time'+tg: data['coords']['time'+tg]})
                elif 'b5' in key:
                    # xarray can't handle coords of length 1
                    ds[key] = ds[key][0]
                    ds[key] = ds[key].rename({'dim_1': 'range_b5',
                                              'dim_2': 'time_b5'})
                    ds[key] = ds[key].assign_coords({'range_b5': data['coords']['range_b5'],
                                                     'time_b5': data['coords']['time_b5']})
                elif 'sl' in key:
                    ds[key] = ds[key].rename({'dim_0': dim0,
                                              'dim_1': 'range_sl',
                                              'dim_2': 'time'})
                    ds[key] = ds[key].assign_coords({'range_sl': data['coords']['range_sl'],
                                                     'time': data['coords']['time']})
                else:
                    ds = ds.drop_vars(key)
                    warnings.warn(f'Variable not included in dataset: {key}')

                ds[key].attrs['coverage_content_type'] = 'physicalMeasurement'

    # coordinate attributes
    for ky in ds.dims:
        ds[ky].attrs['coverage_content_type'] = 'coordinate'
    r_list = [r for r in ds.coords if 'range' in r]
    for ky in r_list:
        ds[ky].attrs['units'] = 'm'
        ds[ky].attrs['long_name'] = 'Profile Range'
        ds[ky].attrs['description'] = 'Distance to the center of each depth bin'
    time_list = [t for t in ds.coords if 'time' in t]
    for ky in time_list:
        ds[ky].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        ds[ky].attrs['long_name'] = 'Time'
        ds[ky].attrs['standard_name'] = 'time'

    # dataset metadata
    ds.attrs = data['attrs']

    return ds
