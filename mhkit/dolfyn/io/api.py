import numpy as np
import scipy.io as sio
import xarray as xr
from os.path import abspath, dirname, join, normpath, relpath
from .nortek import read_nortek
from .nortek2 import read_signature
from .rdi import read_rdi
from .base import _create_dataset, _get_filetype
from ..rotate.base import _set_coords
from ..time import date2matlab, matlab2date, date2dt64, dt642date, date2epoch, epoch2date


def _check_file_ext(path, ext):
    filename = path.replace("\\", "/").rsplit("/")[-1]  # windows/linux
    # for a filename like mcrl.water_velocity-1s.b1.20200813.150000.nc
    file_ext = filename.rsplit(".")[-1]
    if '.' in filename:
        if file_ext != ext:
            raise IOError("File extension must be of the type {}".format(ext))
        if file_ext == ext:
            return path

    return path + '.' + ext


def read(fname, userdata=True, nens=None):
    """Read a binary Nortek (e.g., .VEC, .wpr, .ad2cp, etc.) or RDI
    (.000, .PD0, .ENX, etc.) data file.

    Parameters
    ----------
    filename : string
        Filename of instrument file to read.
    userdata : True, False, or string of userdata.json filename (default ``True``)
        Whether to read the '<base-filename>.userdata.json' file.
    nens : None (default: read entire file), int, or 2-element tuple (start, stop)
        Number of pings or ensembles to read from the file

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from instrument datafile.

    """
    file_type = _get_filetype(fname)
    if file_type == '<GIT-LFS pointer>':
        raise IOError("File '{}' looks like a git-lfs pointer. You may need to "
                      "install and initialize git-lfs. See https://git-lfs.github.com"
                      " for details.".format(fname))
    elif file_type is None:
        raise IOError("File '{}' is not recognized as a file-type that is readable by "
                      "DOLfYN. If you think it should be readable, try using the "
                      "appropriate read function (`read_rdi`, `read_nortek`, or "
                      "`read_signature`) found in dolfyn.io.api.".format(fname))
    else:
        func_map = dict(RDI=read_rdi,
                        nortek=read_nortek,
                        signature=read_signature)
        func = func_map[file_type]
    return func(fname, userdata=userdata, nens=nens)


def read_example(name, **kwargs):
    """Read an ADCP or ADV datafile from the examples directory.

    Parameters
    ----------
    name : str
        A few available files:

            AWAC_test01.wpr
            BenchFile01.ad2cp
            RDI_test01.000
            burst_mode01.VEC
            vector_data01.VEC
            vector_data_imu01.VEC
            winriver01.PD0
            winriver02.PD0

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from the binary instrument data.

    """
    testdir = dirname(abspath(__file__))
    exdir = normpath(join(testdir, relpath('../../../examples/data/dolfyn/')))
    filename = exdir + '/' + name

    return read(filename, **kwargs)


def save(ds, filename,
         format='NETCDF4', engine='netcdf4',
         compression=False,
         **kwargs):
    """Save xarray dataset as netCDF (.nc).

    Parameters
    ----------
    ds : xarray.Dataset
    filename : str
        Filename and/or path with the '.nc' extension
    compression : bool (default: False)
        When true, compress all variables with zlib complevel=1.
    **kwargs : these are passed directly to :func:`xarray.Dataset.to_netcdf`

    Notes
    -----
    Drops 'config' lines.

    More detailed compression options can be specified by specifying
    'encoding' in kwargs. The values in encoding will take precedence
    over whatever is set according to the compression option above.
    See the xarray.to_netcdf documentation for more details.

    """
    filename = _check_file_ext(filename, 'nc')

    # Dropping the detailed configuration stats because netcdf can't save it
    for key in list(ds.attrs.keys()):
        if 'config' in key:
            ds.attrs.pop(key)

    # Handling complex values for netCDF4
    ds.attrs['complex_vars'] = []
    for var in ds.data_vars:
        if np.iscomplexobj(ds[var]):
            ds[var+'_real'] = ds[var].real
            ds[var+'_imag'] = ds[var].imag

            ds = ds.drop_vars(var)
            ds.attrs['complex_vars'].append(var)

    if compression:
        enc = dict()
        for ky in ds.variables:
            enc[ky] = dict(zlib=True, complevel=1)
        if 'encoding' in kwargs:
            # Overwrite ('update') values in enc with whatever is in kwargs['encoding']
            enc.update(kwargs['encoding'])
        else:
            kwargs['encoding'] = enc

    ds.to_netcdf(filename, format=format, engine=engine, **kwargs)


def load(filename):
    """Load xarray dataset from netCDF (.nc)

    Parameters
    ----------
    filename : str
        Filename and/or path with the '.nc' extension

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from the binary instrument data.

    """
    filename = _check_file_ext(filename, 'nc')

    ds = xr.load_dataset(filename, engine='netcdf4')

    # Convert numpy arrays and strings back to lists
    for nm in ds.attrs:
        if type(ds.attrs[nm]) == np.ndarray and ds.attrs[nm].size > 1:
            ds.attrs[nm] = list(ds.attrs[nm])
        elif type(ds.attrs[nm]) == str and nm in ['rotate_vars']:
            ds.attrs[nm] = [ds.attrs[nm]]

    # Rejoin complex numbers
    if hasattr(ds, 'complex_vars') and len(ds.complex_vars):
        if len(ds.complex_vars[0]) == 1:
            ds.attrs['complex_vars'] = [ds.complex_vars]
        for var in ds.complex_vars:
            ds[var] = ds[var+'_real'] + ds[var+'_imag'] * 1j
            ds = ds.drop_vars([var+'_real', var+'_imag'])
    ds.attrs.pop('complex_vars')

    return ds


def save_mat(ds, filename, datenum=True):
    """Save xarray dataset as a MATLAB (.mat) file

    Parameters
    ----------
    ds : xarray.Dataset
        Data to save
    filename : str
        Filename and/or path with the '.mat' extension
    datenum : bool
        If true, converts time to datenum. If false, time will be saved
        in "epoch time".

    Notes
    -----
    The xarray data format is saved as a MATLAB structure with the fields 
    'vars, coords, config, units'.

    See Also
    --------
    scipy.io.savemat()

    """
    filename = _check_file_ext(filename, 'mat')

    # Convert time to datenum
    t_coords = [t for t in ds.coords if np.issubdtype(
        ds[t].dtype, np.datetime64)]
    t_data = [t for t in ds.data_vars if np.issubdtype(
        ds[t].dtype, np.datetime64)]

    if datenum:
        func = date2matlab
    else:
        func = date2epoch

    for ky in t_coords:
        dt = func(dt642date(ds[ky]))
        ds = ds.assign_coords({ky: dt})
    for ky in t_data:
        dt = func(dt642date(ds[ky]))
        ds[ky].data = dt

    ds.attrs['time_coords'] = t_coords
    ds.attrs['time_data_vars'] = t_data

    # Save xarray structure with more descriptive structure names
    matfile = {'vars': {}, 'coords': {}, 'config': {}, 'units': {}}
    for key in ds.data_vars:
        matfile['vars'][key] = ds[key].values
        if hasattr(ds[key], 'units'):
            matfile['units'][key] = ds[key].units
    for key in ds.coords:
        matfile['coords'][key] = ds[key].values
    matfile['config'] = ds.attrs

    sio.savemat(filename, matfile)


def load_mat(filename, datenum=True):
    """Load xarray dataset from MATLAB (.mat) file, complimentary to `save_mat()`

    A .mat file must contain the fields: {vars, coords, config, units},
    where 'coords' contain the dimensions of all variables in 'vars'.

    Parameters
    ----------
    filename : str
        Filename and/or path with the '.mat' extension
    datenum : bool
        If true, converts time from datenum. If false, converts time from 
        "epoch time".

    Returns
    -------
    ds : xarray.Dataset
        An xarray dataset from the binary instrument data.

    See Also
    --------
    scipy.io.loadmat()

    """
    filename = _check_file_ext(filename, 'mat')

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    ds_dict = {'vars': {}, 'coords': {}, 'config': {}, 'units': {}}
    for nm in ds_dict:
        key_list = data[nm]._fieldnames
        for ky in key_list:
            ds_dict[nm][ky] = getattr(data[nm], ky)

    ds_dict['data_vars'] = ds_dict.pop('vars')
    ds_dict['attrs'] = ds_dict.pop('config')

    # Recreate dataset
    ds = _create_dataset(ds_dict)
    ds = _set_coords(ds, ds.coord_sys)

    # Convert numpy arrays and strings back to lists
    for nm in ds.attrs:
        if type(ds.attrs[nm]) == np.ndarray and ds.attrs[nm].size > 1:
            try:
                ds.attrs[nm] = [x.strip(' ') for x in list(ds.attrs[nm])]
            except:
                ds.attrs[nm] = list(ds.attrs[nm])
        elif type(ds.attrs[nm]) == str and nm in ['time_coords', 'time_data_vars', 'rotate_vars']:
            ds.attrs[nm] = [ds.attrs[nm]]

    if hasattr(ds, 'orientation_down'):
        ds['orientation_down'] = ds['orientation_down'].astype(bool)

    if datenum:
        func = matlab2date
    else:
        func = epoch2date

    # Restore datnum to np.dt64
    if hasattr(ds, 'time_coords'):
        for ky in ds.attrs['time_coords']:
            dt = date2dt64(func(ds[ky].values))
            ds = ds.assign_coords({ky: dt})
        ds.attrs.pop('time_coords')
    if hasattr(ds, 'time_data_vars'):
        for ky in ds.attrs['time_data_vars']:
            dt = date2dt64(func(ds[ky].values))
            ds[ky].data = dt
        ds.attrs.pop('time_data_vars')

    return ds
