import scipy.io as sio
import xarray as xr
import pkg_resources

from .nortek import read_nortek
from .nortek2 import read_signature
from .rdi import read_rdi
from .base import create_dataset, WrongFileType as _WTF
from ..rotate.base import _set_coords


def read(fname, userdata=True, nens=None):
    """Read a binary Nortek (e.g., .VEC, .wpr, .ad2cp, etc.) or RDI
    (.000, .PD0, .ENX, etc.) data file.

    Parameters
    ----------
    filename : string
               Filename of instrument file to read.

    userdata : True, False, or string of userdata.json filename
               (default ``True``) Whether to read the
               '<base-filename>.userdata.json' file.

    nens : None (default: read entire file), int, or
           2-element tuple (start, stop)
           Number of pings or ensembles to read from the file

    Returns
    -------
    ds : xr.Dataset
         An xarray dataset from the binary instrument data.

    """
    # Loop over binary readers until we find one that works.
    for func in [read_nortek, read_signature, read_rdi]:
        try:
            ds = func(fname, userdata=userdata, nens=nens)
        except _WTF:
            continue
        else:
            return ds
    raise _WTF("Unable to find a suitable reader for "
               "file {}.".format(fname))


def read_example(name, **kwargs):
    """Read an example data file.

    Parameters
    ----------
    name : string
        Available files:

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
    dat : xr.Dataset

    """
    filename = pkg_resources.resource_filename(
        'dolfyn',
        'example_data/' + name)
    return read(filename, **kwargs)


def save(dataset, filename):
    """
    Save xarray dataset as netCDF (.nc).
    
    Parameters
    ----------
    dataset : xr.Dataset
    
    filename : str
        Filename and/or path with the '.nc' extension
    
    Notes
    -----
    Drops 'config' lines.
    
    """
    if filename[-3:] != '.nc':
        filename += '.nc'
    
    for key in list(dataset.attrs.keys()):
        if 'config' in key:
            dataset.attrs.pop(key)
    
    dataset.to_netcdf(filename, 
                      format='NETCDF4', 
                      engine='h5netcdf', 
                      invalid_netcdf=True)
    

def load(filename):
    """
    Load xarray dataset from netCDF ('.nc')
    
    Parameters
    ----------
    filename : str
        Filename and/or path with the '.nc' extension
    
    """
    if filename[-3:] != '.nc':
        filename += '.nc'
        
    # this engine reorders attributes into alphabetical order
    ds = xr.load_dataset(filename, engine='h5netcdf')
    
    # xarray converts single list items to ints or strings
    if hasattr(ds, 'rotate_vars') and len(ds.rotate_vars)==1:
        ds.attrs['rotate_vars'] = list(ds.rotate_vars)
        
    # reloads lists as numpy arrays???
    if hasattr(ds, 'rotate_vars') and type(ds.rotate_vars) is not list:
        ds.attrs['rotate_vars'] = list(ds.rotate_vars)
    
    return ds


def save_mat(dataset, filename):
    """
    Save xarray dataset as a MATLAB (.mat) file
    
    Parameters
    ----------
    dataset : xr.Dataset
    
    filename : str
        Filename and/or path with the '.mat' extension
    
    """
    if filename[-4:] != '.mat':
        filename += '.mat'
    
    matfile = {'vars':{},'coords':{},'config':{},'units':{}}
    for key in dataset.data_vars:
        matfile['vars'][key] = dataset[key].values
        if hasattr(dataset[key], 'units'):
            matfile['units'][key] = dataset[key].units
    for key in dataset.coords:
        matfile['coords'][key] = dataset[key].values
    matfile['config'] = dataset.attrs
    
    sio.savemat(filename, matfile)
    
    
def load_mat(filename):
    """
    Load xarray dataset from MATLAB (.mat) file
    
    .mat file must contain the fields: {vars, coords, config, units},
    where 'coords' contain the dimensions of all variables in 'vars'.
    
    Parameters
    ----------
    filename : str
        Filename and/or path with the '.mat' extension
    
    """
    if filename[-4:] != '.mat':
        filename += '.mat'
    
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    ds_dict = {'vars':{},'coords':{},'config':{},'units':{}}
    
    for nm in ds_dict:
        key_list = data[nm]._fieldnames
        for ky in key_list:
            ds_dict[nm][ky] = getattr(data[nm], ky)
    
    ds_dict['data_vars'] = ds_dict.pop('vars')
    ds_dict['attrs'] = ds_dict.pop('config')
    
    ds = create_dataset(ds_dict)
    ds = _set_coords(ds, ds.coord_sys)
    ds.attrs['rotate_vars'] = [x.strip(' ') for x in list(ds.rotate_vars)]
    
    return ds