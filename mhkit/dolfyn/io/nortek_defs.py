import numpy as np

nan = np.nan

class VarAtts(object):
    """A data variable attributes class.

    Parameters
    ----------

    dims : (list, optional)
        The dimensions of the array other than the 'time'
        dimension. By default the time dimension is appended to the
        end. To specify a point to place it, place 'n' in that
        location.

    dtype : (type, optional)
        The data type of the array to create (default: float32).

    group : (string, optional)
        The data group to which this variable should be a part
        (default: 'main').

    view_type : (type, optional)
        Specify a numpy view to cast the array into.

    default_val : (numeric, optional)
        The value to initialize with (default: use an empty array).

    offset : (numeric, optional)
        The offset, 'b', by which to adjust the data when converting to
        scientific units.

    factor : (numeric, optional)
        The factor, 'm', by which to adjust the data when converting to
        scientific units.

    title_name : (string, optional)
        The name of the variable\*\*.

    units : (:class:`<ma.unitsDict>`, optional)
        The units of this variable\*\*.

    dim_names : (list, optional)
        A list of names for each dimension of the array\*\*.

    """
    def __init__(self, dims=[], dtype=None, group='main',
                 view_type=None, default_val=None,
                 offset=0, factor=1,
                 title_name=None, units=None, dim_names=None,
                 ):
        self.dims = list(dims)
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype
        self.group = group
        self.view_type = view_type
        self.default_val = default_val
        self.offset = offset
        self.factor = factor
        self.title_name = title_name
        self.units = units
        self.dim_names = dim_names

    def shape(self, **kwargs):
        a = list(self.dims)
        hit = False
        for ky in kwargs:
            if ky in self.dims:
                hit = True
                a[a.index(ky)] = kwargs[ky]
        if hit:
            return a
        else:
            #try:
            return self.dims + [kwargs['n']]
            #except:
            #    return self.dims

    def _empty_array(self, **kwargs):
        out = np.zeros(self.shape(**kwargs), dtype=self.dtype)
        try:
            out[:] = np.NaN
        except:
            pass
        if self.view_type is not None:
            out = out.view(self.view_type)
        # if self.default_val is not None:
        #     out[:] = self.default_val
        return out

    def sci_func(self, data):
        """Scale the data to scientific units.

        Parameters
        ----------
        data : :class:`<numpy.ndarray>`
            The data to scale.

        Returns
        -------
        retval : {None, data}
          If this funciton modifies the data in place it returns None,
          otherwise it returns the new data object.
        """
        if self.offset != 0:
            data += self.offset
        if self.factor != 1:
            data *= self.factor
            return data


vec_data = {
    'AnaIn2LSB': VarAtts(dims=[],
                         dtype=np.uint8,
                         group='sys',
                         units='',
                         ),
    'Count': VarAtts(dims=[],
                     dtype=np.uint8,
                     group='sys',
                     units='',
                     ),
    'PressureMSB': VarAtts(dims=[],
                           dtype=np.uint8,
                           group='data_vars',
                           units='dbar',
                           ),
    'AnaIn2MSB': VarAtts(dims=[],
                         dtype=np.uint8,
                         group='sys',
                         units='',
                         ),
    'PressureLSW': VarAtts(dims=[],
                           dtype=np.uint16,
                           group='data_vars',
                           units='dbar',
                           ),
    'AnaIn1': VarAtts(dims=[],
                      dtype=np.uint16,
                      group='sys',
                      units=''
                      ),
    'vel': VarAtts(dims=[3],
                   dtype=np.float32,
                   group='data_vars',
                   factor=0.001,
                   default_val=nan,
                   units='m/s',
                   ),
    'amp': VarAtts(dims=[3],
                   dtype=np.uint8,
                   group='data_vars',
                   units='dB',
                   ),
    'corr': VarAtts(dims=[3],
                    dtype=np.uint8,
                    group='data_vars',
                    units='%',
                    ),
}

vec_sysdata = {
    'time': VarAtts(dims=[],
                       dtype=np.float64,
                       group='coords',
                       default_val=nan,
                       units='',
                       ),
    'batt': VarAtts(dims=[],
                    dtype=np.float32,
                    group='sys',
                    default_val=nan,
                    factor=0.1,
                    units='V',
                    ),
    'c_sound': VarAtts(dims=[],
                       dtype=np.float32,
                       group='data_vars',
                       default_val=nan,
                       factor=0.1,
                       units='m/s',
                       ),
    'heading': VarAtts(dims=[],
                       dtype=np.float32,
                       group='data_vars',
                       default_val=nan,
                       factor=0.1,
                       units='deg',
                       ),
    'pitch': VarAtts(dims=[],
                     dtype=np.float32,
                     group='data_vars',
                     default_val=nan,
                     factor=0.1,
                     units='deg',
                     ),
    'roll': VarAtts(dims=[],
                    dtype=np.float32,
                    group='data_vars',
                    default_val=nan,
                    factor=0.1,
                    units='deg',
                    ),
    'temp': VarAtts(dims=[],
                    dtype=np.float32,
                    group='data_vars',
                    default_val=nan,
                    factor=0.01,
                    units='deg C',
                    ),
    'error': VarAtts(dims=[],
                     dtype=np.uint8,
                     group='sys',
                     default_val=nan,
                     units='',
                     ),
    'status': VarAtts(dims=[],
                      dtype=np.uint8,
                      group='sys',
                      default_val=nan,
                      units='',
                      ),
    'AnaIn': VarAtts(dims=[],
                     dtype=np.float32,
                     group='sys',
                     default_val=nan,
                     units='',
                     ),
}

awac_profile = {
    'time': VarAtts(dims=[],
                       dtype=np.float64,
                       group='coords',
                       units='',
                       ),
    'Error': VarAtts(dims=[],
                     dtype=np.uint16,
                     group='sys',
                     units='',
                     ),
    'AnaIn1': VarAtts(dims=[],
                      dtype=np.float32,
                      group='sys',
                      default_val=nan,
                      units='n/a',
                      ),
    'batt': VarAtts(dims=[],
                    dtype=np.float32,
                    group='sys',
                    default_val=nan,
                    factor=0.1,
                    units='V',
                    ),
    'c_sound': VarAtts(dims=[],
                       dtype=np.float32,
                       group='data_vars',
                       default_val=nan,
                       factor=0.1,
                       units='m/s',
                       ),
    'heading': VarAtts(dims=[],
                       dtype=np.float32,
                       group='data_vars',
                       default_val=nan,
                       factor=0.1,
                       units='deg',
                       ),
    'pitch': VarAtts(dims=[],
                     dtype=np.float32,
                     group='data_vars',
                     default_val=nan,
                     factor=0.1,
                     units='deg',
                     ),
    'roll': VarAtts(dims=[],
                    dtype=np.float32,
                    group='data_vars',
                    default_val=nan,
                    factor=0.1,
                    units='deg',
                    ),
    'pressure': VarAtts(dims=[],
                        dtype=np.float32,
                        group='data_vars',
                        default_val=nan,
                        factor=0.001,
                        units='dbar',
                        ),
    'status': VarAtts(dims=[],
                      dtype=np.float32,
                      group='sys',
                      default_val=nan,
                      units='',
                      ),
    'temp': VarAtts(dims=[],
                    dtype=np.float32,
                    group='data_vars',
                    default_val=nan,
                    factor=0.01,
                    units='deg C',
                    ),
    'vel': VarAtts(dims=[3, 'nbins', 'n'], # how to change this for different # of beams?
                   dtype=np.float32,
                   group='data_vars',
                   default_val=nan,
                   factor=0.001,
                   units='m/s',
                   ),
    'amp': VarAtts(dims=[3, 'nbins', 'n'],
                   dtype=np.uint8,
                   group='data_vars',
                   units='counts',
                   ),
}
