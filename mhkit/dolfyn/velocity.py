import numpy as np
import xarray as xr
from .binned import TimeBinner
from .time import dt642epoch, dt642date
from .rotate.api import rotate2, set_declination, set_inst2head_rotmat
from .io.api import save


@xr.register_dataset_accessor('velds')  # 'vel dataset'
class Velocity():
    """All ADCP and ADV xarray datasets wrap this base class.

    The turbulence-related attributes defined within this class 
    assume that the  ``'tke_vec'`` and ``'stress'`` data entries are 
    included in the dataset. These are typically calculated using a
    :class:`VelBinner` tool, but the method for calculating these
    variables can depend on the details of the measurement
    (instrument, it's configuration, orientation, etc.).

    See Also
    ========
    :class:`VelBinner`

    """

    ########
    # Major components of the dolfyn-API

    def rotate2(self, out_frame='earth', inplace=True):
        """Rotate the dataset to a new coordinate system.

        Parameters
        ----------
        out_frame : string {'beam', 'inst', 'earth', 'principal'}
          The coordinate system to rotate the data into.

        inplace : bool (default: True)
          When True the existing data object is modified. When False
          a copy is returned.

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when ``inplace=False``**, otherwise
          returns None.

        Notes
        -----
        - This function rotates all variables in ``ds.attrs['rotate_vars']``.

        - To rotate to the 'principal' frame, a value of
          ``ds.attrs['principal_heading']`` must exist. The function
          :func:`calc_principal_heading <dolfyn.calc_principal_heading>`
          is recommended for this purpose, e.g.::

              ds.attrs['principal_heading'] = dolfyn.calc_principal_heading(ds['vel'].mean(range))

          where here we are using the depth-averaged velocity to calculate
          the principal direction.

        """
        return rotate2(self.ds, out_frame, inplace)

    def set_declination(self, declin, inplace=True):
        """Set the magnetic declination

        Parameters
        ----------
        declination : float
          The value of the magnetic declination in degrees (positive
          values specify that Magnetic North is clockwise from True North)

        inplace : bool (default: True)
          When True the existing data object is modified. When False
          a copy is returned.

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when ``inplace=False``**, otherwise
          returns None.

        Notes
        -----
        This method modifies the data object in the following ways:

        - If the dataset is in the *earth* reference frame at the time of
        setting declination, it will be rotated into the "*True-East*,
        *True-North*, Up" (hereafter, ETU) coordinate system

        - ``dat['orientmat']`` is modified to be an ETU to
        instrument (XYZ) rotation matrix (rather than the magnetic-ENU to
        XYZ rotation matrix). Therefore, all rotations to/from the 'earth'
        frame will now be to/from this ETU coordinate system.

        - The value of the specified declination will be stored in
        ``dat.attrs['declination']``

        - ``dat['heading']`` is adjusted for declination
        (i.e., it is relative to True North).

        - If ``dat.attrs['principal_heading']`` is set, it is
        adjusted to account for the orientation of the new 'True'
        earth coordinate system (i.e., calling set_declination on a
        data object in the principal coordinate system, then calling
        dat.rotate2('earth') will yield a data object in the new
        'True' earth coordinate system)

        """
        return set_declination(self.ds, declin, inplace)

    def set_inst2head_rotmat(self, rotmat, inplace=True):
        """
        Set the instrument to head rotation matrix for the Nortek ADV if it
        hasn't already been set through a '.userdata.json' file.

        Parameters
        ----------
        rotmat : float
            3x3 rotation matrix
        inplace : bool (default: True)
            When True the existing data object is rotated. When False
            a copy is returned that is rotated.

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when ``inplace=False``**, otherwise
          returns None.

        Notes
        -----
        If the data object is in earth or principal coords, it is first
        rotated to 'inst' before assigning inst2head_rotmat, it is then
        rotated back to the coordinate system in which it was input. This
        way the inst2head_rotmat gets applied correctly (in inst
        coordinate system).

        """
        return set_inst2head_rotmat(self.ds, rotmat, inplace)

    def save(self, filename, **kwargs):
        """Save the data object (underlying xarray dataset) as netCDF (.nc).

        Parameters
        ----------
        filename : str
            Filename and/or path with the '.nc' extension
        **kwargs : these are passed directly to :func:`xarray.Dataset.to_netcdf`.

        Notes
        -----
        See |dlfn|'s :func:`save <dolfyn.io.api.save>` function for
        additional details.

        """
        save(self.ds, filename, **kwargs)
    
    ########
    # Magic methods of the API

    def __init__(self, ds, *args, **kwargs):
        self.ds = ds

    def __getitem__(self, key):
        return self.ds[key]

    def __contains__(self, val):
        return val in self.ds

    def __repr__(self, ):
        time_string = '{:.2f} {} (started: {})'
        if ('time' not in self or dt642epoch(self['time'][0]) < 1):
            time_string = '-->No Time Information!<--'
        else:
            tm = self['time'][[0, -1]].values
            dt = dt642date(tm[0])[0]
            delta = (dt642epoch(tm[-1]) -
                     dt642epoch(tm[0])) / (3600 * 24)  # days
            if delta > 1:
                units = 'days'
            elif delta * 24 > 1:
                units = 'hours'
                delta *= 24
            elif delta * 24 * 60 > 1:
                delta *= 24 * 60
                units = 'minutes'
            else:
                delta *= 24 * 3600
                units = 'seconds'
            try:
                time_string = time_string.format(delta, units,
                                                 dt.strftime('%b %d, %Y %H:%M'))
            except AttributeError:
                time_string = '-->Error in time info<--'

        p = self.ds.attrs
        t_shape = self['time'].shape
        if len(t_shape) > 1:
            shape_string = '({} bins, {} pings @ {}Hz)'.format(
                t_shape[0], t_shape, p.get('fs'))
        else:
            shape_string = '({} pings @ {}Hz)'.format(
                t_shape[0], p.get('fs', '??'))
        _header = ("<%s data object>: "
                   " %s %s\n"
                   "  . %s\n"
                   "  . %s-frame\n"
                   "  . %s\n" %
                   (p.get('inst_type'),
                    self.ds.attrs['inst_make'], self.ds.attrs['inst_model'],
                    time_string,
                    p.get('coord_sys'),
                    shape_string))
        _vars = '  Variables:\n'

        # Specify which variable show up in this view here.
        # * indicates a wildcard
        # This list also sets the display order.
        # Only the first 12 matches are displayed.
        show_vars = ['time*', 'vel*', 'range', 'range_echo',
                     'orientmat', 'heading', 'pitch', 'roll',
                     'temp', 'press*', 'amp*', 'corr*',
                     'accel', 'angrt', 'mag',
                     'echo',
                     ]
        n = 0
        for v in show_vars:
            if n > 12:
                break
            if v.endswith('*'):
                v = v[:-1]  # Drop the '*'
                for nm in self.variables:
                    if n > 12:
                        break
                    if nm.startswith(v):
                        n += 1
                        _vars += '  - {} {}\n'.format(nm, self.ds[nm].dims)
            elif v in self.ds:
                _vars += '  - {} {}\n'.format(v, self.ds[v].dims)
        if n < len(self.variables):
            _vars += '  ... and others (see `<obj>.variables`)\n'
        return _header + _vars

    ######
    # Duplicate valuable xarray properties here.
    @property
    def variables(self, ):
        """A sorted list of the variable names in the dataset."""
        return sorted(self.ds.variables)

    @property
    def attrs(self, ):
        """The attributes in the dataset."""
        return self.ds.attrs

    @property
    def coords(self, ):
        """The coordinates in the dataset."""
        return self.ds.coords

    ######
    # A bunch of DOLfYN specific properties
    @property
    def u(self,):
        """The first velocity component.

        This is simply a shortcut to self['vel'][0]. Therefore,
        depending on the coordinate system of the data object
        (self.attrs['coord_sys']), it is:

        - beam:      beam1
        - inst:      x
        - earth:     east
        - principal: streamwise
        """
        return self.ds['vel'][0]

    @property
    def v(self,):
        """The second velocity component.

        This is simply a shortcut to self['vel'][1]. Therefore,
        depending on the coordinate system of the data object
        (self.attrs['coord_sys']), it is:

        - beam:      beam2
        - inst:      y
        - earth:     north
        - principal: cross-stream
        """
        return self.ds['vel'][1]

    @property
    def w(self,):
        """The third velocity component.

        This is simply a shortcut to self['vel'][2]. Therefore,
        depending on the coordinate system of the data object
        (self.attrs['coord_sys']), it is:

        - beam:      beam3
        - inst:      z
        - earth:     up
        - principal: up
        """
        return self.ds['vel'][2]

    @property
    def U(self,):
        """Horizontal velocity as a complex quantity
        """
        return xr.DataArray(
            (self.u + self.v * 1j),
            attrs={'units': 'm/s',
                   'description': 'horizontal velocity (complex)'})

    @property
    def U_mag(self,):
        """Horizontal velocity magnitude
        """
        return xr.DataArray(
            np.abs(self.U),
            attrs={'units': 'm/s',
                   'description': 'horizontal velocity magnitude'})

    @property
    def U_dir(self,):
        """Angle of horizontal velocity vector, degrees counterclockwise from
        X/East/streamwise. Direction is 'to', as opposed to 'from'.
        """
        # Convert from radians to degrees
        angle = np.angle(self.U)*(180/np.pi)

        return xr.DataArray(angle,
                            dims=self.U.dims,
                            coords=self.U.coords,
                            attrs={'units': 'deg',
                                   'description': 'horizontal velocity flow direction, CCW from X/East/streamwise'})

    @property
    def E_coh(self,):
        """Coherent turbulent energy

        Niel Kelley's 'coherent turbulence energy', which is the RMS
        of the Reynold's stresses.

        See: NREL Technical Report TP-500-52353
        """
        E_coh = (self.upwp_**2 + self.upvp_**2 + self.vpwp_**2) ** (0.5)

        return xr.DataArray(E_coh,
                            coords={'time': self.ds['stress'].time},
                            dims=['time'],
                            attrs={'units': self.ds['stress'].units},
                            name='E_coh')

    @property
    def I_tke(self, thresh=0):
        """Turbulent kinetic energy intensity.

        Ratio of sqrt(tke) to horizontal velocity magnitude.
        """
        I_tke = np.ma.masked_where(self.U_mag < thresh,
                                   np.sqrt(2 * self.tke) / self.U_mag)
        return xr.DataArray(I_tke.data,
                            coords=self.U_mag.coords,
                            dims=self.U_mag.dims,
                            attrs={'units': '% [0,1]'},
                            name='TKE intensity')

    @property
    def I(self, thresh=0):
        """Turbulence intensity.

        Ratio of standard deviation of horizontal velocity std dev
        to horizontal velocity magnitude.
        """
        I = np.ma.masked_where(self.U_mag < thresh,
                               self.ds['U_std'] / self.U_mag)
        return xr.DataArray(I.data,
                            coords=self.U_mag.coords,
                            dims=self.U_mag.dims,
                            attrs={'units': '% [0,1]'},
                            name='turbulence intensity')

    @property
    def tke(self,):
        """Turbulent kinetic energy (sum of the three components)
        """
        tke = self.ds['tke_vec'].sum('tke') / 2
        tke.name = 'TKE'
        tke.attrs['units'] = self.ds['tke_vec'].units
        return tke

    @property
    def upvp_(self,):
        """u'v'bar Reynolds stress
        """
        return self.ds['stress'].sel(tau="upvp_")

    @property
    def upwp_(self,):
        """u'w'bar Reynolds stress
        """
        return self.ds['stress'].sel(tau="upwp_")

    @property
    def vpwp_(self,):
        """v'w'bar Reynolds stress
        """
        return self.ds['stress'].sel(tau="vpwp_")

    @property
    def upup_(self,):
        """u'u'bar component of the tke
        """
        return self.ds['tke_vec'].sel(tke="upup_")

    @property
    def vpvp_(self,):
        """v'v'bar component of the tke
        """
        return self.ds['tke_vec'].sel(tke="vpvp_")

    @property
    def wpwp_(self,):
        """w'w'bar component of the tke
        """
        return self.ds['tke_vec'].sel(tke="wpwp_")


class VelBinner(TimeBinner):
    """This is the base binning (averaging) tool.
    All |dlfn| binning tools derive from this base class.

    Examples
    ========
    The VelBinner class is used to compute averages and turbulence
    statistics from 'raw' (not averaged) ADV or ADP measurements, for
    example::

        # First read or load some data.
        rawdat = dlfn.read_example('BenchFile01.ad2cp')

        # Now initialize the averaging tool:
        binner = dlfn.VelBinner(n_bin=600, fs=rawdat.fs)

        # This computes the basic averages
        avg = binner.do_avg(rawdat)

    """
    # This defines how cross-spectra and stresses are computed.
    _cross_pairs = [(0, 1), (0, 2), (1, 2)]

    def do_tke(self, ds, out_ds=None):
        """Calculate the tke (variances of u,v,w) and stresses 
        (cross-covariances of u,v,w)

        Parameters
        ----------
        ds : xarray.Dataset
            Xarray dataset containing raw velocity data
        out_ds : xarray.Dataset
            Averaged dataset to save tke and stress dataArrays to, 
            nominally dataset output from `do_avg()`.

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing tke and stress dataArrays

        """
        props = {}
        if out_ds is None:
            out_ds = type(ds)()
            props['fs'] = self.fs
            props['n_bin'] = self.n_bin
            props['n_fft'] = self.n_fft
            out_ds.attrs = props

        out_ds['tke_vec'] = self.calc_tke(ds['vel'])
        out_ds['stress'] = self.calc_stress(ds['vel'])

        return out_ds

    def calc_tke(self, veldat, noise=[0, 0, 0], detrend=True):
        """Calculate the tke (variances of u,v,w).

        Parameters
        ----------
        veldat : xarray.DataArray
            a velocity data array. The last dimension is assumed
            to be time.
        noise : float
            a three-element vector of the noise levels of the
            velocity data for ach component of velocity.
        detrend : bool (default: False)
            detrend the velocity data (True), or simply de-mean it
            (False), prior to computing tke. Note: the psd routines
            use detrend, so if you want to have the same amount of
            variance here as there use ``detrend=True``.

        Returns
        -------
        ds : xarray.DataArray
            dataArray containing u'u'_, v'v'_ and w'w'_

        """
        if 'dir' in veldat.dims:
            vel = veldat[:3].values
        else:  # for single beam input
            vel = veldat.values

        if detrend:
            vel = self._detrend(vel)
        else:
            vel = self._demean(vel)

        if 'b5' in veldat.name:
            time = self._mean(veldat.time_b5.values)
        else:
            time = self._mean(veldat.time.values)

        out = np.nanmean(vel**2, -1,
                         dtype=np.float64,
                         ).astype('float32')

        out[0] -= noise[0] ** 2
        out[1] -= noise[1] ** 2
        out[2] -= noise[2] ** 2

        da = xr.DataArray(out, name='tke_vec',
                          dims=veldat.dims,
                          attrs={'units': 'm^2/^2'})

        if 'dir' in veldat.dims:
            da = da.rename({'dir': 'tke'})
            da = da.assign_coords({'tke': ["upup_", "vpvp_", "wpwp_"],
                                   'time': time})
        else:
            if 'b5' in veldat.name:
                da = da.assign_coords({'time_b5': time})
            else:
                da = da.assign_coords({'time': time})

        return da

    def calc_stress(self, veldat, detrend=True):
        """Calculate the stresses (cross-covariances of u,v,w)

        Parameters
        ----------
        veldat : xr.DataArray
            A velocity data array. The last dimension is assumed
            to be time.
        detrend : bool (default: True)
            detrend the velocity data (True), or simply de-mean it
            (False), prior to computing stress. Note: the psd routines
            use detrend, so if you want to have the same amount of
            variance here as there use ``detrend=True``.

        Returns
        -------
        ds : xarray.DataArray

        """
        time = self._mean(veldat.time.values)
        vel = veldat.values

        out = np.empty(self._outshape(vel[:3].shape)[:-1],
                       dtype=np.float32)

        if detrend:
            vel = self._detrend(vel)
        else:
            vel = self._demean(vel)

        for idx, p in enumerate(self._cross_pairs):
            out[idx] = np.nanmean(vel[p[0]] * vel[p[1]],
                                  -1, dtype=np.float64
                                  ).astype(np.float32)

        da = xr.DataArray(out, name='stress',
                          dims=veldat.dims,
                          attrs={'units': 'm^2/^2'})
        da = da.rename({'dir': 'tau'})
        da = da.assign_coords({'tau': ["upvp_", "upwp_", "vpwp_"],
                               'time': time})
        return da

    def calc_psd(self, veldat,
                 freq_units='Hz',
                 fs=None,
                 window='hann',
                 noise=[0, 0, 0],
                 n_bin=None, n_fft=None, n_pad=None,
                 step=None):
        """Calculate the power spectral density of velocity.

        Parameters
        ----------
        veldat : xr.DataArray
          The raw velocity data (of dims 'dir' and 'time').
        freq_units : string
          Frequency units of the returned spectra in either Hz or rad/s 
          (`f` or :math:`\\omega`)
        fs : float (optional)
          The sample rate (default: from the binner).
        window : string or array
          Specify the window function.
        noise : list(3 floats) (optional)
          Noise level of each component's velocity measurement
          (default 0).
        n_bin : int (optional)
          The bin-size (default: from the binner).
        n_fft : int (optional)
          The fft size (default: from the binner).
        n_pad : int (optional)
          The number of values to pad with zero (default: 0)
        step : int (optional)
          Controls amount of overlap in fft (default: the step size is
          chosen to maximize data use, minimize nens, and have a
          minimum of 50% overlap.).

        Returns
        -------
        psd : xarray.DataArray (3, M, N_FFT)
          The spectra in the 'u', 'v', and 'w' directions.

        """
        try:
            time = self._mean(veldat.time.values)
            time_str = 'time'
        except:
            time = self._mean(veldat.time_b5.values)
            time_str = 'time_b5'
        fs = self._parse_fs(fs)
        n_fft = self._parse_nfft(n_fft)
        veldat = veldat.values

        # Create frequency vector, also checks whether using f or omega
        freq = self.calc_freq(units=freq_units)
        if 'rad' in freq_units:
            fs = 2*np.pi*fs
            freq_units = 'rad/s'
            units = 'm^2/s/rad'
            f_key = 'omega'
        else:
            freq_units = 'Hz'
            units = 'm^2/s^2/Hz'
            f_key = 'f'

        # Spectra, if input is full velocity or a single array
        if len(veldat.shape) == 2:
            assert veldat.shape[0]==3, "Function can only handle 1D or 3D arrays"

            out = np.empty(self._outshape_fft(veldat[:3].shape),
                           dtype=np.float32)
            for idx in range(3):
                out[idx] = self._psd(veldat[idx], fs=fs, noise=noise[idx],
                                     window=window, n_bin=n_bin,
                                     n_pad=n_pad, n_fft=n_fft, step=step)
            coords = {'S': ['Sxx', 'Syy', 'Szz'], time_str: time, f_key: freq}
            dims = ['S', time_str, f_key]
        else:
            out = self._psd(veldat, fs=fs, noise=noise[0], window=window,
                            n_bin=n_bin, n_pad=n_pad, n_fft=n_fft, step=step)
            coords = {time_str: time, f_key: freq}
            dims = [time_str, f_key]

        da = xr.DataArray(out,
                          name='psd',
                          coords=coords,
                          dims=dims,
                          attrs={'units': units, 'n_fft': n_fft})
        da[f_key].attrs['units'] = freq_units

        return da

    def calc_csd(self, veldat,
                 freq_units='Hz',
                 fs=None,
                 window='hann',
                 n_bin=None,
                 n_fft_coh=None):
        """Calculate the cross-spectral density of velocity components.

        Parameters
        ----------
        veldat   : xarray.DataArray
          The raw 3D velocity data.
        freq_units : string
          Frequency units of the returned spectra in either Hz or rad/s 
          (`f` or :math:`\\omega`)
        fs : float (optional)
          The sample rate (default: from the binner).
        window : string or array
          Specify the window function.
        n_bin : int (optional)
          The bin-size (default: from the binner).
        n_fft_coh : int (optional)
          The fft size (default: n_fft_coh from the binner).

        Returns
        -------
        csd : xarray.DataArray (3, M, N_FFT)
          The first-dimension of the cross-spectrum is the three
          different cross-spectra: 'uv', 'uw', 'vw'.

        """
        fs = self._parse_fs(fs)
        n_fft = self._parse_nfft_coh(n_fft_coh)
        time = self._mean(veldat.time.values)
        veldat = veldat.values

        out = np.empty(self._outshape_fft(veldat[:3].shape, n_fft=n_fft),
                       dtype='complex')

        # Create frequency vector, also checks whether using f or omega
        coh_freq = self.calc_freq(units=freq_units, coh=True)
        if 'rad' in freq_units:
            fs = 2*np.pi*fs
            freq_units = 'rad/s'
            units = 'm^2/s/rad'
            f_key = 'omega'
        else:
            freq_units = 'Hz'
            units = 'm^2/s^2/Hz'
            f_key = 'f'

        for ip, ipair in enumerate(self._cross_pairs):
            out[ip] = self._cpsd(veldat[ipair[0]],
                                 veldat[ipair[1]],
                                 n_bin=n_bin,
                                 n_fft=n_fft,
                                 window=window)

        da = xr.DataArray(out,
                          name='csd',
                          coords={'C': ['Cxy', 'Cxz', 'Cyz'],
                                  'time': time,
                                  f_key: coh_freq},
                          dims=['C', 'time', f_key],
                          attrs={'units': units, 'n_fft_coh': n_fft})
        da[f_key].attrs['units'] = freq_units

        return da
