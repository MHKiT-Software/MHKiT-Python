import numpy as np
import xarray as xr
import warnings
from .tools.psd import psd_freq, coherence, psd, cpsd_quasisync, cpsd, \
    phase_angle
from .tools.misc import slice1d_along_axis, detrend
from .time import epoch2dt64, dt642epoch
warnings.simplefilter('ignore', RuntimeWarning)


class TimeBinner:
    def __init__(self, n_bin, fs, n_fft=None, n_fft_coh=None,
                 noise=[0, 0, 0]):
        """Initialize an averaging object

        Parameters
        ----------
        n_bin : int
          Number of data points to include in a 'bin' (ensemble), not the 
          number of bins
        fs : int
          Instrument sampling frequency in Hz
        n_fft : int
          Number of data points to use for fft (`n_fft`<=`n_bin`).
          Default: `n_fft`=`n_bin`
        n_fft_coh : int
          Number of data points to use for coherence and cross-spectra ffts
          Default: `n_fft_coh`=`n_fft`
        noise : list or ndarray
          Instrument's doppler noise in same units as velocity

        """
        self.n_bin = n_bin
        self.fs = fs
        self.n_fft = n_fft
        self.n_fft_coh = n_fft_coh
        self.noise = noise
        if n_fft is None:
            self.n_fft = n_bin
        elif n_fft > n_bin:
            self.n_fft = n_bin
            warnings.warn(
                "n_fft must be smaller than n_bin, setting n_fft = n_bin")
        if n_fft_coh is None:
            self.n_fft_coh = int(self.n_fft)
        elif n_fft_coh > n_bin:
            self.n_fft_coh = int(n_bin // 6)
            warnings.warn("n_fft_coh must be smaller than or equal to n_bin, "
                          "setting n_fft_coh = n_bin/6")

    def _outshape(self, inshape, n_pad=0, n_bin=None):
        """Returns `outshape` (the 'reshape'd shape) for an `inshape` array.
        """
        n_bin = int(self._parse_nbin(n_bin))
        return list(inshape[:-1]) + [int(inshape[-1] // n_bin), int(n_bin + n_pad)]

    def _outshape_fft(self, inshape, n_fft=None, n_bin=None):
        """Returns `outshape` (the fft 'reshape'd shape) for an `inshape` array.
        """
        n_fft = self._parse_nfft(n_fft)
        n_bin = self._parse_nbin(n_bin)
        return list(inshape[:-1]) + [int(inshape[-1] // n_bin), int(n_fft // 2)]

    def _parse_fs(self, fs=None):
        if fs is not None:
            return fs
        return self.fs

    def _parse_nbin(self, n_bin=None):
        if n_bin is None:
            return self.n_bin
        return n_bin

    def _parse_nfft(self, n_fft=None):
        if n_fft is None:
            return self.n_fft
        return n_fft

    def _parse_nfft_coh(self, n_fft_coh=None):
        if n_fft_coh is None:
            return self.n_fft_coh
        return n_fft_coh

    def reshape(self, arr, n_pad=0, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad).

        Parameters
        ----------
        arr : numpy.ndarray
        n_pad : int
          Is used to add `n_pad`/2 points from the end of the previous
          ensemble to the top of the current, and `n_pad`/2 points
          from the top of the next ensemble to the bottom of the
          current.  Zeros are padded in the upper-left and lower-right
          corners of the matrix (beginning/end of timeseries).  In
          this case, the array shape will be (...,`n`,`n_pad`+`n_bin`)
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        Notes
        -----
        `n_bin` can be non-integer, in which case the output array
        size will be `n_pad`+`n_bin`, and the decimal will
        cause skipping of some data points in `arr`.  In particular,
        every mod(`n_bin`,1) bins will have a skipped point. For
        example:
        - for n_bin=2048.2 every 1/5 bins will have a skipped point.
        - for n_bin=4096.9 every 9/10 bins will have a skipped point.

        """
        n_bin = self._parse_nbin(n_bin)
        npd0 = int(n_pad // 2)
        npd1 = int((n_pad + 1) // 2)
        shp = self._outshape(arr.shape, n_pad=0, n_bin=n_bin)
        out = np.zeros(
            self._outshape(arr.shape, n_pad=n_pad, n_bin=n_bin),
            dtype=arr.dtype)
        if np.mod(n_bin, 1) == 0:
            # n_bin needs to be int
            n_bin = int(n_bin)
            # If n_bin is an integer, we can do this simply.
            out[..., npd0: n_bin + npd0] = (
                arr[..., :(shp[-2] * shp[-1])]).reshape(shp, order='C')
        else:
            inds = (np.arange(np.prod(shp[-2:])) * n_bin // int(n_bin)
                    ).astype(int)
            n_bin = int(n_bin)
            out[..., npd0:n_bin + npd0] = (arr[..., inds]
                                           ).reshape(shp, order='C')
            n_bin = int(n_bin)
        if n_pad != 0:
            out[..., 1:, :npd0] = out[..., :-1, n_bin:n_bin + npd0]
            out[..., :-1, -npd1:] = out[..., 1:, npd0:npd0 + npd1]

        return out

    def detrend(self, arr, axis=-1, n_pad=0, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and remove the best-fit trend line from each bin.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int (default is -1)
          Axis along which to take mean
        n_pad : int (default is 0)
          Is used to add `n_pad`/2 points from the end of the previous
          ensemble to the top of the current, and `n_pad`/2 points
          from the top of the next ensemble to the bottom of the
          current.  Zeros are padded in the upper-left and lower-right
          corners of the matrix (beginning/end of timeseries).  In
          this case, the array shape will be (...,`n`,`n_pad`+`n_bin`)
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        """
        return detrend(self.reshape(arr, n_pad=n_pad, n_bin=n_bin), axis=axis)

    def demean(self, arr, axis=-1, n_pad=0, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and remove the mean from each bin.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int (default is -1)
          Axis along which to take mean
        n_pad : int (default is 0)
          Is used to add `n_pad`/2 points from the end of the previous
          ensemble to the top of the current, and `n_pad`/2 points
          from the top of the next ensemble to the bottom of the
          current.  Zeros are padded in the upper-left and lower-right
          corners of the matrix (beginning/end of timeseries).  In
          this case, the array shape will be (...,`n`,`n_pad`+`n_bin`)
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        """
        dt = self.reshape(arr, n_pad=n_pad, n_bin=n_bin)
        return dt - np.nanmean(dt, axis)[..., None]

    def mean(self, arr, axis=-1, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the mean of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int (default is -1)
          Axis along which to take mean
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        """
        if np.issubdtype(arr.dtype, np.datetime64):
            return epoch2dt64(self.mean(dt642epoch(arr), axis=axis, n_bin=n_bin))
        if axis != -1:
            arr = np.swapaxes(arr, axis, -1)
        n_bin = self._parse_nbin(n_bin)
        tmp = self.reshape(arr, n_bin=n_bin)

        return np.nanmean(tmp, -1)

    def var(self, arr, axis=-1, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the variance of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int (default is -1)
          Axis along which to take variance
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        """
        return self.reshape(arr, n_bin=n_bin).var(axis)

    def std(self, arr, axis=-1, n_bin=None):
        """Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the standard deviation of each bin along the 
        specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int (default is -1)
          Axis along which to take std dev
        n_bin : int (default is self.n_bin)
          Override this binner's n_bin.

        Returns
        -------
        out : numpy.ndarray

        """
        return self.reshape(arr, n_bin=n_bin).std(axis)

    def do_avg(self, raw_ds, out_ds=None, names=None, noise=[0, 0, 0]):
        """Bin the dataset and calculate the ensemble averages of each 
        variable.

        Parameters
        ----------
        raw_ds : xarray.Dataset
           The raw data structure to be binned
        out_ds : xarray.Dataset
           The bin'd (output) data object to which averaged data is added.
        names : list of strings
           The names of variables to be averaged.  If `names` is None,
           all data in `raw_ds` will be binned.
        noise : list or numpy.ndarray
          instrument's doppler noise in same units as velocity

        Returns
        -------
        out_ds : xarray.Dataset
          The new (or updated when out_ds is not None) dataset
          with the averages of all the variables in raw_ds.

        Raises
        ------
        AttributeError : when out_ds is supplied as input (not None)
        and the values in out_ds.attrs are inconsistent with
        raw_ds.attrs or the properties of this VelBinner (n_bin,
        n_fft, fs, etc.)

        Notes
        -----
        raw_ds.attrs are copied to out_ds.attrs. Inconsistencies
        between the two (when out_ds is specified as input) raise an
        AttributeError.

        """
        out_ds = self._check_ds(raw_ds, out_ds)

        if names is None:
            names = raw_ds.data_vars

        for ky in names:
            # set up dimensions and coordinates for Dataset
            dims_list = raw_ds[ky].dims
            coords_dict = {}
            for nm in dims_list:
                if 'time' in nm:
                    coords_dict[nm] = self.mean(raw_ds[ky][nm].values)
                else:
                    coords_dict[nm] = raw_ds[ky][nm].values

            # create Dataset
            if 'ensemble' not in ky:
                try:  # variables with time coordinate
                    out_ds[ky] = xr.DataArray(self.mean(raw_ds[ky].values),
                                              coords=coords_dict,
                                              dims=dims_list,
                                              attrs=raw_ds[ky].attrs)
                except:  # variables not needing averaging
                    pass
            # Add standard deviation
            std = (np.nanstd(self.reshape(raw_ds.velds.U_mag.values),
                             axis=-1,
                             dtype=np.float64) - (noise[0] + noise[1])/2)
            out_ds['U_std'] = xr.DataArray(
                std,
                dims=raw_ds.vel.dims[1:],
                attrs={'units': 'm/s',
                       'description': 'horizontal velocity std dev'})

        return out_ds

    def do_var(self, raw_ds, out_ds=None, names=None, suffix='_var'):
        """Bin the dataset and calculate the ensemble variances of each 
        variable. Complementary to `do_avg()`.

        Parameters
        ----------
        raw_ds : xarray.Dataset
           The raw data structure to be binned.
        out_ds : xarray.Dataset
           The binned (output) dataset to which variance data is added,
           nominally dataset output from `do_avg()`
        names : list of strings
           The names of variables of which to calculate variance.  If
           `names` is None, all data in `raw_ds` will be binned.

        Returns
        -------
        out_ds : xarray.Dataset
          The new (or updated when out_ds is not None) dataset
          with the variance of all the variables in raw_ds.

        Raises
        ------
        AttributeError : when out_ds is supplied as input (not None)
        and the values in out_ds.attrs are inconsistent with
        raw_ds.attrs or the properties of this VelBinner (n_bin,
        n_fft, fs, etc.)

        Notes
        -----
        raw_ds.attrs are copied to out_ds.attrs. Inconsistencies
        between the two (when out_ds is specified as input) raise an
        AttributeError.

        """
        out_ds = self._check_ds(raw_ds, out_ds)

        if names is None:
            names = raw_ds.data_vars

        for ky in names:
            # set up dimensions and coordinates for dataarray
            dims_list = raw_ds[ky].dims
            coords_dict = {}
            for nm in dims_list:
                if 'time' in nm:
                    coords_dict[nm] = self.mean(raw_ds[ky][nm].values)
                else:
                    coords_dict[nm] = raw_ds[ky][nm].values

            # create Dataset
            if 'ensemble' not in ky:
                try:  # variables with time coordinate
                    out_ds[ky+suffix] = xr.DataArray(self.var(raw_ds[ky].values),
                                                     coords=coords_dict,
                                                     dims=dims_list,
                                                     attrs=raw_ds[ky].attrs)
                except:  # variables not needing averaging
                    pass

        return out_ds

    def _check_ds(self, raw_ds, out_ds):
        """Check that the attributes between two datasets match up.

        Parameters
        ----------
        raw_ds : xarray.Dataset
          Input dataset
        out_ds : xarray.Dataset
          Dataset to append `raw_ds` to. If None is supplied, this
          dataset is created from `raw_ds`.

        Returns
        -------
        out_ds : xarray.Dataset

        """
        for v in raw_ds.data_vars:
            if np.any(np.array(raw_ds[v].shape) == 0):
                raise RuntimeError(f"{v} cannot be averaged "
                                   "because it is empty.")
        if 'DutyCycle_NBurst' in raw_ds.attrs and \
                raw_ds.attrs['DutyCycle_NBurst'] < self.n_bin:
            warnings.warn(f"The averaging interval (n_bin = {self.n_bin})"
                          "is larger than the burst interval "
                          "(NBurst = {dat.attrs['DutyCycle_NBurst']})")
        if raw_ds.fs != self.fs:
            raise Exception(f"The input data sample rate ({raw_ds.fs}) does not "
                            "match the sample rate of this binning-object "
                            "({self.fs})")

        if out_ds is None:
            out_ds = type(raw_ds)()

        o_attrs = out_ds.attrs

        props = {}
        props['fs'] = self.fs
        props['n_bin'] = self.n_bin
        props['n_fft'] = self.n_fft
        props['description'] = 'Binned averages calculated from ' \
            'ensembles of size "n_bin"'
        props.update(raw_ds.attrs)

        for ky in props:
            if ky in o_attrs and o_attrs[ky] != props[ky]:
                # The values in out_ds must match `props` (raw_ds.attrs,
                # plus those defined above)
                raise AttributeError(
                    "The attribute '{}' of `out_ds` is inconsistent "
                    "with this `VelBinner` or the input data (`raw_ds`)".format(ky))
            else:
                o_attrs[ky] = props[ky]
        return out_ds

    def _new_coords(self, array):
        """Function for setting up a new xarray.DataArray regardless of how 
        many dimensions the input data-array has
        """
        dims = array.dims
        dims_list = []
        coords_dict = {}
        if len(array.shape) == 1 & ('dir' in array.coords):
            array = array.drop_vars('dir')
        for ky in dims:
            dims_list.append(ky)
            if 'time' in ky:
                coords_dict[ky] = self.mean(array.time.values)
            else:
                coords_dict[ky] = array.coords[ky].values

        return dims_list, coords_dict

    def _calc_lag(self, npt=None, one_sided=False):
        if npt is None:
            npt = self.n_bin
        if one_sided:
            return np.arange(int(npt // 2), dtype=np.float32)
        else:
            return np.arange(npt, dtype=np.float32) - int(npt // 2)

    def calc_coh(self, veldat1, veldat2, window='hann', debias=True,
                 noise=(0, 0), n_fft_coh=None, n_bin=None):
        """Calculate coherence between `veldat1` and `veldat2`.

        Parameters
        ----------
        veldat1 : xarray.DataArray
          The first (the longer, if applicable) raw dataArray of which to 
          calculate coherence
        veldat2 : xarray.DataArray
          The second (the shorter, if applicable) raw dataArray of which to 
          calculate coherence
        window : str
          String indicating the window function to use (default: 'hanning')
        noise : float
          The white-noise level of the measurement (in the same units
          as `veldat`).
        n_fft_coh : int
          n_fft of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner

        Returns
        -------
        da : xarray.DataArray
          The coherence between signal veldat1 and veldat2.

        Notes
        -----
        The two velocity inputs do not have to be perfectly synchronized, but 
        they should have the same start and end timestamps.

        """
        if veldat1.size < veldat2.size:
            raise Exception(
                "veldat1 is shorter than veldat2. Please switch these inputs.")

        dat1 = veldat1.values
        dat2 = veldat2.values

        if n_fft_coh is None:
            n_fft = self.n_fft_coh
        else:
            n_fft = int(n_fft_coh)

        # want each slice to carry the same timespan
        n_bin2 = self._parse_nbin(n_bin)  # bins for shorter array
        n_bin1 = int(dat1.shape[-1]/(dat2.shape[-1]/n_bin2))

        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin1)
        oshp[-2] = np.min([oshp[-2], int(dat2.shape[-1] // n_bin2)])
        out = np.empty(oshp, dtype=dat1.dtype)

        # The data is detrended in psd, so we don't need to do it here.
        dat1 = self.reshape(dat1, n_pad=n_fft, n_bin=n_bin1)
        dat2 = self.reshape(dat2, n_pad=n_fft, n_bin=n_bin2)

        for slc in slice1d_along_axis(out.shape, -1):
            out[slc] = coherence(dat1[slc], dat2[slc], n_fft,
                                 window=window, debias=debias,
                                 noise=noise)

        freq = self.calc_freq(self.fs, coh=True)

        # Get time from shorter vector
        dims_list, coords_dict = self._new_coords(veldat2)
        # tack on new coordinate
        dims_list.append('f')
        coords_dict['f'] = freq

        da = xr.DataArray(out, name='coherence',
                          coords=coords_dict,
                          dims=dims_list)
        da['f'].attrs['units'] = 'Hz'

        return da

    def calc_phase_angle(self, veldat1, veldat2, window='hann',
                         n_fft_coh=None, n_bin=None):
        """Calculate the phase difference between two signals as a
        function of frequency (complimentary to coherence).

        Parameters
        ----------
        veldat1 : xarray.DataArray
          The first (the longer, if applicable) raw dataArray of which to 
          calculate phase angle
        veldat2 : xarray.DataArray
          The second (the shorter, if applicable) raw dataArray of which 
          to calculate phase angle
        window : str
          String indicating the window function to use (default: 'hanning').
        n_fft : int
          Number of elements per bin if 'None' is taken from VelBinner
        n_bin : int
          Number of elements per bin from veldat2 if 'None' is taken 
          from VelBinner

        Returns
        -------
        da : xarray.DataArray
          The phase difference between signal veldat1 and veldat2.

        Notes
        -----
        The two velocity inputs do not have to be perfectly synchronized, but 
        they should have the same start and end timestamps.

        """
        if veldat1.size < veldat2.size:
            raise Exception(
                "veldat1 is shorter than veldat2. Please switch these inputs.")

        dat1 = veldat1.values
        dat2 = veldat2.values

        if n_fft_coh is None:
            n_fft = self.n_fft_coh
        else:
            n_fft = int(n_fft_coh)

        # want each slice to carry the same timespan
        n_bin2 = self._parse_nbin(n_bin)  # bins for shorter array
        n_bin1 = int(dat1.shape[-1]/(dat2.shape[-1]/n_bin2))

        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin1)
        oshp[-2] = np.min([oshp[-2], int(dat2.shape[-1] // n_bin2)])

        # The data is detrended in psd, so we don't need to do it here:
        dat1 = self.reshape(dat1, n_pad=n_fft, n_bin=n_bin1)
        dat2 = self.reshape(dat2, n_pad=n_fft, n_bin=n_bin2)
        out = np.empty(oshp, dtype='c{}'.format(dat2.dtype.itemsize * 2))

        for slc in slice1d_along_axis(out.shape, -1):
            # PSD's are computed in radian units:
            out[slc] = phase_angle(dat1[slc], dat2[slc], n_fft,
                                   window=window)

        freq = self.calc_freq(self.fs, coh=True)

        # Get time from shorter vector
        dims_list, coords_dict = self._new_coords(veldat2)
        # tack on new coordinate
        dims_list.append('f')
        coords_dict['f'] = freq

        da = xr.DataArray(out, name='phase_angle',
                          coords=coords_dict,
                          dims=dims_list)
        da['f'].attrs['units'] = 'Hz'

        return da

    def calc_acov(self, veldat, n_bin=None):
        """Calculate the auto-covariance of the raw-signal `veldat`

        Parameters
        ----------
        veldat : xarray.DataArray
          The raw dataArray of which to calculate auto-covariance
        n_bin : float
          Number of data elements to use

        Returns
        -------
        da : xarray.DataArray
          The auto-covariance of veldat

        Notes
        -----
        As opposed to calc_xcov, which returns the full
        cross-covariance between two arrays, this function only
        returns a quarter of the full auto-covariance. It computes the
        auto-covariance over half of the range, then averages the two
        sides (to return a 'quartered' covariance).

        This has the advantage that the 0 index is actually zero-lag.

        """
        indat = veldat.values

        n_bin = self._parse_nbin(n_bin)
        out = np.empty(self._outshape(indat.shape, n_bin=n_bin)[:-1] +
                       [int(n_bin // 4)], dtype=indat.dtype)
        dt1 = self.reshape(indat, n_pad=n_bin / 2 - 2)
        # Here we de-mean only on the 'valid' range:
        dt1 = dt1 - dt1[..., :, int(n_bin // 4):
                        int(-n_bin // 4)].mean(-1)[..., None]
        dt2 = self.demean(indat)
        se = slice(int(n_bin // 4) - 1, None, 1)
        sb = slice(int(n_bin // 4) - 1, None, -1)
        for slc in slice1d_along_axis(dt1.shape, -1):
            tmp = np.correlate(dt1[slc], dt2[slc], 'valid')
            # The zero-padding in reshape means we compute coherence
            # from one-sided time-series for first and last points.
            if slc[-2] == 0:
                out[slc] = tmp[se]
            elif slc[-2] == dt2.shape[-2] - 1:
                out[slc] = tmp[sb]
            else:
                # For the others we take the average of the two sides.
                out[slc] = (tmp[se] + tmp[sb]) / 2

        dims_list, coords_dict = self._new_coords(veldat)
        # tack on new coordinate
        dims_list.append('dt')
        coords_dict['dt'] = np.arange(n_bin//4)

        da = xr.DataArray(out, name='auto-covariance',
                          coords=coords_dict,
                          dims=dims_list,)
        da['dt'].attrs['units'] = 'timestep'

        return da

    def calc_xcov(self, veldat1, veldat2, npt=1,
                  n_bin=None, normed=False):
        """Calculate the cross-covariance between arrays veldat1 and veldat2

        Parameters
        ----------
        veldat1 : xarray.DataArray
          The first raw dataArray of which to calculate cross-covariance
        veldat2 : xarray.DataArray
          The second raw dataArray of which to calculate cross-covariance
        npt : int
          Number of timesteps (lag) to calculate covariance
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner

        Returns
        -------
        da : xarray.DataArray
          The cross-covariance between signal veldat1 and veldat2.

        Notes
        -----
        The two velocity inputs must be the same length

        """
        dat1 = veldat1.values
        dat2 = veldat2.values

        # want each slice to carry the same timespan
        n_bin2 = self._parse_nbin(n_bin)
        n_bin1 = int(dat1.shape[-1]/(dat2.shape[-1]/n_bin2))

        shp = self._outshape(dat1.shape, n_bin=n_bin1)
        shp[-2] = min(shp[-2], self._outshape(dat2.shape, n_bin=n_bin2)[-2])

        # reshape dat1 to be the same size as dat2
        out = np.empty(shp[:-1] + [npt], dtype=dat1.dtype)
        tmp = int(n_bin2) - int(n_bin1) + npt
        dt1 = self.reshape(dat1, n_pad=tmp-1, n_bin=n_bin1)

        # Note here I am demeaning only on the 'valid' range:
        dt1 = dt1 - dt1[..., :, int(tmp // 2)
                                    :int(-tmp // 2)].mean(-1)[..., None]
        # Don't need to pad the second variable:
        dt2 = self.demean(dat2, n_bin=n_bin2)

        for slc in slice1d_along_axis(shp, -1):
            out[slc] = np.correlate(dt1[slc], dt2[slc], 'valid')
        if normed:
            out /= (self.std(dat1, n_bin=n_bin1)[..., :shp[-2]] *
                    self.std(dat2, n_bin=n_bin2)[..., :shp[-2]] *
                    n_bin2)[..., None]

        dims_list, coords_dict = self._new_coords(veldat1)
        # tack on new coordinate
        dims_list.append('dt')
        coords_dict['dt'] = np.arange(npt)

        da = xr.DataArray(out, name='cross-covariance',
                          coords=coords_dict,
                          dims=dims_list)
        return da

    def _psd(self, dat, fs=None, window='hann', noise=0,
             n_bin=None, n_fft=None, n_pad=None, step=None):
        """Calculate power spectral density of `dat`

        Parameters
        ----------
        dat : xarray.DataArray
          The raw dataArray of which to calculate the psd.
        fs : float (optional)
          The sample rate (Hz).
        window : str
          String indicating the window function to use (default: 'hanning').
        noise  : float
          The white-noise level of the measurement (in the same units
          as `dat`).
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner
        n_pad : int (optional)
          The number of values to pad with zero (default: 0)
        step : int (optional)
          Controls amount of overlap in fft (default: the step size is
          chosen to maximize data use, minimize nens, and have a
          minimum of 50% overlap.).

        """
        fs = self._parse_fs(fs)
        n_bin = self._parse_nbin(n_bin)
        n_fft = self._parse_nfft(n_fft)
        if n_pad is None:
            n_pad = min(n_bin - n_fft, n_fft)
        out = np.empty(self._outshape_fft(dat.shape, n_fft=n_fft, n_bin=n_bin))
        # The data is detrended in psd, so we don't need to do it here.
        dat = self.reshape(dat, n_pad=n_pad)

        for slc in slice1d_along_axis(dat.shape, -1):
            # PSD's are computed in radian units: - set prior to function
            out[slc] = psd(dat[slc], n_fft, fs,
                           window=window, step=step)
        if noise != 0:
            out -= noise**2 / (fs/2)
            # Make sure all values of the PSD are >0 (but still small):
            out[out < 0] = np.min(np.abs(out)) / 100
        return out

    def _cpsd(self, dat1, dat2, fs=None, window='hann',
              n_fft=None, n_bin=None):
        """Calculate the cross power spectral density of `dat`.

        Parameters
        ----------
        dat1 : numpy.ndarray
          The first (shorter, if applicable) raw dataArray of which to 
          calculate the cpsd.
        dat2 : numpy.ndarray
          The second (the shorter, if applicable) raw dataArray of which to 
          calculate the cpsd.
        fs : float (optional)
          The sample rate (Hz).
        window : str
          String indicating the window function to use (default: 'hanning').
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner

        Returns
        -------
        out : numpy.ndarray
          The cross-spectral density of `dat1` and `dat2`

        Notes
        -----
        The two velocity inputs do not have to be perfectly synchronized, but 
        they should have the same start and end timestamps

        """
        fs = self._parse_fs(fs)
        if n_fft is None:
            n_fft = self.n_fft_coh
        # want each slice to carry the same timespan
        n_bin2 = self._parse_nbin(n_bin)  # bins for shorter array
        n_bin1 = int(dat1.shape[-1]/(dat2.shape[-1]/n_bin2))

        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin1)
        oshp[-2] = np.min([oshp[-2], int(dat2.shape[-1] // n_bin2)])

        # The data is detrended in psd, so we don't need to do it here:
        dat1 = self.reshape(dat1, n_pad=n_fft)
        dat2 = self.reshape(dat2, n_pad=n_fft)
        out = np.empty(oshp, dtype='c{}'.format(dat1.dtype.itemsize * 2))
        if dat1.shape == dat2.shape:
            cross = cpsd
        else:
            cross = cpsd_quasisync
        for slc in slice1d_along_axis(out.shape, -1):
            # PSD's are computed in radian units: - set prior to function
            out[slc] = cross(dat1[slc], dat2[slc], n_fft,
                             fs, window=window)
        return out

    def calc_freq(self, fs=None, units='Hz', n_fft=None, coh=False):
        """Calculate the ordinary or radial frequency vector for the PSDs

        Parameters
        ----------
        fs : float (optional)
          The sample rate (Hz).
        units : string
          Frequency units in either Hz or rad/s (f or omega)
        coh : bool
          Calculate the frequency vector for coherence/cross-spectra
          (default: False) i.e. use self.n_fft_coh instead of
          self.n_fft.
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken 
          from VelBinner

        Returns
        -------
        out: numpy.ndarray
          Spectrum frequency array in units of 'Hz' or 'rad/s'
        """
        if n_fft is None:
            n_fft = self.n_fft
            if coh:
                n_fft = self.n_fft_coh

        fs = self._parse_fs(fs)

        if ('Hz' not in units) and ('rad' not in units):
            raise Exception('Valid fft frequency vector units are Hz \
                            or rad/s')

        if 'rad' in units:
            return psd_freq(n_fft, 2*np.pi*fs)
        else:
            return psd_freq(n_fft, fs)
