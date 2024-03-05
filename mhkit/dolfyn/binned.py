import numpy as np
import warnings
from .tools.fft import fft_frequency, psd_1D, cpsd_1D, cpsd_quasisync_1D
from .tools.misc import slice1d_along_axis, detrend_array
from .time import epoch2dt64, dt642epoch

warnings.simplefilter("ignore", RuntimeWarning)


class TimeBinner:
    def __init__(self, n_bin, fs, n_fft=None, n_fft_coh=None, noise=[0, 0, 0]):
        """
        Initialize an averaging object

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
            warnings.warn("n_fft must be smaller than n_bin, setting n_fft = n_bin")
        if n_fft_coh is None:
            self.n_fft_coh = int(self.n_fft)
        elif n_fft_coh > n_bin:
            self.n_fft_coh = int(n_bin)
            warnings.warn(
                "n_fft_coh must be smaller than or equal to n_bin, "
                "setting n_fft_coh = n_bin"
            )

    def _outshape(self, inshape, n_pad=0, n_bin=None):
        """
        Returns `outshape` (the 'reshape'd shape) for an `inshape` array.
        """
        n_bin = int(self._parse_nbin(n_bin))
        return list(inshape[:-1]) + [int(inshape[-1] // n_bin), int(n_bin + n_pad)]

    def _outshape_fft(self, inshape, n_fft=None, n_bin=None):
        """
        Returns `outshape` (the fft 'reshape'd shape) for an `inshape` array.
        """
        n_fft = self._parse_nfft(n_fft)
        n_bin = self._parse_nbin(n_bin)
        return list(inshape[:-1]) + [int(inshape[-1] // n_bin), int(n_fft // 2)]

    def _parse_fs(self, fs=None):
        if fs is None:
            return self.fs
        return fs

    def _parse_nbin(self, n_bin=None):
        if n_bin is None:
            return self.n_bin
        return n_bin

    def _parse_nfft(self, n_fft=None):
        if n_fft is None:
            return self.n_fft
        if n_fft > self.n_bin:
            n_fft = self.n_bin
            warnings.warn("n_fft must be smaller than n_bin, setting n_fft = n_bin")
        return n_fft

    def _parse_nfft_coh(self, n_fft_coh=None):
        if n_fft_coh is None:
            return self.n_fft_coh
        if n_fft_coh > self.n_bin:
            n_fft_coh = int(self.n_bin)
            warnings.warn(
                "n_fft_coh must be smaller than or equal to n_bin, "
                "setting n_fft_coh = n_bin"
            )
        return n_fft_coh

    def _check_ds(self, raw_ds, out_ds):
        """
        Check that the attributes between two datasets match up.

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
                raise RuntimeError(f"{v} cannot be averaged " "because it is empty.")
        if (
            "DutyCycle_NBurst" in raw_ds.attrs
            and raw_ds.attrs["DutyCycle_NBurst"] < self.n_bin
        ):
            warnings.warn(
                f"The averaging interval (n_bin = {self.n_bin})"
                "is larger than the burst interval "
                "(NBurst = {dat.attrs['DutyCycle_NBurst']})"
            )
        if raw_ds.fs != self.fs:
            raise Exception(
                f"The input data sample rate ({raw_ds.fs}) does not "
                "match the sample rate of this binning-object "
                "({self.fs})"
            )

        if out_ds is None:
            out_ds = type(raw_ds)()

        o_attrs = out_ds.attrs

        props = {}
        props["fs"] = self.fs
        props["n_bin"] = self.n_bin
        props["n_fft"] = self.n_fft
        props["description"] = (
            "Binned averages calculated from " 'ensembles of size "n_bin"'
        )
        props.update(raw_ds.attrs)

        for ky in props:
            if ky in o_attrs and o_attrs[ky] != props[ky]:
                # The values in out_ds must match `props` (raw_ds.attrs,
                # plus those defined above)
                raise AttributeError(
                    "The attribute '{}' of `out_ds` is inconsistent "
                    "with this `VelBinner` or the input data (`raw_ds`)".format(ky)
                )
            else:
                o_attrs[ky] = props[ky]
        return out_ds

    def _new_coords(self, array):
        """
        Function for setting up a new xarray.DataArray regardless of how
        many dimensions the input data-array has
        """
        dims = array.dims
        dims_list = []
        coords_dict = {}
        if len(array.shape) == 1 & ("dir" in array.coords):
            array = array.drop_vars("dir")
        for ky in dims:
            dims_list.append(ky)
            if "time" in ky:
                coords_dict[ky] = self.mean(array.time.values)
            else:
                coords_dict[ky] = array.coords[ky].values

        return dims_list, coords_dict

    def reshape(self, arr, n_pad=0, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad).

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
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

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
        if arr.shape[-1] < n_bin:
            raise Exception("n_bin is larger than length of input array")
        npd0 = int(n_pad // 2)
        npd1 = int((n_pad + 1) // 2)
        shp = self._outshape(arr.shape, n_pad=0, n_bin=n_bin)
        out = np.zeros(
            self._outshape(arr.shape, n_pad=n_pad, n_bin=n_bin), dtype=arr.dtype
        )
        if np.mod(n_bin, 1) == 0:
            # n_bin needs to be int
            n_bin = int(n_bin)
            # If n_bin is an integer, we can do this simply.
            out[..., npd0 : n_bin + npd0] = (arr[..., : (shp[-2] * shp[-1])]).reshape(
                shp, order="C"
            )
        else:
            inds = (np.arange(np.prod(shp[-2:])) * n_bin // int(n_bin)).astype(int)
            # If there are too many indices, drop one bin
            if inds[-1] >= arr.shape[-1]:
                inds = inds[: -int(n_bin)]
                shp[-2] -= 1
                out = out[..., 1:, :]
            n_bin = int(n_bin)
            out[..., npd0 : n_bin + npd0] = (arr[..., inds]).reshape(shp, order="C")
            n_bin = int(n_bin)
        if n_pad != 0:
            out[..., 1:, :npd0] = out[..., :-1, n_bin : n_bin + npd0]
            out[..., :-1, -npd1:] = out[..., 1:, npd0 : npd0 + npd1]

        return out

    def detrend(self, arr, axis=-1, n_pad=0, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and remove the best-fit trend line from each bin.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        n_pad : int
          Is used to add `n_pad`/2 points from the end of the previous
          ensemble to the top of the current, and `n_pad`/2 points
          from the top of the next ensemble to the bottom of the
          current.  Zeros are padded in the upper-left and lower-right
          corners of the matrix (beginning/end of timeseries).  In
          this case, the array shape will be (...,`n`,`n_pad`+`n_bin`).
          Default = 0
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return detrend_array(self.reshape(arr, n_pad=n_pad, n_bin=n_bin), axis=axis)

    def demean(self, arr, axis=-1, n_pad=0, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and remove the mean from each bin.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        n_pad : int
          Is used to add `n_pad`/2 points from the end of the previous
          ensemble to the top of the current, and `n_pad`/2 points
          from the top of the next ensemble to the bottom of the
          current.  Zeros are padded in the upper-left and lower-right
          corners of the matrix (beginning/end of timeseries).  In
          this case, the array shape will be (...,`n`,`n_pad`+`n_bin`).
          Default = 0
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        dt = self.reshape(arr, n_pad=n_pad, n_bin=n_bin)
        return dt - np.nanmean(dt, axis)[..., None]

    def mean(self, arr, axis=-1, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the mean of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

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

    def variance(self, arr, axis=-1, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the variance of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take variance. Default = -1
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanvar(self.reshape(arr, n_bin=n_bin), axis=axis, dtype=np.float32)

    def standard_deviation(self, arr, axis=-1, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the standard deviation of each bin along the
        specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take std dev. Default = -1
        n_bin : int
          Override this binner's n_bin. Default is `binner.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanstd(self.reshape(arr, n_bin=n_bin), axis=axis, dtype=np.float32)

    def _psd_base(
        self,
        dat,
        fs=None,
        window="hann",
        noise=0,
        n_bin=None,
        n_fft=None,
        n_pad=None,
        step=None,
    ):
        """
        Calculate power spectral density of `dat`

        Parameters
        ----------
        dat : xarray.DataArray
          The raw dataArray of which to calculate the psd.
        fs : float (optional)
          The sample rate (Hz).
        window : str
          String indicating the window function to use. Default is 'hanning'
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
          The number of values to pad with zero. Default = 0
        step : int (optional)
          Controls amount of overlap in fft. Default: the step size is
          chosen to maximize data use, minimize nens, and have a
          minimum of 50% overlap.

        Returns
        -------
        out : numpy.ndarray
          The power spectral density of `dat`

        Notes
        -----
        PSD's are calculated based on sample rate units
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
            out[slc] = psd_1D(dat[slc], n_fft, fs, window=window, step=step)
        if np.any(noise):
            out -= noise**2 / (fs / 2)
            # Make sure all values of the PSD are >0 (but still small):
            out[out < 0] = np.min(np.abs(out)) / 100
        return out

    def _csd_base(self, dat1, dat2, fs=None, window="hann", n_fft=None, n_bin=None):
        """
        Calculate the cross power spectral density of `dat`.

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
          String indicating the window function to use. Default is 'hanning'
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken
          from VelBinner
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken
          from VelBinner

        Returns
        -------
        out : numpy.ndarray
          The cross power spectral density of `dat1` and `dat2`

        Notes
        -----
        PSD's are calculated based on sample rate units

        The two velocity inputs do not have to be perfectly synchronized, but
        they should have the same start and end timestamps
        """

        fs = self._parse_fs(fs)
        if n_fft is None:
            n_fft = self.n_fft_coh
        # want each slice to carry the same timespan
        n_bin2 = self._parse_nbin(n_bin)  # bins for shorter array
        n_bin1 = int(dat1.shape[-1] / (dat2.shape[-1] / n_bin2))

        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin1)
        oshp[-2] = np.min([oshp[-2], int(dat2.shape[-1] // n_bin2)])

        # The data is detrended in psd, so we don't need to do it here:
        dat1 = self.reshape(dat1, n_pad=n_fft)
        dat2 = self.reshape(dat2, n_pad=n_fft)
        out = np.empty(oshp, dtype="c{}".format(dat1.dtype.itemsize * 2))
        if dat1.shape == dat2.shape:
            cross = cpsd_1D
        else:
            cross = cpsd_quasisync_1D
        for slc in slice1d_along_axis(out.shape, -1):
            out[slc] = cross(dat1[slc], dat2[slc], n_fft, fs, window=window)
        return out

    def _fft_freq(self, fs=None, units="Hz", n_fft=None, coh=False):
        """
        Wrapper to calculate the ordinary or radial frequency vector

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

        if ("Hz" not in units) and ("rad" not in units):
            raise Exception(
                "Valid fft frequency vector units are Hz \
                            or rad/s"
            )

        if "rad" in units:
            return fft_frequency(n_fft, 2 * np.pi * fs)
        else:
            return fft_frequency(n_fft, fs)
