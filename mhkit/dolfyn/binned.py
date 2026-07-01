import numpy as np
import warnings
from scipy import signal
from .tools import slice1d_along_axis, detrend_array
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

    def _outshape_fft(self, inshape, n_fft=None, n_bin=None, step=None):
        """
        Returns `outshape` (the fft 'reshape'd shape) for an `inshape` array.
        """
        n_fft = self._parse_nfft(n_fft)
        n_bin = self._parse_nbin(n_bin)
        if step is None:
            step = n_bin
        n_slices = (inshape[-1] - n_bin) // step + 1
        return list(inshape[:-1]) + [int(n_slices), int(n_fft // 2)]

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

    def reshape(self, arr, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows of shape (..., n_slices, n_bin).

        Parameters
        ----------
        arr : numpy.ndarray
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Window (bin) size. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
          Shape (..., n_slices, n_bin) where
          n_slices = (N - n_bin) // step + 1
        """

        n_bin = int(self._parse_nbin(n_bin))
        if arr.shape[-1] < n_bin:
            raise Exception("n_bin is larger than length of input array")
        if step is None:
            step = n_bin
        step = int(step)
        sliding_window = np.lib.stride_tricks.sliding_window_view(arr, n_bin, axis=-1)
        out = sliding_window[..., ::step, :].copy()
        return out

    def detrend(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows and remove the
        best-fit trend line from each window.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to detrend. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return detrend_array(self.reshape(arr, step=step, n_bin=n_bin), axis=axis)

    def demean(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows and remove the
        mean from each window.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        dt = self.reshape(arr, step=step, n_bin=n_bin)
        return dt - np.nanmean(dt, axis)[..., None]

    def mean(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the mean of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        if np.issubdtype(arr.dtype, np.datetime64):
            return epoch2dt64(
                self.mean(dt642epoch(arr), axis=axis, step=step, n_bin=n_bin)
            )
        if axis != -1:
            arr = np.swapaxes(arr, axis, -1)
        n_bin = self._parse_nbin(n_bin)
        tmp = self.reshape(arr, step=step, n_bin=n_bin)

        return np.nanmean(tmp, -1)

    def variance(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the variance of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take variance. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanvar(
            self.reshape(arr, step=step, n_bin=n_bin), axis=axis, dtype=np.float32
        )

    def standard_deviation(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the standard deviation of each bin along the
        specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take std dev. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanstd(
            self.reshape(arr, step=step, n_bin=n_bin), axis=axis, dtype=np.float32
        )

    def _psd_base(
        self,
        dat,
        fs=None,
        window="hann",
        noise=0,
        n_bin=None,
        n_fft=None,
        pct_overlap=0.5,
    ):
        """
        Calculate the power spectral density of `dat`

        Parameters
        ----------
        dat : xarray.DataArray
          The raw dataArray of which to calculate the psd.
        fs : float (optional)
          The sample rate (Hz).
        window : {None, 1, 'hann', numpy.ndarray}
          The window to use (default: 'hann'). Valid entries are:
            - None,1               : uses a 'boxcar' or ones window.
            - 'hann'               : hanning window.
            - a length(nfft) array : use this as the window directly.
        noise  : float
          The white-noise level of the measurement (in the same units
          as `dat`).
        n_bin : int
          n_bin of veldat2, number of elements per bin if 'None' is taken
          from VelBinner
        n_fft : int
          n_fft of veldat2, number of elements per bin if 'None' is taken
          from VelBinner
        pct_overlap : float
          The percent overlap between FFT windows (default: 0.5)

        Returns
        -------
        out : numpy.ndarray
          The power spectral density of `dat`

        Notes
        -----
        PSD's are calculated based on sample rate units
        """

        fs = self._parse_fs(fs)
        # n_bin determines the number of time bins in the output
        n_bin = self._parse_nbin(n_bin)
        # n_fft determines the length and resolution of the frequency vector
        n_fft = self._parse_nfft(n_fft)
        # step is the advance between consecutive bin slices.
        # For pct_overlap fraction of overlap: step = n_bin * (1 - pct_overlap).
        step = int((1 - pct_overlap) * n_bin)
        out = np.empty(
            self._outshape_fft(dat.shape, n_fft=n_fft, n_bin=n_bin, step=step)
        )
        n_samples = out.shape[-2]
        for i in range(n_samples):
            sample_slice = slice(i * step, i * step + int(n_bin))
            _, psd = signal.welch(
                dat[sample_slice],
                fs=fs,
                window=window,
                nperseg=n_fft,
                noverlap=int(pct_overlap * n_fft),
                detrend="linear",
                return_onesided=True,
                scaling="density",
            )
            # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
            out[i, :] = psd[1:]
        if np.any(noise):
            out -= noise**2 / (fs / 2)
            # Make sure all values of the PSD are >0 (but still small):
            out[out < 0] = np.min(np.abs(out)) / 100
        return out

    def _csd_base(
        self,
        dat1,
        dat2,
        fs=None,
        window="hann",
        n_fft=None,
        n_bin=None,
        pct_overlap=0.5,
    ):
        """
        Compute the cross power spectral density (CPSD) of the signals dat1 and dat2.

        Parameters
        ----------
        dat1 : numpy.ndarray
          The first raw dataArray of which to calculate the cpsd.
        dat2 : numpy.ndarray
          The second raw dataArray of which to calculate the cpsd.
        fs : float (optional)
          The sample rate (Hz).
        window : {None, 1, 'hann', numpy.ndarray}
          The window to use (default: 'hann'). Valid entries are:
            - None,1               : uses a 'boxcar' or ones window.
            - 'hann'               : hanning window.
            - a length(nfft) array : use this as the window directly.
        n_fft : int
          Number of elements in the FFT. If 'None', is taken
          from VelBinner (uses n_fft_coh).
        n_bin : int
          Number of elements per bin. If 'None', is taken
          from VelBinner.
        pct_overlap : float
          The percent overlap between sliding windows (default: 0.5).

        Returns
        -------
        out : numpy.ndarray
          The cross power spectral density of `dat1` and `dat2`

        Notes
        -----
        PSDs are calculated based on sample rate units.
        This removes a linear trend from the signals.
        The two signals must be the same length and both be real.

        This performs:

        .. math::

            fft(a)*conj(fft(b))

        This implementation is consistent with the numpy.correlate
        definition of correlation.  (The conjugate of D.B. Chelton's
        definition of correlation.)

        The units of the spectra is the product of the units of `a` and
        `b`, divided by the units of fs.
        """

        fs = self._parse_fs(fs)
        n_fft = self._parse_nfft_coh(n_fft)
        n_bin = self._parse_nbin(n_bin)

        if dat1.shape != dat2.shape:
            raise ValueError(
                "Cross-spectral density requires equal-length input arrays. "
                "Quasi-synchronized (different sample rate) inputs are not supported."
            )
        if np.iscomplexobj(dat1) or np.iscomplexobj(dat2):
            raise ValueError("Velocity cannot be complex")

        step = int((1 - pct_overlap) * n_bin)
        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin, step=step)
        out = np.empty(oshp, dtype="c{}".format(dat1.dtype.itemsize * 2))
        n_samples = oshp[-2]
        for i in range(n_samples):
            sample_slice = slice(i * step, i * step + int(n_bin))
            _, cpsd = signal.csd(
                dat1[sample_slice],
                dat2[sample_slice],
                fs=fs,
                window=window,
                nperseg=n_fft,
                noverlap=int(pct_overlap * n_fft),
                detrend="linear",
                return_onesided=True,
                scaling="density",
            )
            # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
            out[i, :] = cpsd[1:]
        return out

    def _fft_freq(self, fs=None, units="Hz", n_fft=None, coh=False):
        """
        Calculate the ordinary or radial half-frequency vector for
        a given sample rate and FFT size.

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

        fs = float(self._parse_fs(fs))

        if ("Hz" not in units) and ("rad" not in units):
            raise Exception("Valid fft frequency vector units are Hz \
                            or rad/s")
        if "rad" in units:
            fs = 2 * np.pi * fs

        f = np.fft.fftfreq(int(n_fft), 1 / fs)
        half_freqs = np.abs(f[1 : int(n_fft / 2.0 + 1)])

        return half_freqs
