import numpy as np
from ..velocity import VelBinner
import warnings
from ..tools.misc import slice1d_along_axis, _nans_like
from scipy.special import cbrt
import xarray as xr


class ADVBinner(VelBinner):
    """
    A class that builds upon `VelBinner` for calculating turbulence
    statistics and velocity spectra from ADV data

    Parameters
    ----------
    n_bin : int
      The length of each `bin`, in number of points, for this averaging
      operator.
    fs : int
      Instrument sampling frequency in Hz
    n_fft : int
      The length of the FFT for computing spectra (must be <= n_bin).
      Optional, default `n_fft` = `n_bin`
    n_fft_coh : int
      Number of data points to use for coherence and cross-spectra fft's.
      Optional, default `n_fft_coh` = `n_fft`
        noise : float or array-like
          Instrument noise level in same units as velocity. Typically
          found from `adv.turbulence.doppler_noise_level`.
          Default: None.
    """

    def __call__(self, ds, freq_units="rad/s", window="hann"):
        out = type(ds)()
        out = self.bin_average(ds, out)

        noise = ds.get("doppler_noise", [0, 0, 0])
        out["tke_vec"] = self.turbulent_kinetic_energy(ds["vel"], noise=noise)
        out["stress_vec"] = self.reynolds_stress(ds["vel"])

        out["psd"] = self.power_spectral_density(
            ds["vel"], window=window, freq_units=freq_units, noise=noise
        )
        for key in list(ds.attrs.keys()):
            if "config" in key:
                ds.attrs.pop(key)
        out.attrs = ds.attrs
        out.attrs["n_bin"] = self.n_bin
        out.attrs["n_fft"] = self.n_fft
        out.attrs["n_fft_coh"] = self.n_fft_coh

        return out

    def reynolds_stress(self, veldat, detrend=True):
        """
        Calculate the specific Reynolds stresses
        (cross-covariances of u,v,w in m^2/s^2)

        Parameters
        ----------
        veldat : xr.DataArray
          A velocity data array. The last dimension is assumed
          to be time.
        detrend : bool
          Detrend the velocity data (True), or simply de-mean it
          (False), prior to computing stress. Note: the psd routines
          use detrend, so if you want to have the same amount of
          variance here as there use ``detrend=True``.
          Default = True

        Returns
        -------
        out : xarray.DataArray
        """

        if not isinstance(veldat, xr.DataArray):
            raise TypeError("`veldat` must be an instance of `xarray.DataArray`.")

        time = self.mean(veldat.time.values)
        vel = veldat.values

        out = np.empty(self._outshape(vel[:3].shape)[:-1], dtype=np.float32)

        if detrend:
            vel = self.detrend(vel)
        else:
            vel = self.demean(vel)

        for idx, p in enumerate(self._cross_pairs):
            out[idx] = np.nanmean(vel[p[0]] * vel[p[1]], -1, dtype=np.float64).astype(
                np.float32
            )

        da = xr.DataArray(
            out.astype("float32"),
            dims=veldat.dims,
            attrs={"units": "m2 s-2", "long_name": "Specific Reynolds Stress Vector"},
        )
        da = da.rename({"dir": "tau"})
        da = da.assign_coords({"tau": self.tau, "time": time})

        return da

    def cross_spectral_density(
        self,
        veldat,
        freq_units="rad/s",
        fs=None,
        window="hann",
        n_bin=None,
        n_fft_coh=None,
    ):
        """
        Calculate the cross-spectral density of velocity components.

        Parameters
        ----------
        veldat : xarray.DataArray
          The raw 3D velocity data.
        freq_units : string
          Frequency units of the returned spectra in either Hz or rad/s
          (`f` or :math:`\\omega`)
        fs : float (optional)
          The sample rate. Default = `binner.fs`
        window : string or array
          Specify the window function.
         Options: 1, None, 'hann', 'hamm'
        n_bin : int (optional)
          The bin-size. Default = `binner.n_bin`
        n_fft_coh : int (optional)
          The fft size. Default = `binner.n_fft_coh`

        Returns
        -------
        csd : xarray.DataArray (3, M, N_FFT)
          The first-dimension of the cross-spectrum is the three
          different cross-spectra: 'uv', 'uw', 'vw'.
        """

        if not isinstance(veldat, xr.DataArray):
            raise TypeError("`veldat` must be an instance of `xarray.DataArray`.")
        if ("rad" not in freq_units) and ("Hz" not in freq_units):
            raise ValueError("`freq_units` should be one of 'Hz' or 'rad/s'")

        fs_in = self._parse_fs(fs)
        n_fft = self._parse_nfft_coh(n_fft_coh)
        time = self.mean(veldat.time.values)
        veldat = veldat.values
        if len(np.shape(veldat)) != 2:
            raise Exception(
                "This function is only valid for calculating TKE using "
                "the 3D velocity vector from an ADV."
            )

        out = np.empty(
            self._outshape_fft(veldat[:3].shape, n_fft=n_fft, n_bin=n_bin),
            dtype="complex",
        )

        # Create frequency vector, also checks whether using f or omega
        if "rad" in freq_units:
            fs = 2 * np.pi * fs_in
            freq_units = "rad s-1"
            units = "m2 s-1 rad-1"
        else:
            fs = fs_in
            freq_units = "Hz"
            units = "m2 s-2 Hz-1"
        coh_freq = xr.DataArray(
            self._fft_freq(fs=fs_in, units=freq_units, n_fft=n_fft, coh=True),
            dims=["coh_freq"],
            name="coh_freq",
            attrs={
                "units": freq_units,
                "long_name": "FFT Frequency Vector",
                "coverage_content_type": "coordinate",
            },
        ).astype("float32")

        for ip, ipair in enumerate(self._cross_pairs):
            out[ip] = self._csd_base(
                veldat[ipair[0]],
                veldat[ipair[1]],
                fs=fs,
                window=window,
                n_bin=n_bin,
                n_fft=n_fft,
            )

        csd = xr.DataArray(
            out.astype("complex64"),
            coords={"C": self.C, "time": time, "coh_freq": coh_freq},
            dims=["C", "time", "coh_freq"],
            attrs={
                "units": units,
                "n_fft_coh": n_fft,
                "long_name": "Cross Spectral Density",
            },
        )
        csd["coh_freq"].attrs["units"] = freq_units

        return csd

    def doppler_noise_level(self, psd, pct_fN=0.8):
        """
        Calculate bias due to Doppler noise using the noise floor
        of the velocity spectra.

        Parameters
        ----------
        psd : xarray.DataArray (dir, time, f)
          The ADV power spectral density of velocity (auto-spectra)
        pct_fN : float
          Percent of Nyquist frequency to calculate characeristic frequency

        Returns
        -------
        doppler_noise (xarray.DataArray):
          Doppler noise level in units of m/s

        Notes
        -----
        Approximates bias from

        .. :math: \\sigma^{2}_{noise} = N x f_{c}

        where :math: `\\sigma_{noise}` is the bias due to Doppler noise,
        `N` is the constant variance or spectral density, and `f_{c}`
        is the characteristic frequency.

        The characteristic frequency is then found as

        .. :math: f_{c} = pct_fN * (f_{s}/2)

        where `f_{s}/2` is the Nyquist frequency.


        Richard, Jean-Baptiste, et al. "Method for identification of Doppler noise
        levels in turbulent flow measurements dedicated to tidal energy." International
        Journal of Marine Energy 3 (2013): 52-64.

        Thiébaut, Maxime, et al. "Investigating the flow dynamics and turbulence at a
        tidal-stream energy site in a highly energetic estuary." Renewable Energy 195
        (2022): 252-262.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if not isinstance(pct_fN, float) or not 0 <= pct_fN <= 1:
            raise ValueError("`pct_fN` must be a float within the range [0, 1].")

        # Characteristic frequency set to 80% of Nyquist frequency
        fN = self.fs / 2
        fc = pct_fN * fN

        # Get units right
        if psd.freq.units == "Hz":
            f_range = slice(fc, fN)
        else:
            f_range = slice(2 * np.pi * fc, 2 * np.pi * fN)

        # Noise floor
        N2 = psd.sel(freq=f_range) * psd.freq.sel(freq=f_range)
        noise_level = np.sqrt(N2.mean(dim="freq"))

        return xr.DataArray(
            noise_level.values.astype("float32"),
            coords={"S": psd["S"], "time": psd["time"]},
            attrs={
                "units": "m/s",
                "long_name": "Doppler Noise Level",
                "description": "Doppler noise level calculated " "from PSD white noise",
            },
        )

    def check_turbulence_cascade_slope(self, psd, freq_range=[6.28, 12.57]):
        """
        This function calculates the slope of the PSD, the power spectra
        of velocity, within the given frequency range. The purpose of this
        function is to check that the region of the PSD containing the
        isotropic turbulence cascade decreases at a rate of :math:`f^{-5/3}`.

        Parameters
        ----------
        psd : xarray.DataArray ([time,] freq)
          The power spectral density (1D or 2D)
        freq_range : iterable(2) (default: [6.28, 12.57])
          The range over which the isotropic turbulence cascade occurs, in
          units of the psd frequency vector (Hz or rad/s)

        Returns
        -------
        (m, b): tuple (slope, y-intercept)
          A tuple containing the coefficients of the log-adjusted linear
          regression between PSD and frequency

        Notes
        -----
        Calculates slope based on the `standard` formula for dissipation:

        .. math:: S(k) = \\alpha \\epsilon^{2/3} k^{-5/3} + N

        The slope of the isotropic turbulence cascade, which should be
        equal to :math:`k^{-5/3}` or :math:`f^{-5/3}`, where k and f are
        the wavenumber and frequency vectors, is estimated using linear
        regression with a log transformation:

        .. math:: log10(y) = m*log10(x) + b

        Which is equivalent to

        .. math:: y = 10^{b} x^{m}

        Where :math:`y` is S(k) or S(f), :math:`x` is k or f, :math:`m`
        is the slope (ideally -5/3), and :math:`10^{b}` is the intercept of
        y at x^m=1.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        idx = np.where((freq_range[0] < psd.freq) & (psd.freq < freq_range[1]))
        idx = idx[0]

        x = np.log10(psd["freq"].isel(freq=idx))
        y = np.log10(psd.isel(freq=idx))

        y_bar = y.mean("freq")
        x_bar = x.mean("freq")

        # using the formula to calculate the slope and intercept
        n = np.sum((x - x_bar) * (y - y_bar), axis=0)
        d = np.sum((x - x_bar) ** 2, axis=0)

        m = n / d
        b = y_bar - m * x_bar

        return m, b

    def dissipation_rate_LT83(self, psd, U_mag, freq_range=[6.28, 12.57], noise=None):
        """
        Calculate the dissipation rate from the PSD

        Parameters
        ----------
        psd : xarray.DataArray (...,time,f)
          The power spectral density
        U_mag : xarray.DataArray (...,time)
          The bin-averaged horizontal velocity [m/s] (from dataset shortcut)
        freq_range : iterable(2)
          The range over which to integrate/average the spectrum, in units
          of the psd frequency vector (Hz or rad/s).
          Default = [6.28, 12.57] rad/s
        noise : float or array-like
          Instrument noise level in same units as velocity. Typically
          found from `adv.turbulence.calc_doppler_noise`.
          Default: None.

        Returns
        -------
        epsilon : xarray.DataArray (...,n_time)
          dataArray of the dissipation rate

        Notes
        -----
        This uses the `standard` formula for dissipation:

        .. math:: S(k) = \\alpha \\epsilon^{2/3} k^{-5/3} + N

        where :math:`\\alpha = 0.5` (1.5 for all three velocity
        components), `k` is wavenumber, `S(k)` is the turbulent
        kinetic energy spectrum, and `N' is the doppler noise level
        associated with the TKE spectrum.

        With :math:`k \\rightarrow \\omega / U`, then -- to preserve variance --
        :math:`S(k) = U S(\\omega)`, and so this becomes:

        .. math:: S(\\omega) = \\alpha \\epsilon^{2/3} \\omega^{-5/3} U^{2/3} + N

        With :math:`k \\rightarrow (2\\pi f) / U`, then

        .. math:: S(\\omega) = \\alpha \\epsilon^{2/3} f^{-5/3} (U/(2*\\pi))^{2/3} + N

        LT83 : Lumley and Terray, "Kinematics of turbulence convected
        by a random wave field". JPO, 1983, vol13, pp2000-2007.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if len(U_mag.shape) != 1:
            raise Exception("U_mag should be 1-dimensional (time)")
        if len(psd.time) != len(U_mag.time):
            raise Exception("`U_mag` should be from ensembled-averaged dataset")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        if noise is not None:
            if np.shape(noise)[0] != 3:
                raise Exception("Noise should have same first dimension as velocity")
        else:
            noise = np.array([0, 0, 0])[:, None, None]

        # Noise subtraction from binner.TimeBinner.calc_psd_base
        psd = psd.copy()
        if noise is not None:
            psd -= noise**2 / (self.fs / 2)
            psd = psd.where(psd > 0, np.min(np.abs(psd)) / 100)

        freq = psd.freq
        idx = np.where((freq_range[0] < freq) & (freq < freq_range[1]))
        idx = idx[0]

        if freq.units == "Hz":
            U = U_mag / (2 * np.pi)
        else:
            U = U_mag

        a = 0.5
        out = (psd.isel(freq=idx) * freq.isel(freq=idx) ** (5 / 3) / a).mean(
            axis=-1
        ) ** (3 / 2) / U

        return xr.DataArray(
            out.astype("float32"),
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using "
                "the method from Lumley and Terray, 1983",
            },
        )

    def dissipation_rate_SF(self, vel_raw, U_mag, fs=None, freq_range=[2.0, 4.0]):
        """
        Calculate dissipation rate using the "structure function" (SF) method

        Parameters
        ----------
        vel_raw : xarray.DataArray (time)
          The raw velocity data upon which to perform the SF technique.
        U_mag : xarray.DataArray
          The bin-averaged horizontal velocity (from dataset shortcut)
        fs : float
          The sample rate of `vel_raw` [Hz]
        freq_range : iterable(2)
          The frequency range over which to compute the SF [Hz]
          (i.e. the frequency range within which the isotropic
          turbulence cascade falls).
          Default = [2., 4.] Hz

        Returns
        -------
        epsilon : xarray.DataArray
          dataArray of the dissipation rate
        """

        if not isinstance(vel_raw, xr.DataArray):
            raise TypeError("`vel_raw` must be an instance of `xarray.DataArray`.")
        if len(vel_raw.time) == len(U_mag.time):
            raise Exception("`U_mag` should be from ensembled-averaged dataset")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        veldat = vel_raw.values
        if len(veldat.shape) > 1:
            raise Exception("Function input should be a 1D velocity vector")

        fs = self._parse_fs(fs)
        if freq_range[1] > fs:
            warnings.warn("Max freq_range cannot be greater than fs")

        dt = self.reshape(veldat)
        out = np.empty(dt.shape[:-1], dtype=dt.dtype)
        for slc in slice1d_along_axis(dt.shape, -1):
            up = dt[slc]
            lag = U_mag.values[slc[:-1]] / fs * np.arange(up.shape[0])
            DAA = _nans_like(lag)
            for L in range(int(fs / freq_range[1]), int(fs / freq_range[0])):
                DAA[L] = np.nanmean((up[L:] - up[:-L]) ** 2, dtype=np.float64)
            cv2 = DAA / (lag ** (2 / 3))
            cv2m = np.median(cv2[np.logical_not(np.isnan(cv2))])
            out[slc[:-1]] = (cv2m / 2.1) ** (3 / 2)

        return xr.DataArray(
            out.astype("float32"),
            coords=U_mag.coords,
            dims=U_mag.dims,
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using the "
                '"structure function" method',
            },
        )

    def _up_angle(self, U_complex):
        """
        Calculate the angle of the turbulence fluctuations.

        Parameters
        ----------
        U_complex  : numpy.ndarray (..., n_time * n_bin)
          The complex, raw horizontal velocity (non-binned)

        Returns
        -------
        theta : numpy.ndarray (..., n_time)
          The angle of the turbulence [rad]
        """

        dt = self.demean(U_complex)
        fx = dt.imag <= 0
        dt[fx] = dt[fx] * np.exp(1j * np.pi)

        return np.angle(np.mean(dt, -1, dtype=np.complex128))

    def _integral_TE01(self, I_tke, theta):
        """
        The integral, equation A13, in [TE01].

        Parameters
        ----------
        I_tke : numpy.ndarray
          (beta in TE01) is the turbulence intensity ratio:
          \\sigma_u / V
        theta : numpy.ndarray
          is the angle between the mean flow and the primary axis of
          velocity fluctuations
        """

        x = np.arange(-20, 20, 1e-2)  # I think this is a long enough range.
        out = np.empty_like(I_tke.flatten())
        for i, (b, t) in enumerate(zip(I_tke.flatten(), theta.flatten())):
            out[i] = np.trapz(
                cbrt(x**2 - 2 / b * np.cos(t) * x + b ** (-2)) * np.exp(-0.5 * x**2),
                x,
            )

        return out.reshape(I_tke.shape) * (2 * np.pi) ** (-0.5) * I_tke ** (2 / 3)

    def dissipation_rate_TE01(self, dat_raw, dat_avg, freq_range=[6.28, 12.57]):
        """
        Calculate the dissipation rate according to TE01.

        Parameters
        ----------
        dat_raw : xarray.Dataset
          The raw (off the instrument) adv dataset
        dat_avg : xarray.Dataset
          The bin-averaged adv dataset (calc'd from 'calc_turbulence' or
          'do_avg'). The spectra (psd) and basic turbulence statistics
          ('tke_vec' and 'stress_vec') must already be computed.
        freq_range : iterable(2)
          The range over which to integrate/average the spectrum, in units
          of the psd frequency vector (Hz or rad/s).
          Default = [6.28, 12.57] rad/s

        Notes
        -----
        TE01 : Trowbridge, J and Elgar, S, "Turbulence measurements in
        the Surf Zone". JPO, 2001, vol31, pp2403-2417.
        """

        if not isinstance(dat_raw, xr.Dataset):
            raise TypeError("`dat_raw` must be an instance of `xarray.Dataset`.")
        if not isinstance(dat_avg, xr.Dataset):
            raise TypeError("`dat_avg` must be an instance of `xarray.Dataset`.")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        # Assign local names
        U_mag = dat_avg.velds.U_mag.values
        I_tke = dat_avg.velds.I_tke.values
        theta = np.angle(dat_avg.velds.U.values) - self._up_angle(
            dat_raw.velds.U.values
        )
        freq = dat_avg["psd"].freq.values

        # Calculate constants
        alpha = 1.5
        intgrl = self._integral_TE01(I_tke, theta)

        # Index data to be used
        inds = (freq_range[0] < freq) & (freq < freq_range[1])
        psd = dat_avg.psd[..., inds].values
        freq = freq[inds].reshape([1] * (dat_avg.psd.ndim - 2) + [sum(inds)])

        # Estimate values
        # u & v components (equation 6)
        out = (
            np.nanmean((psd[0] + psd[1]) * freq ** (5 / 3), -1)
            / (21 / 55 * alpha * intgrl)
        ) ** (3 / 2) / U_mag

        # Add w component
        out += (
            np.nanmean(psd[2] * freq ** (5 / 3), -1) / (12 / 55 * alpha * intgrl)
        ) ** (3 / 2) / U_mag

        # Average the two estimates
        out *= 0.5

        return xr.DataArray(
            out.astype("float32"),
            coords={"time": dat_avg.psd.time},
            dims="time",
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using the "
                "method from Trowbridge and Elgar, 2001",
            },
        )

    def integral_length_scales(self, a_cov, U_mag, fs=None):
        """
        Calculate integral length scales.

        Parameters
        ----------
        a_cov : xarray.DataArray
          The auto-covariance array (i.e. computed using `autocovariance`).
        U_mag : xarray.DataArray
          The bin-averaged horizontal velocity (from dataset shortcut)
        fs : numeric
          The raw sample rate

        Returns
        -------
        L_int : numpy.ndarray (..., n_time)
          The integral length scale (T_int*U_mag).

        Notes
        ----
        The integral time scale (T_int) is the lag-time at which the
        auto-covariance falls to 1/e.

        If T_int is not reached, L_int will default to '0'.
        """

        if not isinstance(a_cov, xr.DataArray):
            raise TypeError("`a_cov` must be an instance of `xarray.DataArray`.")
        if len(a_cov.time) != len(U_mag.time):
            raise Exception("`U_mag` should be from ensembled-averaged dataset")

        acov = a_cov.values
        fs = self._parse_fs(fs)

        scale = np.argmin((acov / acov[..., :1]) > (1 / np.e), axis=-1)
        L_int = U_mag.values / fs * scale

        return xr.DataArray(
            L_int.astype("float32"),
            coords={"dir": a_cov.dir, "time": a_cov.time},
            attrs={
                "units": "m",
                "long_name": "Integral Length Scale",
                "standard_name": "turbulent_mixing_length_of_sea_water",
            },
        )


def turbulence_statistics(
    ds_raw, n_bin, fs, n_fft=None, freq_units="rad/s", window="hann"
):
    """
    Functional version of `ADVBinner` that computes a suite of turbulence
    statistics for the input dataset, and returns a `binned` data object.

    Parameters
    ----------
    ds_raw : xarray.Dataset
      The raw adv datset to `bin`, average and compute
      turbulence statistics of.
    freq_units : string
      Frequency units of the returned spectra in either Hz or rad/s
      (`f` or :math:`\\omega`). Default is 'rad/s'
    window : string or array
      The window to use for calculating spectra.


    Returns
    -------
    ds : xarray.Dataset
      Returns an 'binned' (i.e. 'averaged') data object. All
      fields (variables) of the input data object are averaged in n_bin
      chunks. This object also computes the following items over
      those chunks:

      - tke_vec : The energy in each component, each components is
        alternatively accessible as:
        :attr:`upup_ <dolfyn.velocity.Velocity.upup_>`,
        :attr:`vpvp_ <dolfyn.velocity.Velocity.vpvp_>`,
        :attr:`wpwp_ <dolfyn.velocity.Velocity.wpwp_>`)

      - stress_vec : The Reynolds stresses, each component is
        alternatively accessible as:
        :attr:`upwp_ <dolfyn.data.velocity.Velocity.upwp_>`,
        :attr:`vpwp_ <dolfyn.data.velocity.Velocity.vpwp_>`,
        :attr:`upvp_ <dolfyn.data.velocity.Velocity.upvp_>`)

      - U_std : The standard deviation of the horizontal
        velocity `U_mag`.

      - psd : DataArray containing the spectra of the velocity
        in radial frequency units. The data-array contains:
        - vel : the velocity spectra array (m^2/s/rad))
        - omega : the radial frequncy (rad/s)
    """

    calculator = ADVBinner(n_bin, fs, n_fft=n_fft)

    return calculator(ds_raw, freq_units=freq_units, window=window)
