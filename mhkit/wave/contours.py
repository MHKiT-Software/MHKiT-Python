from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.stats as stats
import scipy.interpolate as interp
import numpy as np
import warnings
from mhkit.utils import to_numeric_array

import matplotlib

mpl_version = tuple(map(int, matplotlib.__version__.split(".")))


# Contours
def environmental_contours(x1, x2, sea_state_duration, return_period, method, **kwargs):
    """
    Returns a Dictionary of x1 and x2 components for each contour
    method passed. A method  may be one of the following:
    Principal Component Analysis, Gaussian, Gumbel, Clayton, Rosenblatt,
    nonparametric Gaussian, nonparametric Clayton,
    nonparametric Gumbel, bivariate KDE, log bivariate KDE

    Parameters
    ----------
    x1: list, np.ndarray, pd.Series, xr.DataArray
        Component 1 data
    x2: list, np.ndarray, pd.Series, xr.DataArray
        Component 2 data
    sea_state_duration : int or float
        `x1` and `x2` averaging period in seconds
    return_period: int, float
        Return period of interest in years
    method: string or list
        Copula method to apply. Options include ['PCA','gaussian',
        'gumbel', 'clayton', 'rosenblatt', 'nonparametric_gaussian',
        'nonparametric_clayton', 'nonparametric_gumbel', 'bivariate_KDE'
        'bivariate_KDE_log']

    **kwargs
        min_bin_count: int
            Passed to _copula_parameters to sets the minimum number of
            bins allowed. Default = 40.
        initial_bin_max_val: int, float
            Passed to _copula_parameters to set the max value of the
            first bin. Default = 1.
        bin_val_size: int, float
            Passed to _copula_parameters to set the size of each bin
            after the initial bin.  Default 0.25.
        nb_steps: int
            Discretization of the circle in the normal space is used for
            copula component calculation. Default nb_steps=1000.
        bandwidth:
            Must specify bandwidth for bivariate KDE method.
            Default = None.
        Ndata_bivariate_KDE: int
            Must specify bivariate KDE method. Defines the contoured
            space from which samples are taken. Default = 100.
        max_x1: float
            Defines the max value of x1 to discretize the KDE space
        max_x2: float
            Defines the max value of x2 to discretize the KDE space
        PCA: dict
            If provided, the principal component analysis (PCA) on x1,
            x2 is skipped. The PCA will be the same for a given x1, x2
            therefore this step may be skipped if multiple calls to
            environmental contours are made for the same x1, x2 pair.
            The PCA dict may be obtained by setting return_fit=True when
            calling the PCA method.
        return_fit: boolean
            Will return fitting parameters used for each method passed.
            Default False.

    Returns
    -------
    copulas: Dictionary
        Dictionary of x1 and x2 copula components for each copula method
    """
    x1 = to_numeric_array(x1, "x1")
    x2 = to_numeric_array(x2, "x2")
    if not isinstance(x1, np.ndarray) or x1.ndim == 0:
        raise TypeError(f"x1 must be a non-scalar array. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray) or x2.ndim == 0:
        raise TypeError(f"x2 must be a non-scalar array. Got: {type(x2)}")
    if len(x1) != len(x2):
        raise ValueError("The lengths of x1 and x2 must be equal.")
    if not isinstance(sea_state_duration, (int, float)):
        raise TypeError(
            f"sea_state_duration must be of type int or float. Got: {type(sea_state_duration)}"
        )
    if not isinstance(return_period, (int, float, np.ndarray)):
        raise TypeError(
            f"return_period must be of type int, float, or np.ndarray. Got: {type(return_period)}"
        )

    bin_val_size = kwargs.get("bin_val_size", 0.25)
    nb_steps = kwargs.get("nb_steps", 1000)
    initial_bin_max_val = kwargs.get("initial_bin_max_val", 1.0)
    min_bin_count = kwargs.get("min_bin_count", 40)
    bandwidth = kwargs.get("bandwidth", None)
    Ndata_bivariate_KDE = kwargs.get("Ndata_bivariate_KDE", 100)
    max_x1 = kwargs.get("max_x1", None)
    max_x2 = kwargs.get("max_x2", None)
    PCA = kwargs.get("PCA", None)
    PCA_bin_size = kwargs.get("PCA_bin_size", 250)
    return_fit = kwargs.get("return_fit", False)

    if not isinstance(max_x1, (int, float, type(None))):
        raise TypeError(f"If specified, max_x1 must be a dict. Got: {type(PCA)}")
    if not isinstance(max_x2, (int, float, type(None))):
        raise TypeError(f"If specified, max_x2 must be a dict. Got: {type(PCA)}")
    if not isinstance(PCA, (dict, type(None))):
        raise TypeError(f"If specified, PCA must be a dict. Got: {type(PCA)}")
    if not isinstance(PCA_bin_size, int):
        raise TypeError(f"PCA_bin_size must be of type int. Got: {type(PCA_bin_size)}")
    if not isinstance(return_fit, bool):
        raise TypeError(f"return_fit must be of type bool. Got: {type(return_fit)}")
    if not isinstance(bin_val_size, (int, float)):
        raise TypeError(
            f"bin_val_size must be of type int or float. Got: {type(bin_val_size)}"
        )
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")
    if not isinstance(min_bin_count, int):
        raise TypeError(
            f"min_bin_count must be of type int. Got: {type(min_bin_count)}"
        )
    if not isinstance(initial_bin_max_val, (int, float)):
        raise TypeError(
            f"initial_bin_max_val must be of type int or float. Got: {type(initial_bin_max_val)}"
        )
    if "bivariate_KDE" in method and bandwidth == None:
        raise TypeError(
            f"Must specify keyword bandwidth with bivariate KDE method. Got: {type(bandwidth)}"
        )

    if isinstance(method, str):
        method = [method]
    if not (len(set(method)) == len(method)):
        raise ValueError(
            f"Can only pass a unique "
            + "method once per function call. Consider wrapping this "
            + "function in a for loop to investage variations on the same method"
        )

    method_class = {
        "PCA": "parametric",
        "gaussian": "parametric",
        "gumbel": "parametric",
        "clayton": "parametric",
        "rosenblatt": "parametric",
        "nonparametric_gaussian": "nonparametric",
        "nonparametric_clayton": "nonparametric",
        "nonparametric_gumbel": "nonparametric",
        "bivariate_KDE": "KDE",
        "bivariate_KDE_log": "KDE",
    }

    classification = []
    methods = method
    for method in methods:
        classification.append(method_class[method])

    fit = _iso_prob_and_quantile(sea_state_duration, return_period, nb_steps)
    fit_parametric = None
    fit_nonparametric = None
    component_1 = None
    if "parametric" in classification:
        (para_dist_1, para_dist_2, mean_cond, std_cond) = _copula_parameters(
            x1, x2, min_bin_count, initial_bin_max_val, bin_val_size
        )

        x_quantile = fit["x_quantile"]
        a = para_dist_1[0]
        c = para_dist_1[1]
        loc = para_dist_1[2]
        scale = para_dist_1[3]

        component_1 = stats.exponweib.ppf(x_quantile, a, c, loc=loc, scale=scale)

        fit_parametric = fit
        fit_parametric["para_dist_1"] = para_dist_1
        fit_parametric["para_dist_2"] = para_dist_2
        fit_parametric["mean_cond"] = mean_cond
        fit_parametric["std_cond"] = std_cond
        if PCA == None:
            PCA = fit_parametric

    if "nonparametric" in classification:
        (
            nonpara_dist_1,
            nonpara_dist_2,
            nonpara_pdf_2,
        ) = _nonparametric_copula_parameters(x1, x2, nb_steps=nb_steps)
        fit_nonparametric = fit
        fit_nonparametric["nonpara_dist_1"] = nonpara_dist_1
        fit_nonparametric["nonpara_dist_2"] = nonpara_dist_2
        fit_nonparametric["nonpara_pdf_2"] = nonpara_pdf_2

    copula_functions = {
        "PCA": {
            "func": PCA_contour,
            "vals": (
                x1,
                x2,
                PCA,
                {
                    "nb_steps": nb_steps,
                    "return_fit": return_fit,
                    "bin_size": PCA_bin_size,
                },
            ),
        },
        "gaussian": {
            "func": _gaussian_copula,
            "vals": (x1, x2, fit_parametric, component_1, {"return_fit": return_fit}),
        },
        "gumbel": {
            "func": _gumbel_copula,
            "vals": (
                x1,
                x2,
                fit_parametric,
                component_1,
                nb_steps,
                {"return_fit": return_fit},
            ),
        },
        "clayton": {
            "func": _clayton_copula,
            "vals": (x1, x2, fit_parametric, component_1, {"return_fit": return_fit}),
        },
        "rosenblatt": {
            "func": _rosenblatt_copula,
            "vals": (x1, x2, fit_parametric, component_1, {"return_fit": return_fit}),
        },
        "nonparametric_gaussian": {
            "func": _nonparametric_gaussian_copula,
            "vals": (x1, x2, fit_nonparametric, nb_steps, {"return_fit": return_fit}),
        },
        "nonparametric_clayton": {
            "func": _nonparametric_clayton_copula,
            "vals": (x1, x2, fit_nonparametric, nb_steps, {"return_fit": return_fit}),
        },
        "nonparametric_gumbel": {
            "func": _nonparametric_gumbel_copula,
            "vals": (x1, x2, fit_nonparametric, nb_steps, {"return_fit": return_fit}),
        },
        "bivariate_KDE": {
            "func": _bivariate_KDE,
            "vals": (
                x1,
                x2,
                bandwidth,
                fit,
                nb_steps,
                Ndata_bivariate_KDE,
                {"max_x1": max_x1, "max_x2": max_x2, "return_fit": return_fit},
            ),
        },
        "bivariate_KDE_log": {
            "func": _bivariate_KDE,
            "vals": (
                x1,
                x2,
                bandwidth,
                fit,
                nb_steps,
                Ndata_bivariate_KDE,
                {
                    "max_x1": max_x1,
                    "max_x2": max_x2,
                    "log_transform": True,
                    "return_fit": return_fit,
                },
            ),
        },
    }
    copulas = {}

    for method in methods:
        vals = copula_functions[method]["vals"]
        if return_fit:
            component_1, component_2, fit = copula_functions[method]["func"](*vals)
            copulas[f"{method}_fit"] = fit
        else:
            component_1, component_2 = copula_functions[method]["func"](*vals)
        copulas[f"{method}_x1"] = component_1
        copulas[f"{method}_x2"] = component_2

    return copulas


def PCA_contour(x1, x2, fit, kwargs):
    """
    Calculates environmental contours of extreme sea
    states using the improved joint probability distributions
    with the inverse first-order reliability method (I-FORM)
    probability for the desired return period (`return_period`). Given
    the return_period of interest, a circle of iso-probability is
    created in the principal component analysis (PCA) joint probability
    (`x1`, `x2`) reference frame.
    Using the joint probability value, the cumulative distribution
    function (CDF) of the marginal distribution is used to find
    the quantile of each component.
    Finally, using the improved PCA methodology,
    the component 2 contour lines are calculated from component 1 using
    the relationships defined in Eckert-Gallup et. al. 2016.

    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., &
    Neary, V. S. (2016). Application of principal component
    analysis (PCA) and improved joint probability distributions to
    the inverse first-order reliability method (I-FORM) for predicting
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ----------
    x1: list, np.ndarray, pd.Series, xr.DataArray
        Component 1 data
    x2: list, np.ndarray, pd.Series, xr.DataArray
        Component 2 data
    fit: dict
        Dictionary of the iso-probability results. May additionally
        contain the principal component analysis (PCA) on x1, x2
        The PCA will be the same for a given x1, x2
        therefore this step may be skipped if multiple calls to
        environmental contours are made for the same x1, x2 pair.
        The PCA dict may be obtained by setting return_fit=True when
        calling the PCA method.
    kwargs : optional
        bin_size : int
            Data points in each bin for the PCA fit. Default bin_size=250.
        nb_steps : int
            Discretization of the circle in the normal space used for
            I-FORM calculation. Default nb_steps=1000.
        return_fit: boolean
            Default False, if True will return the PCA fit dictionary

    Returns
    -------
    x1_contour : numpy array
        Calculated x1 values along the contour boundary following
        return to original input orientation.
    x2_contour : numpy array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: dict (optional)
            principal component analysis dictionary
        Keys:
        -----
        'principal_axes': sign corrected PCA axes
        'shift'         : The shift applied to x2
        'x1_fit'        : gaussian fit of x1 data
        'mu_param'      : fit to _mu_fcn
        'sigma_param'   : fit to _sig_fits

    """
    x1 = to_numeric_array(x1, "x1")
    x2 = to_numeric_array(x2, "x2")
    if not isinstance(x1, np.ndarray) or x1.ndim == 0:
        raise TypeError(f"x1 must be a non-scalar array. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray) or x2.ndim == 0:
        raise TypeError(f"x2 must be a non-scalar array. Got: {type(x2)}")
    if len(x1) != len(x2):
        raise ValueError("The lengths of x1 and x2 must be equal.")

    bin_size = kwargs.get("bin_size", 250)
    nb_steps = kwargs.get("nb_steps", 1000)
    return_fit = kwargs.get("return_fit", False)

    if not isinstance(bin_size, int):
        raise TypeError(f"bin_size must be of type int. Got: {type(bin_size)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")
    if not isinstance(return_fit, bool):
        raise TypeError(f"return_fit must be of type bool. Got: {type(return_fit)}")

    if "x1_fit" not in fit:
        pca_fit = _principal_component_analysis(x1, x2, bin_size=bin_size)
        for key in pca_fit:
            fit[key] = pca_fit[key]

    x_quantile = fit["x_quantile"]
    y_quantile = fit["y_quantile"]

    # Use the inverse of cdf to calculate component 1 values
    component_1 = stats.invgauss.ppf(
        x_quantile,
        mu=fit["x1_fit"]["mu"],
        loc=fit["x1_fit"]["loc"],
        scale=fit["x1_fit"]["scale"],
    )

    # Find Component 2 mu using first order linear regression
    mu_slope = fit["mu_fit"].slope
    mu_intercept = fit["mu_fit"].intercept
    component_2_mu = mu_slope * component_1 + mu_intercept

    # Find Componenet 2 sigma using second order polynomial fit
    sigma_polynomial_coeffcients = fit["sigma_fit"].x
    component_2_sigma = np.polyval(sigma_polynomial_coeffcients, component_1)

    # Use calculated mu and sigma values to calculate C2 along the contour
    component_2 = stats.norm.ppf(
        y_quantile, loc=component_2_mu, scale=component_2_sigma
    )

    # Convert contours back to the original reference frame
    principal_axes = fit["principal_axes"]
    shift = fit["shift"]
    pa00 = principal_axes[0, 0]
    pa01 = principal_axes[0, 1]

    x1_contour = (pa00 * component_1 + pa01 * (component_2 - shift)) / (
        pa01**2 + pa00**2
    )
    x2_contour = (pa01 * component_1 - pa00 * (component_2 - shift)) / (
        pa01**2 + pa00**2
    )

    # Assign 0 value to any negative x1 contour values
    x1_contour = np.maximum(0, x1_contour)

    if return_fit:
        return np.transpose(x1_contour), np.transpose(x2_contour), fit
    return np.transpose(x1_contour), np.transpose(x2_contour)


def _principal_component_analysis(x1, x2, bin_size=250):
    """
    Performs a modified principal component analysis (PCA)
    [Eckert et. al 2016] on two variables (`x1`, `x2`). The additional
    PCA is performed in 5 steps:
    1) Transform `x1` & `x2` into the principal component domain and
       shift the y-axis so that all values are positive and non-zero
    2) Fit the `x1` data in the transformed reference frame with an
       inverse Gaussian Distribution
    3) Bin the transformed data into groups of size bin and find the
       mean of `x1`, the mean of `x2`, and the standard deviation of
       `x2`
    4) Perform a first-order linear regression to determine a continuous
       the function relating the mean of the `x1` bins to mean of the
       `x2` bins
    5) Find a second-order polynomial which best relates the means of
       `x1` to the standard deviation of `x2` using constrained
       optimization

    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., &
    Neary, V. S. (2016). Application of principal component
    analysis (PCA) and improved joint probability distributions to
    the inverse first-order reliability method (I-FORM) for predicting
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ----------
    x1: numpy array
        Component 1 data
    x2: numpy array
        Component 2 data
    bin_size : int
        Number of data points in each bin

    Returns
    -------
    PCA: dict
       Keys:
       -----
       'principal_axes': sign corrected PCA axes
       'shift'         : The shift applied to x2
       'x1_fit'        : gaussian fit of x1 data
       'mu_param'      : fit to _mu_fcn
       'sigma_param'   : fit to _sig_fits
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(bin_size, int):
        raise TypeError(f"bin_size must be of type int. Got: {type(bin_size)}")

    # Step 0: Perform Standard PCA
    mean_location = 0
    x1_mean_centered = x1 - x1.mean(axis=0)
    x2_mean_centered = x2 - x2.mean(axis=0)
    n_samples_by_n_features = np.column_stack((x1_mean_centered, x2_mean_centered))
    pca = skPCA(n_components=2)
    pca.fit(n_samples_by_n_features)
    principal_axes = pca.components_

    # STEP 1: Transform data into new reference frame
    # Apply correct/expected sign convention
    principal_axes = abs(principal_axes)
    principal_axes[1, 1] = -principal_axes[1, 1]

    # Rotate data into Principal direction
    x1_and_x2 = np.column_stack((x1, x2))
    x1_x2_components = np.dot(x1_and_x2, principal_axes)
    x1_components = x1_x2_components[:, 0]
    x2_components = x1_x2_components[:, 1]

    # Apply shift to Component 2 to make all values positive
    shift = abs(min(x2_components)) + 0.1
    x2_components = x2_components + shift

    # STEP 2: Fit Component 1 data using a Gaussian Distribution
    x1_sorted_index = x1_components.argsort()
    x1_sorted = x1_components[x1_sorted_index]
    x2_sorted = x2_components[x1_sorted_index]

    x1_fit_results = stats.invgauss.fit(x1_sorted, floc=mean_location)
    x1_fit = {
        "mu": x1_fit_results[0],
        "loc": x1_fit_results[1],
        "scale": x1_fit_results[2],
    }

    # Step 3: Bin Data & find order 1 linear relation between x1 & x2 means
    N = len(x1)
    minimum_4_bins = np.floor(N * 0.25)
    if bin_size > minimum_4_bins:
        bin_size = minimum_4_bins
        msg = (
            "To allow for a minimum of 4 bins, the bin size has been "
            + f"set to {minimum_4_bins}"
        )
        warnings.warn(msg, UserWarning)

    N_multiples = int(N // bin_size)
    max_N_multiples_index = int(N_multiples * bin_size)

    x1_integer_multiples_of_bin_size = x1_sorted[0:max_N_multiples_index]
    x2_integer_multiples_of_bin_size = x2_sorted[0:max_N_multiples_index]

    x1_bins = np.split(x1_integer_multiples_of_bin_size, N_multiples)
    x2_bins = np.split(x2_integer_multiples_of_bin_size, N_multiples)

    x1_last_bin = x1_sorted[max_N_multiples_index:]
    x2_last_bin = x2_sorted[max_N_multiples_index:]

    x1_bins.append(x1_last_bin)
    x2_bins.append(x2_last_bin)

    x1_means = np.array([])
    x2_means = np.array([])
    x2_sigmas = np.array([])

    for x1_bin, x2_bin in zip(x1_bins, x2_bins):
        x1_means = np.append(x1_means, x1_bin.mean())
        x2_means = np.append(x2_means, x2_bin.mean())
        x2_sigmas = np.append(x2_sigmas, x2_bin.std())

    mu_fit = stats.linregress(x1_means, x2_means)

    # STEP 4: Find order 2 relation between x1_mean and x2 standard deviation
    sigma_polynomial_order = 2
    sig_0 = 0.1 * np.ones(sigma_polynomial_order + 1)

    def _objective_function(sig_p, x1_means, x2_sigmas):
        return mean_squared_error(np.polyval(sig_p, x1_means), x2_sigmas)

    # Constraint Functions
    def y_intercept_gt_0(sig_p):
        return sig_p[2]

    def sig_polynomial_min_gt_0(sig_p):
        return sig_p[2] - (sig_p[1] ** 2) / (4 * sig_p[0])

    constraints = (
        {"type": "ineq", "fun": y_intercept_gt_0},
        {"type": "ineq", "fun": sig_polynomial_min_gt_0},
    )

    sigma_fit = optim.minimize(
        _objective_function,
        x0=sig_0,
        args=(x1_means, x2_sigmas),
        method="SLSQP",
        constraints=constraints,
    )

    PCA = {
        "principal_axes": principal_axes,
        "shift": shift,
        "x1_fit": x1_fit,
        "mu_fit": mu_fit,
        "sigma_fit": sigma_fit,
    }

    return PCA


def _iso_prob_and_quantile(sea_state_duration, return_period, nb_steps):
    """
    Calculates the iso-probability and the x, y quantiles along
    the iso-probability radius

    Parameters
    ----------
    sea_state_duration : int or float
        `x1` and `x2` sample rate (seconds)
    return_period: int, float
        Return period of interest in years
    nb_steps: int
        Discretization of the circle in the normal space.
        Default nb_steps=1000.

    Returns
    -------
    results: Dictionay
        Dictionary of the iso-probability results
        Keys:
        'exceedance_probability' - probability of exceedance
        'x_component_iso_prob' - x-component of iso probability circle
        'y_component_iso_prob' - y-component of iso probability circle
        'x_quantile' - CDF of x-component
        'y_quantile' - CDF of y-component
    """

    if not isinstance(sea_state_duration, (int, float)):
        raise TypeError(
            f"sea_state_duration must be of type int or float. Got: {type(sea_state_duration)}"
        )
    if not isinstance(return_period, (int, float)):
        raise TypeError(
            f"return_period must be of type int or float. Got: {type(return_period)}"
        )
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")

    dt_yrs = sea_state_duration / (3600 * 24 * 365)
    exceedance_probability = 1 / (return_period / dt_yrs)
    iso_probability_radius = stats.norm.ppf(
        (1 - exceedance_probability), loc=0, scale=1
    )
    discretized_radians = np.linspace(0, 2 * np.pi, nb_steps)

    x_component_iso_prob = iso_probability_radius * np.cos(discretized_radians)
    y_component_iso_prob = iso_probability_radius * np.sin(discretized_radians)

    x_quantile = stats.norm.cdf(x_component_iso_prob, loc=0, scale=1)
    y_quantile = stats.norm.cdf(y_component_iso_prob, loc=0, scale=1)

    results = {
        "exceedance_probability": exceedance_probability,
        "x_component_iso_prob": x_component_iso_prob,
        "y_component_iso_prob": y_component_iso_prob,
        "x_quantile": x_quantile,
        "y_quantile": y_quantile,
    }
    return results


def _copula_parameters(x1, x2, min_bin_count, initial_bin_max_val, bin_val_size):
    """
    Returns an estimate of the Weibull and Lognormal distribution for
    x1 and x2 respectively. Additionally returns the estimates of the
    coefficients from the mean and standard deviation of the Log of x2
    given x1.

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    min_bin_count: int
        Sets the minimum number of bins allowed
    initial_bin_max_val: int, float
        Sets the max value of the first bin
    bin_val_size: int, float
        The size of each bin after the initial bin

    Returns
    -------
    para_dist_1: array
        Weibull distribution parameters for  for component 1
    para_dist_2: array
        Lognormal distribution parameters for component 2
    mean_cond: array
        Estimate coefficients of mean of Ln(x2|x1)
    std_cond: array
        Estimate coefficients of the standard deviation of Ln(x2|x1)
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(min_bin_count, int):
        raise TypeError(
            f"min_bin_count must be of type int. Got: {type(min_bin_count)}"
        )
    if not isinstance(bin_val_size, (int, float)):
        raise TypeError(
            f"bin_val_size must be of type int or float. Got: {type(bin_val_size)}"
        )
    if not isinstance(initial_bin_max_val, (int, float)):
        raise TypeError(
            f"initial_bin_max_val must be of type int or float. Got: {type(initial_bin_max_val)}"
        )

    # Binning
    x1_sorted_index = x1.argsort()
    x1_sorted = x1[x1_sorted_index]
    x2_sorted = x2[x1_sorted_index]

    # Because x1 is sorted we can find the max index as follows:
    ind = np.array([])
    N_vals_lt_limit = sum(x1_sorted <= initial_bin_max_val)
    ind = np.append(ind, N_vals_lt_limit)

    # Make sure first bin isn't empty or too small to avoid errors
    while ind == 0 or ind < min_bin_count:
        ind = np.array([])
        initial_bin_max_val += bin_val_size
        N_vals_lt_limit = sum(x1_sorted <= initial_bin_max_val)
        ind = np.append(ind, N_vals_lt_limit)

    # Add bins until the total number of vals in between bins is
    # < the min bin size
    i = 0
    bin_size_i = np.inf
    while bin_size_i >= min_bin_count:
        i += 1
        bin_i_max_val = initial_bin_max_val + bin_val_size * (i)
        N_vals_lt_limit = sum(x1_sorted <= bin_i_max_val)
        ind = np.append(ind, N_vals_lt_limit)
        bin_size_i = ind[i] - ind[i - 1]

    # Weibull distribution parameters for component 1 using MLE
    para_dist_1 = stats.exponweib.fit(x1_sorted, floc=0, fa=1)
    # Lognormal distribution parameters for component 2 using MLE
    para_dist_2 = stats.norm.fit(np.log(x2_sorted))

    # Parameters for conditional distribution of T|Hs for each bin
    num = len(ind)  # num+1: number of bins
    para_dist_cond = []
    hss = []

    # Bin zero special case (lognormal dist over only 1 bin)
    # parameters for zero bin
    ind0 = range(0, int(ind[0]))
    x2_log0 = np.log(x2_sorted[ind0])
    x2_lognormal_dist0 = stats.norm.fit(x2_log0)
    para_dist_cond.append(x2_lognormal_dist0)
    # mean of x1 (component 1 for zero bin)
    x1_bin0 = x1_sorted[range(0, int(ind[0]) - 1)]
    hss.append(np.mean(x1_bin0))

    # Special case 2-bin lognormal Dist
    # parameters for 1 bin
    ind1 = range(0, int(ind[1]))
    x2_log1 = np.log(x2_sorted[ind1])
    x2_lognormal_dist1 = stats.norm.fit(x2_log1)
    para_dist_cond.append(x2_lognormal_dist1)

    # mean of Hs (component 1 for bin 1)
    hss.append(np.mean(x1_sorted[range(0, int(ind[1]) - 1)]))

    # lognormal Dist (lognormal dist over only 2 bins)
    for i in range(2, num):
        ind_i = range(int(ind[i - 2]), int(ind[i]))
        x2_log_i = np.log(x2_sorted[ind_i])
        x2_lognormal_dist_i = stats.norm.fit(x2_log_i)
        para_dist_cond.append(x2_lognormal_dist_i)

        hss.append(np.mean(x1_sorted[ind_i]))

    # Estimate coefficient using least square solution (mean: 3rd order,
    # sigma: 2nd order)
    ind_f = range(int(ind[num - 2]), int(len(x1)))
    x2_log_f = np.log(x2_sorted[ind_f])
    x2_lognormal_dist_f = stats.norm.fit(x2_log_f)
    para_dist_cond.append(x2_lognormal_dist_f)  # parameters for last bin

    # mean of Hs (component 1 for last bin)
    hss.append(np.mean(x1_sorted[ind_f]))

    para_dist_cond = np.array(para_dist_cond)
    hss = np.array(hss)

    # cubic in Hs: a + bx + cx**2 + dx**3
    phi_mean = np.column_stack((np.ones(num + 1), hss, hss**2, hss**3))
    # quadratic in Hs  a + bx + cx**2
    phi_std = np.column_stack((np.ones(num + 1), hss, hss**2))

    # Estimate coefficients of mean of Ln(T|Hs)(vector 4x1) (cubic in Hs)
    mean_cond = np.linalg.lstsq(phi_mean, para_dist_cond[:, 0], rcond=None)[0]
    # Estimate coefficients of standard deviation of Ln(T|Hs)
    #    (vector 3x1) (quadratic in Hs)
    std_cond = np.linalg.lstsq(phi_std, para_dist_cond[:, 1], rcond=None)[0]

    return para_dist_1, para_dist_2, mean_cond, std_cond


def _gaussian_copula(x1, x2, fit, component_1, kwargs):
    """
    Extreme Sea State Gaussian Copula Contour function.
    This function calculates environmental contours of extreme sea
    states using a Gaussian copula and the inverse first-order
    reliability method.

    Parameters
    ----------
    x1: numpy array
        Component 1 data
    x2: numpy array
        Component 2 data
    fit: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    component_2_Gaussian
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    try:
        x1 = np.array(x1)
    except:
        pass
    try:
        x2 = np.array(x2)
    except:
        pass
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(component_1, np.ndarray):
        raise TypeError(
            f"component_1 must be of type np.ndarray. Got: {type(component_1)}"
        )
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    x_component_iso_prob = fit["x_component_iso_prob"]
    y_component_iso_prob = fit["y_component_iso_prob"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    rho_gau = np.sin(tau * np.pi / 2.0)

    z2_Gauss = stats.norm.cdf(
        y_component_iso_prob * np.sqrt(1.0 - rho_gau**2.0)
        + rho_gau * x_component_iso_prob
    )

    para_dist_2 = fit["para_dist_2"]
    s = para_dist_2[1]
    loc = 0
    scale = np.exp(para_dist_2[0])

    # lognormal inverse
    component_2_Gaussian = stats.lognorm.ppf(z2_Gauss, s=s, loc=loc, scale=scale)
    fit["tau"] = tau
    fit["rho"] = rho_gau
    fit["z2"] = z2_Gauss

    if return_fit:
        return component_1, component_2_Gaussian, fit
    return component_1, component_2_Gaussian


def _gumbel_density(u, alpha):
    """
    Calculates the Gumbel copula density.

    Parameters
    ----------
    u: np.array
        Vector of equally spaced points between 0 and twice the
            maximum value of T.
    alpha: float
        Copula parameter. Must be greater than or equal to 1.

    Returns
    -------
    y: np.array
        Copula density function.
    """

    # Ignore divide by 0 warnings and resulting NaN warnings
    np.seterr(all="ignore")
    v = -np.log(u)
    v = np.sort(v, axis=0)
    vmin = v[0, :]
    vmax = v[1, :]
    nlogC = vmax * (1 + (vmin / vmax) ** alpha) ** (1 / alpha)
    y = (alpha - 1 + nlogC) * np.exp(
        -nlogC
        + np.sum((alpha - 1) * np.log(v) + v, axis=0)
        + (1 - 2 * alpha) * np.log(nlogC)
    )
    np.seterr(all="warn")
    return y


def _gumbel_copula(x1, x2, fit, component_1, nb_steps, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Gumbel copula and the inverse first-order reliability
    method.

    Parameters
    ----------
    x1: numpy array
        Component 1 data
    x2: numpy array
        Component 2 data
    fit: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    component_2_Gumbel: array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    try:
        x1 = np.array(x1)
    except:
        pass
    try:
        x2 = np.array(x2)
    except:
        pass
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(component_1, np.ndarray):
        raise TypeError(
            f"component_1 must be of type np.ndarray. Got: {type(component_1)}"
        )
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    x_quantile = fit["x_quantile"]
    y_quantile = fit["y_quantile"]
    para_dist_2 = fit["para_dist_2"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    theta_gum = 1.0 / (1.0 - tau)

    min_limit_2 = 0
    max_limit_2 = np.ceil(np.amax(x2) * 2)
    Ndata = 1000

    x = np.linspace(min_limit_2, max_limit_2, Ndata)

    s = para_dist_2[1]
    scale = np.exp(para_dist_2[0])
    z2 = stats.lognorm.cdf(x, s=s, loc=0, scale=scale)

    fit["tau"] = tau
    fit["theta"] = theta_gum
    fit["z2"] = z2

    component_2_Gumbel = np.zeros(nb_steps)
    for k in range(nb_steps):
        z1 = np.array([x_quantile[k]] * Ndata)
        Z = np.array((z1, z2))
        Y = _gumbel_density(Z, theta_gum)
        Y = np.nan_to_num(Y)
        # pdf 2|1, f(comp_2|comp_1)=c(z1,z2)*f(comp_2)
        p_x_x1 = Y * (stats.lognorm.pdf(x, s=s, loc=0, scale=scale))
        # Estimate CDF from PDF
        dum = np.cumsum(p_x_x1)
        cdf = dum / (dum[Ndata - 1])
        # Result of conditional CDF derived based on Gumbel copula
        table = np.array((x, cdf))
        table = table.T
        for j in range(Ndata):
            if y_quantile[k] <= table[0, 1]:
                component_2_Gumbel[k] = min(table[:, 0])
                break
            elif y_quantile[k] <= table[j, 1]:
                component_2_Gumbel[k] = (table[j, 0] + table[j - 1, 0]) / 2
                break
            else:
                component_2_Gumbel[k] = table[:, 0].max()
    if return_fit:
        return component_1, component_2_Gumbel, fit
    return component_1, component_2_Gumbel


def _clayton_copula(x1, x2, fit, component_1, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Clayton copula and the inverse first-order reliability
    method.

    Parameters
    ----------
    x1: numpy array
        Component 1 data
    x2: numpy array
        Component 2 data
    fit: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    component_2_Clayton: array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(component_1, np.ndarray):
        raise TypeError(
            f"component_1 must be of type np.ndarray. Got: {type(component_1)}"
        )
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    x_quantile = fit["x_quantile"]
    y_quantile = fit["y_quantile"]
    para_dist_2 = fit["para_dist_2"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    theta_clay = (2.0 * tau) / (1.0 - tau)

    s = para_dist_2[1]
    scale = np.exp(para_dist_2[0])
    z2_Clay = (
        (1.0 - x_quantile ** (-theta_clay) + x_quantile ** (-theta_clay) / y_quantile)
        ** (theta_clay / (1.0 + theta_clay))
    ) ** (-1.0 / theta_clay)

    # lognormal inverse
    component_2_Clayton = stats.lognorm.ppf(z2_Clay, s=s, loc=0, scale=scale)

    fit["theta_clay"] = theta_clay
    fit["tau"] = tau
    fit["z2_Clay"] = z2_Clay

    if return_fit:
        return component_1, component_2_Clayton, fit
    return component_1, component_2_Clayton


def _rosenblatt_copula(x1, x2, fit, component_1, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Rosenblatt transformation and the inverse first-order
    reliability method.

    Parameters
    ----------
    x1: numpy array
        Component 1 data
    x2: numpy array
        Component 2 data
    fit: Dictionay
        Dictionary of the iso-probability results
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1: array
        Calculated x1 values along the contour boundary following
        return to original input orientation. component_1 is not
        specifically used in this calculation but is passed through to
        create a consistent output from all copula methods.
    component_2_Rosenblatt: array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    try:
        x1 = np.array(x1)
    except:
        pass
    try:
        x2 = np.array(x2)
    except:
        pass
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(component_1, np.ndarray):
        raise TypeError(
            f"component_1 must be of type np.ndarray. Got: {type(component_1)}"
        )
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    y_quantile = fit["y_quantile"]
    mean_cond = fit["mean_cond"]
    std_cond = fit["std_cond"]

    # mean of Ln(T) as a function of x1
    lamda_cond = (
        mean_cond[0]
        + mean_cond[1] * component_1
        + mean_cond[2] * component_1**2
        + mean_cond[3] * component_1**3
    )
    # Standard deviation of Ln(x2) as a function of x1
    sigma_cond = std_cond[0] + std_cond[1] * component_1 + std_cond[2] * component_1**2
    # lognormal inverse
    component_2_Rosenblatt = stats.lognorm.ppf(
        y_quantile, s=sigma_cond, loc=0, scale=np.exp(lamda_cond)
    )

    fit["lamda_cond"] = lamda_cond
    fit["sigma_cond"] = sigma_cond

    if return_fit:
        return component_1, component_2_Rosenblatt, fit
    return component_1, component_2_Rosenblatt


def _nonparametric_copula_parameters(x1, x2, max_x1=None, max_x2=None, nb_steps=1000):
    """
    Calculates nonparametric copula parameters

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    max_x1: float
        Defines the max value of x1 to discretize the KDE space
    max_x2:float
        Defines the max value of x2 to discretize the KDE space
    nb_steps: int
        number of points used to discretize KDE space

    Returns
    -------
    nonpara_dist_1:
        x1 points in KDE space and Nonparametric CDF for x1
    nonpara_dist_2:
        x2 points in KDE space and Nonparametric CDF for x2
    nonpara_pdf_2:
        x2 points in KDE space and Nonparametric PDF for x2
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not max_x1:
        max_x1 = x1.max() * 2
    if not max_x2:
        max_x2 = x2.max() * 2
    if not isinstance(max_x1, float):
        raise TypeError(f"max_x1 must be of type float. Got: {type(max_x1)}")
    if not isinstance(max_x2, float):
        raise TypeError(f"max_x2 must be of type float. Got: {type(max_x2)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")

    # Binning
    x1_sorted_index = x1.argsort()
    x1_sorted = x1[x1_sorted_index]
    x2_sorted = x2[x1_sorted_index]

    # Calcualte KDE bounds (potential input)
    min_limit_1 = 0
    min_limit_2 = 0

    # Discretize for KDE
    pts_x1 = np.linspace(min_limit_1, max_x1, nb_steps)
    pts_x2 = np.linspace(min_limit_2, max_x2, nb_steps)

    # Calculate optimal bandwidth for T and Hs
    sig = stats.median_abs_deviation(x2_sorted)
    num = float(len(x2_sorted))
    bwT = sig * (4.0 / (3.0 * num)) ** (1.0 / 5.0)

    sig = stats.median_abs_deviation(x1_sorted)
    num = float(len(x1_sorted))
    bwHs = sig * (4.0 / (3.0 * num)) ** (1.0 / 5.0)

    # Nonparametric PDF for x2
    temp = KDEUnivariate(x2_sorted)
    temp.fit(bw=bwT)
    f_x2 = temp.evaluate(pts_x2)

    # Nonparametric CDF for x1
    temp = KDEUnivariate(x1_sorted)
    temp.fit(bw=bwHs)
    tempPDF = temp.evaluate(pts_x1)
    F_x1 = tempPDF / sum(tempPDF)
    F_x1 = np.cumsum(F_x1)

    # Nonparametric CDF for x2
    F_x2 = f_x2 / sum(f_x2)
    F_x2 = np.cumsum(F_x2)

    nonpara_dist_1 = np.transpose(np.array([pts_x1, F_x1]))
    nonpara_dist_2 = np.transpose(np.array([pts_x2, F_x2]))
    nonpara_pdf_2 = np.transpose(np.array([pts_x2, f_x2]))

    return nonpara_dist_1, nonpara_dist_2, nonpara_pdf_2


def _nonparametric_component(z, nonpara_dist, nb_steps):
    """
    A generalized method for calculating copula components

    Parameters
    ----------
    z: array
        CDF of isoprobability
    nonpara_dist: array
        x1 or x2 points in KDE space and Nonparametric CDF for x1 or x2
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.

    Returns
    -------
    component: array
        nonparametic component values
    """
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")

    component = np.zeros(nb_steps)
    for k in range(0, nb_steps):
        for j in range(0, np.size(nonpara_dist, 0)):
            if z[k] <= nonpara_dist[0, 1]:
                component[k] = min(nonpara_dist[:, 0])
                break
            elif z[k] <= nonpara_dist[j, 1]:
                component[k] = (nonpara_dist[j, 0] + nonpara_dist[j - 1, 0]) / 2
                break
            else:
                component[k] = max(nonpara_dist[:, 0])
    return component


def _nonparametric_gaussian_copula(x1, x2, fit, nb_steps, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Gaussian copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    fit: Dictionary
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula
    component_2_np_gaussian: array
        Component 2 nonparametric Gaussian copula
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    x_component_iso_prob = fit["x_component_iso_prob"]
    y_component_iso_prob = fit["y_component_iso_prob"]
    nonpara_dist_1 = fit["nonpara_dist_1"]
    nonpara_dist_2 = fit["nonpara_dist_2"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    rho_gau = np.sin(tau * np.pi / 2.0)

    # Component 1
    z1 = stats.norm.cdf(x_component_iso_prob)
    z2 = stats.norm.cdf(
        y_component_iso_prob * np.sqrt(1.0 - rho_gau**2.0)
        + rho_gau * x_component_iso_prob
    )

    comps = {
        1: {"z": z1, "nonpara_dist": nonpara_dist_1},
        2: {"z": z2, "nonpara_dist": nonpara_dist_2},
    }

    for c in comps:
        z = comps[c]["z"]
        nonpara_dist = comps[c]["nonpara_dist"]
        comps[c]["comp"] = _nonparametric_component(z, nonpara_dist, nb_steps)

    component_1_np = comps[1]["comp"]
    component_2_np_gaussian = comps[2]["comp"]

    fit["tau"] = tau
    fit["rho"] = rho_gau
    fit["z1"] = z1
    fit["z2"] = z2

    if return_fit:
        return component_1_np, component_2_np_gaussian, fit
    return component_1_np, component_2_np_gaussian


def _nonparametric_clayton_copula(x1, x2, fit, nb_steps, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Clayton copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    fit: Dictionary
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula
    component_2_np_gaussian: array
        Component 2 nonparametric Clayton copula
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    x_component_iso_prob = fit["x_component_iso_prob"]
    x_quantile = fit["x_quantile"]
    y_quantile = fit["y_quantile"]
    nonpara_dist_1 = fit["nonpara_dist_1"]
    nonpara_dist_2 = fit["nonpara_dist_2"]
    nonpara_pdf_2 = fit["nonpara_pdf_2"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    theta_clay = (2.0 * tau) / (1.0 - tau)

    # Component 1 (Hs)
    z1 = stats.norm.cdf(x_component_iso_prob)
    z2_clay = (
        (1 - x_quantile ** (-theta_clay) + x_quantile ** (-theta_clay) / y_quantile)
        ** (theta_clay / (1.0 + theta_clay))
    ) ** (-1.0 / theta_clay)

    comps = {
        1: {"z": z1, "nonpara_dist": nonpara_dist_1},
        2: {"z": z2_clay, "nonpara_dist": nonpara_dist_2},
    }

    for c in comps:
        z = comps[c]["z"]
        nonpara_dist = comps[c]["nonpara_dist"]
        comps[c]["comp"] = _nonparametric_component(z, nonpara_dist, nb_steps)

    component_1_np = comps[1]["comp"]
    component_2_np_clayton = comps[2]["comp"]

    fit["tau"] = tau
    fit["theta"] = theta_clay
    fit["z1"] = z1
    fit["z2"] = z2_clay

    if return_fit:
        return component_1_np, component_2_np_clayton, fit
    return component_1_np, component_2_np_clayton


def _nonparametric_gumbel_copula(x1, x2, fit, nb_steps, kwargs):
    """
    This function calculates environmental contours of extreme sea
    states using a Gumbel copula with non-parametric marginal
    distribution fits and the inverse first-order reliability method.

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    results: Dictionay
        Dictionary of the iso-probability results
    nb_steps: int
        Discretization of the circle in the normal space used for
        copula component calculation.
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    component_1_np: array
        Component 1 nonparametric copula
    component_2_np_gumbel: array
        Component 2 nonparametric Gumbel copula
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")
    return_fit = kwargs.get("return_fit", False)
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be a bool. Got: {type(return_fit)}"
        )

    Ndata = 1000

    x_quantile = fit["x_quantile"]
    y_quantile = fit["y_quantile"]
    nonpara_dist_1 = fit["nonpara_dist_1"]
    nonpara_dist_2 = fit["nonpara_dist_2"]
    nonpara_pdf_2 = fit["nonpara_pdf_2"]

    # Calculate Kendall's tau
    tau = stats.kendalltau(x2, x1)[0]
    theta_gum = 1.0 / (1.0 - tau)

    # Component 1 (Hs)
    z1 = x_quantile
    component_1_np = _nonparametric_component(z1, nonpara_dist_1, nb_steps)

    pts_x2 = nonpara_pdf_2[:, 0]
    f_x2 = nonpara_pdf_2[:, 1]
    F_x2 = nonpara_dist_2[:, 1]

    component_2_np_gumbel = np.zeros(nb_steps)
    for k in range(nb_steps):
        z1 = np.array([x_quantile[k]] * Ndata)
        Z = np.array((z1.T, F_x2))
        Y = _gumbel_density(Z, theta_gum)
        Y = np.nan_to_num(Y)
        # pdf 2|1
        p_x2_x1 = Y * f_x2
        # Estimate CDF from PDF
        dum = np.cumsum(p_x2_x1)
        cdf = dum / (dum[Ndata - 1])
        table = np.array((pts_x2, cdf))
        table = table.T
        for j in range(Ndata):
            if y_quantile[k] <= table[0, 1]:
                component_2_np_gumbel[k] = min(table[:, 0])
                break
            elif y_quantile[k] <= table[j, 1]:
                component_2_np_gumbel[k] = (table[j, 0] + table[j - 1, 0]) / 2
                break
            else:
                component_2_np_gumbel[k] = max(table[:, 0])

    fit["tau"] = tau
    fit["theta"] = theta_gum
    fit["z1"] = z1
    fit["pts_x2"] = pts_x2
    fit["f_x2"] = f_x2
    fit["F_x2"] = F_x2

    if return_fit:
        return component_1_np, component_2_np_gumbel, fit
    return component_1_np, component_2_np_gumbel


def _bivariate_KDE(x1, x2, bw, fit, nb_steps, Ndata_bivariate_KDE, kwargs):
    """
    Contours generated under this class will use a non-parametric KDE to
    fit the joint distribution. This function calculates environmental
    contours of extreme sea states using a bivariate KDE to estimate
    the joint distribution. The contour is then calculated directly
    from the joint distribution.

    Parameters
    ----------
    x1: array
        Component 1 data
    x2: array
        Component 2 data
    bw: np.array
        Array containing KDE bandwidth for x1 and x2
    fit: Dictionay
        Dictionary of the iso-probability results
    nb_steps: int
        number of points used to discretize KDE space
    max_x1: float
        Defines the max value of x1 to discretize the KDE space
    max_x2: float
        Defines the max value of x2 to discretize the KDE space
    kwargs : optional
        return_fit: boolean
              Will return fitting parameters used. Default False.

    Returns
    -------
    x1_bivariate_KDE: array
        Calculated x1 values along the contour boundary following
        return to original input orientation.
    x2_bivariate_KDE: array
        Calculated x2 values along the contour boundary following
        return to original input orientation.
    fit: Dictionary (optional)
        If return_fit=True. Dictionary with iso-probabilities passed
        with additional fit metrics from the copula method.
    """
    if not isinstance(x1, np.ndarray):
        raise TypeError(f"x1 must be of type np.ndarray. Got: {type(x1)}")
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be of type np.ndarray. Got: {type(x2)}")
    if not isinstance(nb_steps, int):
        raise TypeError(f"nb_steps must be of type int. Got: {type(nb_steps)}")

    max_x1 = kwargs.get("max_x1", None)
    max_x2 = kwargs.get("max_x2", None)
    log_transform = kwargs.get("log_transform", False)
    return_fit = kwargs.get("return_fit", False)

    if isinstance(max_x1, type(None)):
        max_x1 = x1.max() * 2
    if isinstance(max_x2, type(None)):
        max_x2 = x2.max() * 2
    if not isinstance(max_x1, float):
        raise TypeError(f"max_x1 must be of type float. Got: {type(max_x1)}")
    if not isinstance(max_x2, float):
        raise TypeError(f"max_x2 must be of type float. Got: {type(max_x2)}")
    if not isinstance(log_transform, bool):
        raise TypeError(
            f"If specified, log_transform must be of type bool. Got: {type(log_transform)}"
        )
    if not isinstance(return_fit, bool):
        raise TypeError(
            f"If specified, return_fit must be of type bool. Got: {type(return_fit)}"
        )

    p_f = fit["exceedance_probability"]

    min_limit_1 = 0.01
    min_limit_2 = 0.01
    pts_x1 = np.linspace(min_limit_1, max_x1, Ndata_bivariate_KDE)
    pts_x2 = np.linspace(min_limit_2, max_x2, Ndata_bivariate_KDE)
    pt1, pt2 = np.meshgrid(pts_x2, pts_x1)
    mesh_pts_x2 = pt1.flatten()
    mesh_pts_x1 = pt2.flatten()

    # Transform gridded points using log
    ty = [x2, x1]
    xi = [mesh_pts_x2, mesh_pts_x1]
    txi = xi
    if log_transform:
        ty = [np.log(x2), np.log(x1)]
        txi = [np.log(mesh_pts_x2), np.log(mesh_pts_x1)]

    m = len(txi[0])
    n = len(ty[0])
    d = 2

    # Create contour
    f = np.zeros((1, m))
    weight = np.ones((1, n))
    for i in range(0, m):
        ftemp = np.ones((n, 1))
        for j in range(0, d):
            z = (txi[j][i] - ty[j]) / bw[j]
            fk = stats.norm.pdf(z)
            if log_transform:
                fnew = fk * (1 / np.transpose(xi[j][i]))
            else:
                fnew = fk
            fnew = np.reshape(fnew, (n, 1))
            ftemp = np.multiply(ftemp, fnew)
        f[:, i] = np.dot(weight, ftemp)

    fhat = f.reshape(100, 100)
    vals = plt.contour(pt1, pt2, fhat, levels=[p_f])
    plt.clf()
    x1_bivariate_KDE = []
    x2_bivariate_KDE = []

    if mpl_version < (3, 8):  # For versions before 3.8
        segments = vals.allsegs[0]
    else:
        segments = [path.vertices for path in vals.get_paths()]

    for seg in segments:
        x1_bivariate_KDE.append(seg[:, 1])
        x2_bivariate_KDE.append(seg[:, 0])

    x1_bivariate_KDE = np.transpose(np.asarray(x1_bivariate_KDE)[0])
    x2_bivariate_KDE = np.transpose(np.asarray(x2_bivariate_KDE)[0])

    fit["mesh_pts_x1"] = mesh_pts_x1
    fit["mesh_pts_x2"] = mesh_pts_x2
    fit["ty"] = ty
    fit["xi"] = xi
    fit["contour_vals"] = vals

    if return_fit:
        return x1_bivariate_KDE, x2_bivariate_KDE, fit
    return x1_bivariate_KDE, x2_bivariate_KDE


# Sampling
def samples_full_seastate(
    x1,
    x2,
    points_per_interval,
    return_periods,
    sea_state_duration,
    method="PCA",
    bin_size=250,
):
    """
    Sample a sea state between contours of specified return periods.

    This function is used for the full sea state approach for the
    extreme load. See Coe et al. 2018 for more details. It was
    originally part of WDRT.

    Coe, R. G., Michelen, C., Eckert-Gallup, A., &
    Sallaberry, C. (2018). Full long-term design response analysis of a
    wave energy converter. Renewable Energy, 116, 356-366.

    Parameters
    ----------
    x1: list, np.ndarray, pd.Series, xr.DataArray
        Component 1 data
    x2: list, np.ndarray, pd.Series, xr.DataArray
        Component 2 data
    points_per_interval : int
        Number of sample points to be calculated per contour interval.
    return_periods: np.array
        Vector of return periods that define the contour intervals in
        which samples will be taken. Values must be greater than zero
        and must be in increasing order.
    sea_state_duration : int or float
        `x1` and `x2` sample rate (seconds)
    method: string or list
        Copula method to apply. Currently only 'PCA' is implemented.
    bin_size : int
        Number of data points in each bin

    Returns
    -------
    Hs_Samples: np.array
        Vector of Hs values for each sample point.
    Te_Samples: np.array
        Vector of Te values for each sample point.
    weight_points: np.array
        Vector of probabilistic weights for each sampling point
        to be used in risk calculations.
    """
    if method != "PCA":
        raise NotImplementedError(
            "Full sea state sampling is currently only implemented using "
            + "the 'PCA' method."
        )
    x1 = to_numeric_array(x1, "x1")
    x2 = to_numeric_array(x2, "x2")
    if not isinstance(points_per_interval, int):
        raise TypeError(
            f"points_per_interval must be of int. Got: {type(points_per_interval)}"
        )
    if not isinstance(return_periods, np.ndarray):
        raise TypeError(
            f"return_periods must be of type np.ndarray. Got: {type(return_periods)}"
        )
    if not isinstance(sea_state_duration, (int, float)):
        raise TypeError(
            f"sea_state_duration must be of int or float. Got: {type(sea_state_duration)}"
        )
    if not isinstance(method, (str, list)):
        raise TypeError(f"method must be of type string or list. Got: {type(method)}")
    if not isinstance(bin_size, int):
        raise TypeError(f"bin_size must be of int. Got: {type(bin_size)}")

    pca_fit = _principal_component_analysis(x1, x2, bin_size)

    # Calculate line where Hs = 0 to avoid sampling Hs in negative space
    t_zeroline = np.linspace(2.5, 30, 1000)
    t_zeroline = np.transpose(t_zeroline)
    h_zeroline = np.zeros(len(t_zeroline))

    # Transform zero line into principal component space
    coeff = pca_fit["principal_axes"]
    shift = pca_fit["shift"]
    comp_zeroline = np.dot(np.transpose(np.vstack([h_zeroline, t_zeroline])), coeff)
    comp_zeroline[:, 1] = comp_zeroline[:, 1] + shift

    comp1 = pca_fit["x1_fit"]
    c1_zeroline_prob = stats.invgauss.cdf(
        comp_zeroline[:, 0], mu=comp1["mu"], loc=0, scale=comp1["scale"]
    )

    mu_slope = pca_fit["mu_fit"].slope
    mu_intercept = pca_fit["mu_fit"].intercept
    mu_zeroline = mu_slope * comp_zeroline[:, 0] + mu_intercept

    sigma_polynomial_coeffcients = pca_fit["sigma_fit"].x
    sigma_zeroline = np.polyval(sigma_polynomial_coeffcients, comp_zeroline[:, 0])
    c2_zeroline_prob = stats.norm.cdf(
        comp_zeroline[:, 1], loc=mu_zeroline, scale=sigma_zeroline
    )

    c1_normzeroline = stats.norm.ppf(c1_zeroline_prob, 0, 1)
    c2_normzeroline = stats.norm.ppf(c2_zeroline_prob, 0, 1)

    return_periods = np.asarray(return_periods)
    contour_probs = 1 / (365 * 24 * 60 * 60 / sea_state_duration * return_periods)

    # Reliability contour generation
    # Calculate reliability
    beta_lines = stats.norm.ppf((1 - contour_probs), 0, 1)
    # Add zero as lower bound to first contour
    beta_lines = np.hstack((0, beta_lines))
    # Discretize the circle
    theta_lines = np.linspace(0, 2 * np.pi, 1000)
    # Add probablity of 1 to the reliability set, corresponding to
    # probability of the center point of the normal space
    contour_probs = np.hstack((1, contour_probs))

    # Vary U1,U2 along circle sqrt(U1^2+U2^2) = beta
    u1_lines = np.dot(np.cos(theta_lines[:, None]), beta_lines[None, :])

    # Removing values on the H_s = 0 line that are far from the circles in the
    # normal space that will be evaluated to speed up calculations
    minval = np.amin(u1_lines) - 0.5
    mask = c1_normzeroline > minval
    c1_normzeroline = c1_normzeroline[mask]
    c2_normzeroline = c2_normzeroline[mask]

    # Transform to polar coordinates
    theta_zeroline = np.arctan2(c2_normzeroline, c1_normzeroline)
    rho_zeroline = np.sqrt(c1_normzeroline**2 + c2_normzeroline**2)
    theta_zeroline[theta_zeroline < 0] = theta_zeroline[theta_zeroline < 0] + 2 * np.pi

    sample_alpha, sample_beta, weight_points = _generate_sample_data(
        beta_lines, rho_zeroline, theta_zeroline, points_per_interval, contour_probs
    )

    # Sample transformation to principal component space
    sample_u1 = sample_beta * np.cos(sample_alpha)
    sample_u2 = sample_beta * np.sin(sample_alpha)

    comp1_sample = stats.invgauss.ppf(
        stats.norm.cdf(sample_u1, loc=0, scale=1),
        mu=comp1["mu"],
        loc=0,
        scale=comp1["scale"],
    )
    mu_sample = mu_slope * comp1_sample + mu_intercept

    # Calculate sigma values at each point on the circle
    sigma_sample = np.polyval(sigma_polynomial_coeffcients, comp1_sample)

    # Use calculated mu and sigma values to calculate C2 along the contour
    comp2_sample = stats.norm.ppf(
        stats.norm.cdf(sample_u2, loc=0, scale=1), loc=mu_sample, scale=sigma_sample
    )

    # Sample transformation into Hs-T space
    h_sample, t_sample = _princomp_inv(comp1_sample, comp2_sample, coeff, shift)

    return h_sample, t_sample, weight_points


def samples_contour(t_samples, t_contour, hs_contour):
    """
    Get Hs points along a specified environmental contour using
    user-defined T values.

    Parameters
    ----------
    t_samples : list, np.ndarray, pd.Series, xr.DataArray
        Points for sampling along return contour
    t_contour : list, np.ndarray, pd.Series, xr.DataArray
        T values along contour
    hs_contour : list, np.ndarray, pd.Series, xr.DataArray
        Hs values along contour

    Returns
    -------
    hs_samples : np.ndarray
        points sampled along return contour
    """
    t_samples = to_numeric_array(t_samples, "t_samples")
    t_contour = to_numeric_array(t_contour, "t_contour")
    hs_contour = to_numeric_array(hs_contour, "hs_contour")

    # finds minimum and maximum energy period values
    amin = np.argmin(t_contour)
    amax = np.argmax(t_contour)
    aamin = np.min([amin, amax])
    aamax = np.max([amin, amax])
    # finds points along the contour
    w1 = hs_contour[aamin:aamax]
    w2 = np.concatenate((hs_contour[aamax:], hs_contour[:aamin]))
    if np.max(w1) > np.max(w2):
        x1 = t_contour[aamin:aamax]
        y1 = hs_contour[aamin:aamax]
    else:
        x1 = np.concatenate((t_contour[aamax:], t_contour[:aamin]))
        y1 = np.concatenate((hs_contour[aamax:], hs_contour[:aamin]))
    # sorts data based on the max and min energy period values
    ms = np.argsort(x1)
    x = x1[ms]
    y = y1[ms]
    # interpolates the sorted data
    si = interp.interp1d(x, y)
    # finds the wave height based on the user specified energy period values
    hs_samples = si(t_samples)

    return hs_samples


def _generate_sample_data(
    beta_lines, rho_zeroline, theta_zeroline, points_per_interval, contour_probs
):
    """
    Calculate radius, angle, and weight for each sample point

    Parameters
    ----------
    beta_lines: list, np.ndarray, pd.Series, xr.DataArray
        Array of mu fitting function parameters.
    rho_zeroline: list, np.ndarray, pd.Series, xr.DataArray
        Array of radii
    theta_zeroline: list, np.ndarray, pd.Series, xr.DataArray
    points_per_interval: int
    contour_probs: list, np.ndarray, pd.Series, xr.DataArray

    Returns
    -------
    sample_alpha: np.array
        Array of fitted sample angle values.
    sample_beta: np.array
        Array of fitted sample radius values.
    weight_points: np.array
        Array of weights for each point.
    """
    beta_lines = to_numeric_array(beta_lines, "beta_lines")
    rho_zeroline = to_numeric_array(rho_zeroline, "rho_zeroline")
    theta_zeroline = to_numeric_array(theta_zeroline, "theta_zeroline")
    contour_probs = to_numeric_array(contour_probs, "contour_probs")
    if not isinstance(points_per_interval, int):
        raise TypeError(
            f"points_per_interval must be of type int. Got: {type(points_per_interval)}"
        )

    num_samples = (len(beta_lines) - 1) * points_per_interval
    alpha_bounds = np.zeros((len(beta_lines) - 1, 2))
    angular_dist = np.zeros(len(beta_lines) - 1)
    angular_ratio = np.zeros(len(beta_lines) - 1)
    alpha = np.zeros((len(beta_lines) - 1, points_per_interval + 1))
    weight = np.zeros(len(beta_lines) - 1)
    sample_beta = np.zeros(num_samples)
    sample_alpha = np.zeros(num_samples)
    weight_points = np.zeros(num_samples)

    # Loop over contour intervals
    for i in range(len(beta_lines) - 1):
        # Check if any of the radii for the Hs=0, line are smaller than
        # the radii of the contour, meaning that these lines intersect
        r = rho_zeroline - beta_lines[i + 1] + 0.01
        if any(r < 0):
            left = np.amin(np.where(r < 0))
            right = np.amax(np.where(r < 0))
            # Save sampling bounds
            alpha_bounds[i, :] = (
                theta_zeroline[left],
                theta_zeroline[right] - 2 * np.pi,
            )
        else:
            alpha_bounds[i, :] = np.array((0, 2 * np.pi))
        # Find the angular distance that will be covered by sampling the disc
        angular_dist[i] = sum(abs(alpha_bounds[i]))
        # Calculate ratio of area covered for each contour
        angular_ratio[i] = angular_dist[i] / (2 * np.pi)
        # Discretize the remaining portion of the disc into 10 equally spaced
        # areas to be sampled
        alpha[i, :] = np.arange(
            min(alpha_bounds[i]),
            max(alpha_bounds[i]) + 0.1,
            angular_dist[i] / points_per_interval,
        )
        # Calculate the weight of each point sampled per contour
        weight[i] = (
            (contour_probs[i] - contour_probs[i + 1])
            * angular_ratio[i]
            / points_per_interval
        )
        for j in range(points_per_interval):
            # Generate sample radius by adding a randomly sampled distance to
            # the 'disc' lower bound
            sample_beta[(i) * points_per_interval + j] = beta_lines[
                i
            ] + np.random.random_sample() * (beta_lines[i + 1] - beta_lines[i])
            # Generate sample angle by adding a randomly sampled distance to
            # the lower bound of the angle defining a discrete portion of the
            # 'disc'
            sample_alpha[(i) * points_per_interval + j] = alpha[
                i, j
            ] + np.random.random_sample() * (alpha[i, j + 1] - alpha[i, j])
            # Save the weight for each sample point
            weight_points[i * points_per_interval + j] = weight[i]

    return sample_alpha, sample_beta, weight_points


def _princomp_inv(princip_data1, princip_data2, coeff, shift):
    """
    Take the inverse of the principal component rotation given data,
    coefficients, and shift.

    Parameters
    ----------
    princip_data1: np.array
        Array of Component 1 values.
    princip_data2: np.array
        Array of Component 2 values.
    coeff: np.array
        Array of principal component coefficients.
    shift: float
        Shift applied to Component 2 to make all values positive.

    Returns
    -------
    original1: np.array
        Hs values following rotation from principal component space.
    original2: np.array
        T values following rotation from principal component space.
    """
    if not isinstance(princip_data1, np.ndarray):
        raise TypeError(
            f"princip_data1 must be of type np.ndarray. Got: {type(princip_data1)}"
        )
    if not isinstance(princip_data2, np.ndarray):
        raise TypeError(
            f"princip_data2 must be of type np.ndarray. Got: {type(princip_data2)}"
        )
    if not isinstance(coeff, np.ndarray):
        raise TypeError(f"coeff must be of type np.ndarray. Got: {type(coeff)}")
    if not isinstance(shift, float):
        raise TypeError(f"shift must be of type float. Got: {type(shift)}")

    original1 = np.zeros(len(princip_data1))
    original2 = np.zeros(len(princip_data1))
    for i in range(len(princip_data2)):
        original1[i] = (
            (coeff[0, 1] * (princip_data2[i] - shift))
            + (coeff[0, 0] * princip_data1[i])
        ) / (coeff[0, 1] ** 2 + coeff[0, 0] ** 2)
        original2[i] = (
            (coeff[0, 1] * princip_data1[i])
            - (coeff[0, 0] * (princip_data2[i] - shift))
        ) / (coeff[0, 1] ** 2 + coeff[0, 0] ** 2)
    return original1, original2
