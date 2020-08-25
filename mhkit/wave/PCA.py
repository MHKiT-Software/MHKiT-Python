from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import mean_squared_error
import scipy.optimize as optim
import scipy.stats as stats
import numpy as np

def principal_component_analysis(x1, x2, size_bin=250):
    '''
    Performs a modified principal component analysis (PCA) 
    [Eckert et. al 2015] on two variables (x1, x2). This is donce 
    by converting the the x1 and x2 data into the principal componet 
    domain using the scikit-learn PCA method. For wave resource 
    characterization (Hm0 and Te (or Tp))  the standard PCA method
    is known to not remove all of the dependence between the 
    two variables. To remove this dependence this PCA function
    quantifies the relation between the two variables in the PCA space
    by binning the data into bin sizes of "size_bin".  
    generated will use principal component analysis (PCA) with improved 
    distribution fitting  and the I-FORM.
    
    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., & 
    Neary, V. S. (2016). Application of principal component 
    analysis (PCA) and improved joint probability distributions to 
    the inverse first-order reliability method (I-FORM) for predicting 
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ----------
    x1: array like
        Component 1 data
    x2: array like
        Component 2 data        
    size_bin : float
        Data points in each bin 
        
    Returns
    -------
    PCA: Dictionary 
       Keys:
       -----       
       'principal_axes': sign corrected PCA axes 
       'shift'         : The shift applied to x2 
       'x1_fit'        : gaussian fit of x1 data
       'mu_param'      : fit to _mu_fcn
       'sigma_param'   : fit to _sig_fits            
    '''

    N = len(x1)  # Number of observations
    max_bin = N*0.25
    if size_bin > max_bin:
        size_bin = max_bin
        max_bin_N = round(N*0.25, 2)
        msg=['The bin size has been set to the max bin size for' +
             f'this buoy: {max_bin_N}']           
        print(msg[0])
           
    pca = skPCA(n_components=2)
    
    mean_location=0    
    x1_mean_centered = x1 - x1.mean(axis=0)
    x2_mean_centered = x2 - x2.mean(axis=0)
    pca.fit(np.array((x1_mean_centered, x2_mean_centered)).T)
    
    # The directions of maximum variance in the data
    principal_axes = pca.components_
    
    # Apply correct/expected sign convention    
    principal_axes = abs(principal_axes)  
    principal_axes[1, 1] = -1.0 * principal_axes[1, 1]  

    # Principal direction components of each component
    x1_x2_components = np.dot(np.array((x1, x2)).T, principal_axes)  
    x1_components = x1_x2_components[:, 0]
    x2_components = x1_x2_components[:, 1]
       
    # Apply shift to Component 2 to make all values positive
    shift = abs(min(x2_components)) + 0.1 
    x2_components = x2_components + shift    

    # Fitting distribution of component 1
    x1_sorted_index = x1_components.argsort() 
    x1_sorted = x1_components[x1_sorted_index]
    x2_sorted = x2_components[x1_sorted_index]    
    
    x1_fit_results = stats.invgauss.fit(x1_sorted, floc=mean_location)
    x1_fit = { 'mu'    : x1_fit_results[0],
               'loc'   : x1_fit_results[1],
               'scale' : x1_fit_results[2]
             }
         
    size_bin_integer_multiple_of_N = int(np.floor(N / size_bin))
    last_bin_of_size_bin = size_bin*size_bin_integer_multiple_of_N
    
    x1_integer_multiples_of_bin_size = x1_sorted[0:last_bin_of_size_bin]    
    x2_integer_multiples_of_bin_size = x2_sorted[0:last_bin_of_size_bin] 
    
    x1_bins = np.split(x1_integer_multiples_of_bin_size, 
                       size_bin_integer_multiple_of_N)
    x2_bins = np.split(x2_integer_multiples_of_bin_size, 
                       size_bin_integer_multiple_of_N)
    
    x1_last_bin = x1_sorted[last_bin_of_size_bin:]    
    x2_last_bin = x2_sorted[last_bin_of_size_bin:]    
    
    x1_bins.append(x1_last_bin)
    x2_bins.append(x2_last_bin)
    
    x1_means = np.array([]) 
    x2_means = np.array([]) 
    x2_stds   = np.array([])     
    
    for x1_bin, x2_bin in zip(x1_bins, x2_bins):                    
        x1_bin_mean = x1_bin.mean()
        x1_means = np.append(x1_means, x1_bin_mean)        
        
        # Calcualte normal distribution parameters for x2 in each bin
        x2_bin_sorted = np.sort(x2_bin)
        x2_bin_mean = x2_bin_sorted.mean()
        x2_bin_std  = x2_bin_sorted.std()
        
        x2_bin_mean = np.mean(x2_bin_sorted)
        x2_means = np.append(x2_means, x2_bin_mean) 
        
        x2_bin_std  = np.std(x2_bin_sorted)
        x2_stds = np.append(x2_stds, x2_bin_std) 
    
    mu_fit = stats.linregress(x1_means, x2_means)    
    
    # Constrained optimization of sigma
    sigma_polynomial_order=2
    sig_0 = 0.1 * np.ones(sigma_polynomial_order+1)
    
    def _objective_function(sig_p, x1_means, x2_sigs):
        return mean_squared_error(np.polyval(sig_p, x1_means), x2_sigs)
    
    # Constraint Functions
    y_intercept_gt_0 = lambda sig_p: (sig_p[2])
    sig_polynomial_min_gt_0 = lambda sig_p: (sig_p[2] - (sig_p[1]**2) / \
                                             (4 * sig_p[0]))    
    constraints = ({'type': 'ineq', 'fun': y_intercept_gt_0},
                   {'type': 'ineq', 'fun': sig_polynomial_min_gt_0})    
    
    sigma_fit = optim.minimize(_objective_function, x0=sig_0, 
                               args=(x1_means, x2_stds),
                               method='SLSQP',constraints=constraints)     

    PCA = {
           'principal_axes': principal_axes, 
           'shift'         : shift, 
           'x1_fit'        : x1_fit, 
           'mu_fit'        : mu_fit, 
           'sigma_fit'     : sigma_fit 
           }
    
    return PCA


def getContours(time_ss, time_r, PCA,  nb_steps=1000):
    '''
    
    This function calculates environmental contours of extreme sea states using
    principal component analysis and the inverse first-order reliability
    method (IFORM) failure probability for the desired return period 
    (time_R) given the duration of the measurements (time_ss)

    Eckert-Gallup, A. C., Sallaberry, C. J., Dallman, A. R., & 
    Neary, V. S. (2016). Application of principal component 
    analysis (PCA) and improved joint probability distributions to 
    the inverse first-order reliability method (I-FORM) for predicting 
    extreme sea states. Ocean Engineering, 112, 307-319.

    Parameters
    ___________
    time_ss : float
        Sea state duration (hours) of measurements in input.
    time_r : np.array
        Desired return period (years) for calculation of environmental
        contour, can be a scalar or a vector.
    nb_steps : int
        Discretization of the circle in the normal space used for
        inverse FORM calculation.

    Returns
    -------
    x1_Return : np.array
        Calculated x1 values along the contour boundary following
        return to original input orientation.
    T_Return : np.array
       Calculated T values along the contour boundary following
       return to original input orientation.
    nb_steps : float
        Discretization of the circle in the normal space

    '''

    
    exceedance_probability = 1 / (365 * (24 / time_ss) * time_r)
    iso_probability_radius = stats.norm.ppf((1 - exceedance_probability), 
                                             loc=0, scale=1)  
    discretized_radians = np.linspace(0, 2 * np.pi, num = nb_steps)
    
    x_componenet_iso_prob = iso_probability_radius * \
                            np.cos(discretized_radians)
    y_componenet_iso_prob = iso_probability_radius * \
                            np.sin(discretized_radians)
    
    
    mu       = PCA['x1_fit']['mu']
    mu_loc   = PCA['x1_fit']['loc']
    mu_scale = PCA['x1_fit']['scale']
    # Calculate C1 values along the contour
    x_quantile = stats.norm.cdf(x_componenet_iso_prob, loc=0, scale=1)
    compoenent_1 = stats.invgauss.ppf(x_quantile, mu=mu , loc=mu_loc, 
                                      scale=mu_scale )
    mu_slope     = PCA['mu_fit'].slope
    mu_intercept = PCA['mu_fit'].intercept    
    # Calculate mu values at each point on the circle
    mu_R = mu_slope * compoenent_1 + mu_intercept
    # Calculate sigma values at each point on the circle
    sigma_val = PCA['sigma_fit'].x[0] * compoenent_1**2 + \
                PCA['sigma_fit'].x[1] * compoenent_1 + PCA['sigma_fit'].x[2]
                
    # Use calculated mu and sigma values to calculate C2 along the contour
    Comp2_R = stats.norm.ppf(stats.norm.cdf(y_componenet_iso_prob, 
                                            loc=0, scale=1),
                             loc=mu_R, scale=sigma_val)
    # Calculate x1 and T along the contour
    x1_Return, T_Return = _princomp_inv(compoenent_1, 
                                        Comp2_R, 
                                        PCA['principal_axes'], 
                                        PCA['shift'])
    x1_Return = np.maximum(0, x1_Return)  # Remove negative values
    x1_ReturnContours = x1_Return
    T_ReturnContours = T_Return
    return x1_Return, T_Return


def _princomp_inv(princip_data1, princip_data2, principal_axes, shift):
    '''
    Takes the inverse of the principal component rotation given data,
    coefficients, and shift. Used in the EA and getSamples functions.
    Parameters
    ----------
    princip_data1: np.array
                   Array of Component 1 values.
    princip_data2: np.array
                   Array of Component 2 values.
    principal_axes: np.array
           Array of principal component coefficients.
    shift: float
           Shift applied to Component 2 to make all values positive.
    Returns
    -------
    original1: np.array
               x1 values following rotation from principal component space.
    original2: np.array
               T values following rotation from principal component space.
    '''
    original1 = np.zeros(len(princip_data1))
    original2 = np.zeros(len(princip_data1))
    for i in range(len(princip_data2)):
        original1[i] = (((principal_axes[0, 1] * (princip_data2[i] - shift)) +
                         (principal_axes[0, 0] * princip_data1[i])) / (principal_axes[0, 1]**2 +
                                                              principal_axes[0, 0]**2))
        original2[i] = (((principal_axes[0, 1] * princip_data1[i]) -
                         (principal_axes[0, 0] * (princip_data2[i] -
                                         shift))) / (principal_axes[0, 1]**2 + principal_axes[0, 0]**2))
    return original1, original2
