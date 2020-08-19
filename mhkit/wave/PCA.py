from sklearn.decomposition import PCA as skPCA
import scipy.optimize as optim
import scipy.stats as stats
import numpy as np

def principal_component_analysis(x1, x2, size_bin=250):
    '''
    Performs a PCA given x1, and x2. Contours generated will use 
    principal component analysis (PCA) with improved 
    distribution fitting (Eckert et. al 2015) and the I-FORM.

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
    
    x2_bins_fit = np.zeros([2,1])  
    x1_means = np.array([])   
    for x1_bin, x2_bin in zip(x1_bins, x2_bins):                    
        x1_bin_mean = x1_bin.mean()
        x1_means = np.append(x1_means, x1_bin_mean)        
        
        # Calcualte normal distribution parameters for x2 in each bin
        x2_bin_sorted = np.sort(x2_bin)
        x2_bin_fit = np.array(stats.norm.fit(x2_bin_sorted))   
        x2_bins_fit = np.append(x2_bins_fit,x2_bin_fit.reshape(2,1), axis=1)
    x2_bins_fit = np.delete(x2_bins_fit,0,axis=1)

    # Use non-linear least squares to fit a function, mu_fcn, to data.
    mu_param, pcov = optim.curve_fit(_mu_fcn,
                                     x1_means.T, 
                                     x2_bins_fit[0, :]
                                     )
    
    sigma_param = _sigma_fits(x1_means, x2_bins_fit[1, :])
    
    PCA = {
           'principal_axes': principal_axes, 
           'shift'         : shift, 
           'x1_fit'        : x1_fit, 
           'mu_param'      : mu_param, 
           'sigma_param'   : sigma_param 
           }
    
    return PCA


def _mu_fcn(x, mu_p_1, mu_p_2):
    ''' 
    Linear fitting function for the mean(mu) of Component 2 normal
    distribution as a function of the Component 1 mean for each bin.
    
    Parameters
    ----------
    mu_p: np.array
           Array of mu fitting function parameters.
    x: np.array
       Array of values (Component 1 mean for each bin) at which to 
       evaluate the mu fitting function.
    
    Returns
    -------
    mu_fit: np.array
            Array of fitted mu values.
    '''
    mu_fit = mu_p_1 * x + mu_p_2
    return mu_fit


def _sigma_fits( Comp1_mean, sigma_vals):
    '''
    Sigma parameter fitting function using penalty optimization.
    
    Parameters
    ----------
    Comp1_mean: np.array
                Mean value of Component 1 for each bin of Component 2.
    sigma_vals: np.array
                Value of Component 2 sigma for each bin derived from normal
                distribution fit.
                
    Returns
    -------
    sig_final: np.array
               Final sigma parameter values after constrained optimization.
    '''
    sig_0 = np.array((0.1, 0.1, 0.1))  # Set initial guess
    rho = 1.0  # Set initial penalty value
    # Set tolerance, very small values (i.e.,smaller than 10^-5) may cause
    # instabilities
    epsilon = 10**-5
    # Set inital beta values using beta function
    Beta1, Beta2 = _betafcn(sig_0, rho)
    # Initial search for minimum value using initial guess
    sig_1 = optim.fmin(func=_objfun_penalty, x0=sig_0,
                       args=(Comp1_mean, sigma_vals, Beta1, Beta2), disp=False)
    # While either the difference between iterations or the difference in
    # objective function evaluation is greater than the tolerance, continue
    # iterating
    while (np.amin(abs(sig_1 - sig_0)) > epsilon and
           abs(_objfun(sig_1, Comp1_mean, sigma_vals) -
               _objfun(sig_0, Comp1_mean, sigma_vals)) > epsilon):
        sig_0 = sig_1
        # Calculate penalties for this iteration
        Beta1, Beta2 = _betafcn(sig_0, rho)
        # Find a new minimum
        sig_1 = optim.fmin(func=_objfun_penalty, x0=sig_0,
                           args=(Comp1_mean, sigma_vals, Beta1, Beta2), disp=False)
        rho = 10 * rho  # Increase penalization
    sig_final = sig_1
    return sig_final


def _betafcn(sig_p, rho):
    '''
    Penalty calculation for sigma parameter fitting function to impose
    positive value constraint.
    Parameters
    ----------
    sig_p: np.array
           Array of sigma fitting function parameters.
    rho: float
         Penalty function variable that drives the solution towards
         required constraint.
    Returns
    -------
    Beta1: float
           Penalty function variable that applies the constraint requiring
           the y-intercept of the sigma fitting function to be greater than
           or equal to 0.
    Beta2: float
           Penalty function variable that applies the constraint requiring
           the minimum of the sigma fitting function to be greater than or
           equal to 0.
    '''
    if -sig_p[2] <= 0:
        Beta1 = 0.0
    else:
        Beta1 = rho
    if -sig_p[2] + (sig_p[1]**2) / (4 * sig_p[0]) <= 0:
        Beta2 = 0.0
    else:
        Beta2 = rho
    return Beta1, Beta2
        
        
def _objfun_penalty(sig_p, x, y_actual, Beta1, Beta2):
    '''
    Penalty function used for sigma function constrained optimization.
    Parameters
    ----------
    sig_p: np.array
           Array of sigma fitting function parameters.
    x: np.array
       Array of values (Component 1 mean for each bin) at which to evaluate
       the sigma fitting function.
    y_actual: np.array
              Array of actual sigma values for each bin to use in least
              square error calculation with fitted values.
    Beta1: float
           Penalty function variable that applies the constraint requiring
           the y-intercept of the sigma fitting function to be greater than
           or equal to 0.
    Beta2: float
           Penalty function variable that applies the constraint requiring
           the minimum of the sigma fitting function to be greater than or
           equal to 0.
    Returns
    -------
    penalty_fcn: float
                 Objective function result with constraint penalties
                 applied for out of bound solutions.
    '''
    penalty_fcn = (_objfun(sig_p, x, y_actual) + Beta1 * (-sig_p[2])**2 +
                   Beta2 * (-sig_p[2] + (sig_p[1]**2) / (4 * sig_p[0]))**2)
    return penalty_fcn        
        
        
def _objfun(sig_p, x, y_actual):
    '''
    Sum of least square error objective function used in sigma
    minimization.
    
    Parameters
    ----------
    sig_p: np.array
           Array of sigma fitting function parameters.
    x: np.array
       Array of values (Component 1 mean for each bin) at which to evaluate
       the sigma fitting function.
    y_actual: np.array
              Array of actual sigma values for each bin to use in least
              square error calculation with fitted values.
              
    Returns
    -------
    obj_fun_result: float
                    Sum of least square error objective function for fitted
                    and actual values.
    '''
    obj_fun_result = np.sum((_sigma_fcn(sig_p, x) - y_actual)**2)
    return obj_fun_result  # Sum of least square error


def _sigma_fcn(sig_p, x):
    '''
    Quadratic fitting formula for the standard deviation(sigma) of 
    Component 2 normal distribution as a function of the Component 1 
    mean for each bin. Used in the EA and getSamples functions.
    
    Parameters
    ----------
    sig_p: np.array
           Array of sigma fitting function parameters.
    x: np.array
       Array of values (Component 1 mean for each bin) at which to evaluate
       the sigma fitting function.
       
    Returns
    -------
    sigma_fit: np.array
               Array of fitted sigma values.
    '''
    sigma_fit = sig_p[0] * x**2 + sig_p[1] * x + sig_p[2]
    return sigma_fit


def getContours(time_ss, time_r, PCA,  nb_steps=1000):
    '''
    WDRT Extreme Sea State PCA Contour function
    This function calculates environmental contours of extreme sea states using
    principal component analysis and the inverse first-order reliability
    method.

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

    # IFORM
    # Failure probability for the desired return period (time_R) given the
    # duration of the measurements (time_ss)
    p_f = 1 / (365 * (24 / time_ss) * time_r)
    beta = stats.norm.ppf((1 - p_f), loc=0, scale=1)  # Reliability
    theta = np.linspace(0, 2 * np.pi, num = nb_steps)
    # Vary U1, U2 along circle sqrt(U1^2+U2^2)=beta
    U1 = beta * np.cos(theta)
    U2 = beta * np.sin(theta)
    # Calculate C1 values along the contour
    Comp1_R = stats.invgauss.ppf(stats.norm.cdf(U1, loc=0, scale=1),
                                 mu= PCA['x1_fit']['mu'], loc=0,
                                 scale= PCA['x1_fit']['scale'])
    # Calculate mu values at each point on the circle
    mu_R = _mu_fcn(Comp1_R, PCA['mu_param'][0], PCA['mu_param'][1])
    # Calculate sigma values at each point on the circle
    sigma_R = _sigma_fcn(PCA['sigma_param'], Comp1_R)
    # Use calculated mu and sigma values to calculate C2 along the contour
    Comp2_R = stats.norm.ppf(stats.norm.cdf(U2, loc=0, scale=1),
                             loc=mu_R, scale=sigma_R)

    # Calculate x1 and T along the contour
    x1_Return, T_Return = _princomp_inv(Comp1_R, 
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
