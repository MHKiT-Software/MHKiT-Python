from scipy.stats import binned_statistic
import pandas as pd 
import numpy as np
import fatpack

def bin_statistics(data,bin_against,bin_edges,data_signal=[]):
    """
    Bins calculated statistics against data signal (or channel) 
    according to IEC TS 62600-3:2020 ED1.
    
    Parameters
    -----------
    data : pandas DataFrame
       Time-series statistics of data signal(s) 
    bin_against : array
        Data signal to bin data against (e.g. wind speed)
    bin_edges : array
        Bin edges with consistent step size
    data_signal : list, optional 
        List of data signal(s) to bin, default = all data signals
    
    Returns
    --------
    bin_mean : pandas DataFrame
        Mean of each bin
    bin_std : pandas DataFrame
        Standard deviation of each bim
    """

    assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFram'   
    try: bin_against = np.asarray(bin_against) 
    except: 'bin_against must be of type np.ndarray'
    try: bin_edges = np.asarray(bin_edges)
    except: 'bin_edges must be of type np.ndarray'    

    # Determine variables to analyze
    if len(data_signal)==0: # if not specified, bin all variables
        data_signal=data.columns.values
    else:
        assert isinstance(data_signal, list), 'must be of type list'

    # Pre-allocate list variables
    bin_stat_list = []
    bin_std_list = []

    # loop through data_signal and get binned means
    for signal_name in data_signal:
        # Bin data
        bin_stat = binned_statistic(bin_against,data[signal_name],
                                    statistic='mean',bins=bin_edges)
        # Calculate std of bins
        std = []
        stdev = pd.DataFrame(data[signal_name])
        stdev.set_index(bin_stat.binnumber,inplace=True)
        for i in range(1,len(bin_stat.bin_edges)):
            try:
                temp = stdev.loc[i].std(ddof=0)
                std.append(temp[0])
            except:
                std.append(np.nan)
        bin_stat_list.append(bin_stat.statistic)
        bin_std_list.append(std)
 
    # Convert to DataFrames
    bin_mean = pd.DataFrame(np.transpose(bin_stat_list),columns=data_signal)
    bin_std = pd.DataFrame(np.transpose(bin_std_list),columns=data_signal)

    # Check for nans 
    if bin_mean.isna().any().any():
        print('Warning: some bins may be empty!')

    return bin_mean, bin_std


def blade_moments(blade_coefficients,flap_offset,flap_raw,edge_offset,edge_raw):
    '''
    Transfer function for deriving blade flap and edge moments using blade matrix.

    Parameters
    -----------
    blade_coefficients : numpy array
        Derived blade calibration coefficients listed in order of D1, D2, D3, D4
    flap_offset : float
        Derived offset of raw flap signal obtained during calibration process
    flap_raw : numpy array
        Raw strain signal of blade in the flapwise direction
    edge_offset : float
        Derived offset of raw edge signal obtained during calibration process
    edge_raw : numpy array
        Raw strain signal of blade in the edgewise direction
    
    Returns
    --------
    M_flap : numpy array
        Blade flapwise moment in SI units
    M_edge : numpy array
        Blade edgewise moment in SI units
    '''
    
    try: blade_coefficients = np.asarray(blade_coefficients)
    except: 'blade_coefficients must be of type np.ndarray'
    try: flap_raw = np.asarray(flap_raw)
    except: 'flap_raw must be of type np.ndarray'    
    try: edge_raw = np.asarray(edge_raw)
    except:  'edge_raw must be of type np.ndarray'    
    
    assert isinstance(flap_offset, (float,int)), 'flap_offset must be of type int or float'
    assert isinstance(edge_offset, (float,int)), 'edge_offset must be of type int or float'
    
    # remove offset from raw signal
    flap_signal = flap_raw - flap_offset
    edge_signal = edge_raw - edge_offset

    # apply matrix to get load signals
    M_flap = blade_coefficients[0]*flap_signal + blade_coefficients[1]*edge_signal
    M_edge = blade_coefficients[2]*flap_signal + blade_coefficients[3]*edge_signal

    return M_flap, M_edge


def damage_equivalent_load(data_signal, m, bin_num=100, data_length=600):
    '''
    Calculates the damage equivalent load of a single data signal (or channel) 
    based on IEC TS 62600-3:2020 ED1. 4-point rainflow counting algorithm from 
    fatpack module is based on the following resources:
        
    - `C. Amzallag et. al. Standardization of the rainflow counting method for
      fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293`
    - `ISO 12110-2, Metallic materials - Fatigue testing - Variable amplitude
      fatigue testing.`
    - `G. Marsh et. al. Review and application of Rainflow residue processing
      techniques for accurate fatigue damage estimation. International Journal
      of Fatigue, 82 (2016) 757-765`
    

    Parameters:
    -----------
    data_signal : array
        Data signal being analyzed
    m : float/int
        Fatigue slope factor of material
    bin_num : int
        Number of bins for rainflow counting method (minimum=100)
    data_length : float/int
        Length of measured data (seconds)
    
    Returns
    --------
    DEL : float
        Damage equivalent load (DEL) of single data signal
    '''
    
    try: data_signal = np.array(data_signal)
    except: 'data_signal must be of type np.ndarray'
    assert isinstance(m, (float,int)), 'm must be of type float or int'
    assert isinstance(bin_num, (float,int)), 'bin_num must be of type float or int'
    assert isinstance(data_length, (float,int)), 'data_length must be of type float or int'

    rainflow_ranges = fatpack.find_rainflow_ranges(data_signal,k=256)

    # Range count and bin
    Nrf, Srf = fatpack.find_range_count(rainflow_ranges, bin_num)

    DELs = Srf**m * Nrf / data_length
    DEL = DELs.sum() ** (1/m)

    return DEL
