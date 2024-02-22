"""
This package provides tools and functions for extreme value analysis
and wave data statistics.

It includes methods for calculating peaks over threshold, estimating
short-term extreme distributions,and performing wave amplitude 
normalization for most likely extreme response analysis.
"""

from mhkit.loads.extreme.extremes import (
    ste_peaks,
    block_maxima,
    ste_block_maxima_gev,
    ste_block_maxima_gumbel,
    ste,
    short_term_extreme,
    full_seastate_long_term_extreme,
)

from mhkit.loads.extreme.mler import (
    mler_coefficients,
    mler_simulation,
    mler_wave_amp_normalize,
    mler_export_time_series,
)

from mhkit.loads.extreme.peaks import (
    _peaks_over_threshold,
    global_peaks,
    number_of_short_term_peaks,
    peaks_distribution_weibull,
    peaks_distribution_weibull_tail_fit,
    automatic_hs_threshold,
    peaks_distribution_peaks_over_threshold,
)

from mhkit.loads.extreme.sample import (
    return_year_value,
)
