"""
This module initializes and imports the essential utility functions for data 
conversion, statistical analysis, caching, and event detection for the 
MHKiT library.
"""

from .time_utils import matlab_to_datetime, excel_to_datetime, index_to_datetime
from .stat_utils import (
    _calculate_statistics,
    get_statistics,
    vector_statistics,
    unwrap_vector,
    magnitude_phase,
    unorm,
)
from .cache import handle_caching, clear_cache
from .upcrossing import upcrossing, peaks, troughs, heights, periods, custom
from .type_handling import (
    to_numeric_array,
    convert_to_dataset,
    convert_to_dataarray,
    convert_nested_dict_and_pandas,
)

# pylint: disable=invalid-name
_matlab = False  # Private variable indicating if mhkit is run through matlab
