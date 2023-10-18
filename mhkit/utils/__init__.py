from .time_utils import matlab_to_datetime, excel_to_datetime, index_to_datetime
from .stat_utils import get_statistics, vector_statistics, unwrap_vector, magnitude_phase, unorm
from .cache import handle_caching, clear_cache
from .upcrossing import upcrossing, peaks, troughs, heights, periods, custom

_matlab = False  # Private variable indicating if mhkit is run through matlab
