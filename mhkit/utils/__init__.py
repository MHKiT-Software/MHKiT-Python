from .time_utils import matlab_to_datetime, excel_to_datetime
from .stat_utils import (
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

_matlab = False  # Private variable indicating if mhkit is run through matlab
