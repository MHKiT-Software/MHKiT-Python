"""

This module provides functionality for importing and processing wave hindcast data,
including wind toolkit data and WPTO hindcast data. THe hindcast io module is 
seperated from the general geio module to allow for more efficient handling of
CI tests. 
"""

from mhkit.wave.io.hindcast import wind_toolkit

try:
    from mhkit.wave.io.hindcast import hindcast
except ImportError:
    print(
        "WARNING: Wave WPTO hindcast functions not imported from"
        "MHKiT-Python. If you are using Windows and calling from"
        "MHKiT-MATLAB this is expected."
    )
