"""Wave hindcast data import and processing module.

This module provides functionality for importing and processing wave hindcast data,
including wind toolkit data and WPTO hindcast data. The hindcast io module is
separated from the general io module to allow for more efficient handling of
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
