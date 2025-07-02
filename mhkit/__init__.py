import warnings as _warn
import importlib

# Register datetime converter for a matplotlib plotting methods
from pandas.plotting import register_matplotlib_converters as _rmc

_rmc()

# Ignore future warnings
_warn.simplefilter(action="ignore", category=FutureWarning)

__version__ = "v0.9.0"

__copyright__ = """
Copyright 2019, Alliance for Sustainable Energy, LLC under the terms of 
Contract DE-AC36-08GO28308, Battelle Memorial Institute under the terms of 
Contract DE-AC05-76RL01830, and National Technology & Engineering Solutions of 
Sandia, LLC under the terms of Contract DE-NA0003525. The U.S. Government 
retains certain rights in this software."""

__license__ = "Revised BSD License"


def __getattr__(name):
    """Lazy import modules to handle pip optional dependencies."""
    if name in [
        "wave",
        "river",
        "tidal",
        "qc",
        "utils",
        "power",
        "loads",
        "dolfyn",
        "mooring",
        "acoustics",
    ]:
        return importlib.import_module(f"mhkit.{name}")

    # Enhanced error message with installation instructions
    error_msg = f"module 'mhkit' has no attribute '{name}'"

    # Check if it's a known module that might not be installed
    known_modules = {
        "wave": "wave analysis and resource assessment",
        "river": "river hydrokinetic analysis",
        "tidal": "tidal energy analysis",
        "qc": "quality control tools",
        "utils": "utility functions",
        "power": "power performance analysis",
        "loads": "load analysis and extreme value statistics",
        "dolfyn": "acoustic Doppler current profiler (ADCP/ADV) data processing",
        "mooring": "mooring analysis tools",
        "acoustics": "acoustic analysis tools",
    }

    if name in known_modules:
        error_msg += f"\n\nTo install the {name} module and its dependencies, run:\n"
        error_msg += f"pip install mhkit[{name}]\n\n"
        error_msg += f"Or install all modules with:\n"
        error_msg += f"pip install mhkit[all]\n\n"
        error_msg += f"The {name} module provides {known_modules[name]}."

    raise AttributeError(error_msg)
