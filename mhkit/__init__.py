import warnings as _warn
import importlib

# Register datetime converter for a matplotlib plotting methods
from pandas.plotting import register_matplotlib_converters as _rmc

_rmc()

# Use targeted warning configuration
from mhkit.warnings import configure_warnings

configure_warnings()

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
    known_modules = [
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
    ]

    if name in known_modules:
        try:
            return importlib.import_module(f"mhkit.{name}")
        except ModuleNotFoundError:
            error_msg = "Module dependencies not found.\n"
            error_msg += f"To install the {name} module, run:\n"
            error_msg += f"  pip install mhkit[{name}]\n\n"
            error_msg += "Or install all modules with:\n"
            error_msg += "  pip install mhkit[all]"
    else:
        error_msg = f"module 'mhkit' has no attribute '{name}'"

    raise AttributeError(error_msg)
