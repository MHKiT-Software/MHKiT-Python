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
    """Lazy import modules."""
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
    raise AttributeError(f"module 'mhkit' has no attribute '{name}'")
