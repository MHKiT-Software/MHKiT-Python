import warnings as _warn
from mhkit import wave
from mhkit import river
from mhkit import tidal
from mhkit import qc
from mhkit import utils
from mhkit import power
from mhkit import loads
from mhkit import dolfyn
from mhkit import mooring
from mhkit import acoustics

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
