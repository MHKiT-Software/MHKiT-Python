"""
The passive acoustics module provides a set of functions
for analyzing and visualizing passive acoustic monitoring 
data deployed in water bodies. This package reads in raw
*.wav* files and conducts basic acoustics analysis and 
visualization.

To start using the module, import it directly from MHKiT:
``from mhkit import acoustics``. The analysis functions
are available directly from the main import, while the 
I/O and graphics submodules are available from 
``acoustics.io`` and  ``acoustics.graphics``, respectively.
The base functions are intended to be used on top of the I/O submodule, and 
include functionality to calibrate data, create spectral densities, sound 
pressure levels, and time or band aggregate spectral data.
"""

from mhkit.acoustics import io, graphics
from .analysis import *
