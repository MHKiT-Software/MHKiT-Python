"""
The `acoustics` package of the MHKiT (Marine and Hydrokinetic Toolkit) library
provides tools and functionalities for analyzing and visualizing passive
acoustic monitoring data deployed in water bodies. This package reads in raw
wav files and conducts basic acoustics analysis and visualization.
"""

from mhkit.acoustics import io, graphics
from .analysis import *
