"""
This module contains functions for processing strain measurements from 90 degree,
3-gauge strain rosettes, calculating moments and torsion with multiple methods.

Functions in this module include:

- harmonics: ...
"""

import pandas as pd
import numpy as np
import xarray as xr
from mhkit.utils import convert_to_dataset


def load_strain():
    a = 1

def calculate_primary_strains(ec, eb, ea, gauge_type):
    if gauge_type == 90:
        axial_strain = ec
        shear_strain = ea
        coupled_strain = eb - (ea+ec)/2
    elif gauge_type == 120:
        # TODO
        pass
    else:
        raise ValueError(f"gauge_type must be 90 or 120. Got: {gauge_type}")

    return axial_strain, shear_strain, coupled_strain
 
def theoretical_loads(rosette1, rosette2, elastic_modulus, shear_modulus, shear_width, transverse_width, radius):
    normal = 0.5 * (rosette1['axial_strain'] + rosette2['axial_strain']) * \
             elastic_modulus * (shear_width * transverse_width - np.pi * radius * radius)
    moment = (rosette1['axial_strain'] - rosette2['axial_strain'])/shear_width * \
             elastic_modulus * (transverse_width*shear_width**3 / 12 - np.pi*radius**4 / 4)
    torsion1 = theoretical_torsion(rosette1['coupled_strain'], shear_modulus, shear_width, transverse_width, radius)
    torsion2 = theoretical_torsion(rosette2['coupled_strain'], shear_modulus, shear_width, transverse_width, radius)

    return normal, moment, torsion1, torsion2

def theoretical_torsion(coupled_strain, shear_modulus, shear_width, transverse_width, radius):
    torsion = coupled_strain * shear_modulus / (shear_width/2) * \
               ((shear_width**3*transverse_width + shear_width*transverse_width**3)/12 - np.pi*radius**4 / 2)

    return torsion
