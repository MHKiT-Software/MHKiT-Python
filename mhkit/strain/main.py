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


def calculate_primary_strains(ec, eb, ea, gauge_type):
    # ec, eb, ea = rosette strains
    if gauge_type == 90:
        # Assumes ea and ec are oriented in an axial directions
        axial_strain_x = ec
        axial_strain_y = ea
        shear_strain = eb - (ea + ec) / 2
    elif gauge_type == 120:
        # Assumes eb is oriented in an axial direction
        axial_strain_x = 2 / 3 * (ea - eb / 2 + ec)
        axial_strain_y = eb
        shear_strain = 1 / np.sqrt(3) * (ea - ec)
    else:
        raise ValueError(f"gauge_type must be 90 or 120. Got: {gauge_type}")

    return axial_strain_x, axial_strain_y, shear_strain


def calculate_loads(
    rosette1,
    rosette2,
    elastic_modulus,
    shear_modulus,
    shear_width,
    transverse_width,
    radius,
):
    normal = (
        0.5
        * (rosette1["axial_strain_x"] + rosette2["axial_strain_x"])
        * elastic_modulus
        * (shear_width * transverse_width - np.pi * radius**2)
    )
    moment = (
        (rosette1["axial_strain_x"] - rosette2["axial_strain_x"])
        / shear_width
        * elastic_modulus
        * (transverse_width * shear_width**3 / 12 - np.pi * radius**4 / 4)
    )
    torsion1 = calculate_torsion(
        rosette1["shear_strain"], shear_modulus, shear_width, transverse_width, radius
    )
    torsion2 = calculate_torsion(
        rosette2["shear_strain"], shear_modulus, shear_width, transverse_width, radius
    )

    return normal, moment, torsion1, torsion2


def calculate_torsion(
    shear_strain, shear_modulus, shear_width, transverse_width, radius
):
    geometry_factor = (
        (shear_width**3 * transverse_width + shear_width * transverse_width**3) / 12
        - np.pi * radius**4 / 2
    ) / (shear_width / 2)
    torsion = shear_strain * shear_modulus * geometry_factor

    return torsion
