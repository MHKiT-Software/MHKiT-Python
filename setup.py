import os
import re
from setuptools import setup, find_packages

DISTNAME = "mhkit"
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = "Marine and Hydrokinetic Toolkit"
AUTHOR = "MHKiT developers"
MAINTAINER_EMAIL = ""
LICENSE = "Revised BSD"
URL = "https://github.com/MHKiT-Software/mhkit-python"
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
]
DEPENDENCIES = [
    "pandas>=1.0.0",
    "numpy>=1.21.0",
    "scipy",
    "matplotlib",
    "requests",
    "pecos>=0.3.0",
    "fatpack",
    "lxml",
    "scikit-learn",
    "NREL-rex>=0.2.63",
    "six>=1.13.0",
    "h5py>=3.6.0",
    "h5pyd >=0.7.0",
    "netCDF4",
    "xarray",
    "statsmodels",
    "pytz",
    "bottleneck",
    "beautifulsoup4",
]

LONG_DESCRIPTION = """
MHKiT-Python is a Python package designed for marine renewable energy applications to assist in 
data processing and visualization.  The software package includes functionality for:

* Data processing
* Data visualization
* Data quality control
* Resource assessment
* Device performance
* Device loads

Documentation
------------------
MHKiT-Python documentation includes overview information, installation instructions, API documentation, and examples.
See the [MHKiT documentation](https://mhkit-software.github.io/MHKiT) for more information.

Installation
------------------------
MHKiT-Python requires Python (3.7, 3.8, or 3.9) along with several Python 
package dependencies.  MHKiT-Python can be installed from PyPI using the command ``pip install mhkit``.
See [installation instructions](https://mhkit-software.github.io/MHKiT/installation.html) for more information.

Copyright and license
------------------------
MHKiT-Python is copyright through the National Renewable Energy Laboratory, 
Pacific Northwest National Laboratory, and Sandia National Laboratories. 
The software is distributed under the Revised BSD License.
See [copyright and license](LICENSE.md) for more information.
"""


# get version from __init__.py
file_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(file_dir, "mhkit", "__init__.py")) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        VERSION = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name=DISTNAME,
    version=VERSION,
    packages=PACKAGES,
    ext_modules=EXTENSIONS,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url=URL,
    classifiers=CLASSIFIERS,
    zip_safe=False,
    install_requires=DEPENDENCIES,
    scripts=[],
    include_package_data=True,
)
