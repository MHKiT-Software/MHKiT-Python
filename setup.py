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
    "numpy>=2.0.0",
    "pandas>=2.2.2",
    "scipy>=1.14.0",
    "xarray>=2024.6.0",
    "matplotlib>=3.9.1",
    "scikit-learn>=1.5.1",
    "h5py>=3.11.0",
    "h5pyd>=0.18.0",
    "netCDF4>=1.7.1.post1",
    "statsmodels>=0.14.2",
    "requests",
    "pecos>=0.3.0",
    "fatpack",
    "NREL-rex>=0.2.63",
    "pytz",
    "beautifulsoup4",
    "numexpr>=2.10.0",
    "lxml",
    "bottleneck",
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
MHKiT-Python requires Python (3.10, or 3.11) along with several Python 
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
