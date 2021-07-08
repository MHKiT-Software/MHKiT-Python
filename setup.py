from setuptools import setup, find_packages
from distutils.core import Extension
import os
import re

DISTNAME = 'mhkit'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Marine and Hydrokinetic Toolkit'
AUTHOR = 'MHKiT developers'
MAINTAINER_EMAIL = ''
LICENSE = 'Revised BSD'
URL = 'https://github.com/MHKiT-Software/mhkit-python'
CLASSIFIERS=['Development Status :: 3 - Alpha',
             'Programming Language :: Python :: 3',
             'Topic :: Scientific/Engineering',
             'Intended Audience :: Science/Research',
             'Operating System :: OS Independent',
            ]
DEPENDENCIES = ['pandas>=1.0.0', 
                'numpy<1.21.0', 
                'scipy',
                'matplotlib', 
                'requests', 
                'pecos>=0.1.9',
                'fatpack',
                'lxml',
                'scikit-learn',
		        'NREL-rex>=0.2.61',
                'six>=1.13.0',
                'netCDF4', 
                'xarray']

# use README file as the long description
file_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(file_dir, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# get version from __init__.py
with open(os.path.join(file_dir, 'mhkit', '__init__.py')) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        VERSION = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")
        
setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      zip_safe=False,
      install_requires=DEPENDENCIES,
      scripts=[],
      include_package_data=True
  )
