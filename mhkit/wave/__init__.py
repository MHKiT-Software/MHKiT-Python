import importlib

from mhkit.wave import resource
from mhkit.wave import graphics
from mhkit.wave import performance
from mhkit.wave import contours


def __getattr__(name):
    """
    Lazy load the wave.io submodule using PEP 562 module-level __getattr__.

    This defers importing heavy wave.io dependencies (rex, netCDF4, etc,) until
    they are actually accessed, improving import time for users who don't need
    all wave submodules, and avoiding import errors for users who have specified
    module level installs that need wave module functions, but not wave.io functions.
    """
    if name == "io":
        # This uses importlib.import_module() here, not "from mhkit.wave import io"
        # because when Python executes getattr(), it looks for 'io' as an attribute of
        # mhkit.wave. At this point in the module loading code 'io' doesn't exist yet and
        # Python calls __getattr__('io') again. This triggers the same "from" statement,
        # which calls __getattr__('io') again yielding a RecursionError.
        #
        # To fix this uses importlib.import_module("mhkit.wave.io") which loads the module directly
        # using the  absolute path without doing attribute lookup on the parent.
        #
        # The statement "from mhkit.wave import io" is equivalent to:
        #   io = getattr(mhkit.wave, 'io')
        #
        io = importlib.import_module("mhkit.wave.io")

        # Cache the module so subsequent accesses bypass __getattr__ entirely
        globals()[name] = io
        return io

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
