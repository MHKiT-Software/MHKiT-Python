from mhkit.wave import resource
from mhkit.wave import graphics
from mhkit.wave import performance
from mhkit.wave import contours


def __getattr__(name):
    """Lazy import for wave.io to avoid loading heavy dependencies unless needed."""
    if name == "io":
        from mhkit.wave import io as _io

        # Cache it in the module namespace to avoid repeated imports
        globals()[name] = _io
        return _io
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
