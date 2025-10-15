from mhkit.wave import resource
from mhkit.wave import graphics
from mhkit.wave import performance
from mhkit.wave import contours


def __getattr__(name):
    """Lazy import for wave.io to avoid loading heavy dependencies unless needed."""
    if name == "io":
        from mhkit.wave import io

        return io
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
