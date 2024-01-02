from mhkit.wave.io.hindcast import wind_toolkit

try:
    from mhkit.wave.io.hindcast import hindcast
except ImportError:
    print(
        "WARNING: Wave WPTO hindcast functions not imported from"
        "MHKiT-Python. If you are using Windows and calling from"
        "MHKiT-MATLAB this is expected."
    )
    pass
