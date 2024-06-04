from mhkit import acoustics

P = acoustics.io.read_iclisten("examples/data/acoustics/RBW_6661_20240601_053114.wav")

# spsd = acoustics.sound_pressure_spectral_density(P, P.fs)

# spsd.plot()
print("here")
