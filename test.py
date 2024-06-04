from mhkit import acoustics

P = acoustics.io.read_iclisten("examples/data/acoustics/RBW_6661_20240601_053114.wav")

spsd = acoustics.sound_pressure_spectral_density(P, P.fs)

spsdl = acoustics.sound_pressure_spectral_density_level(spsd)

spl = acoustics.sound_pressure_level(spsd, fmin=20, fmax=512000 // 2)

# spsd.plot()
print("here")
