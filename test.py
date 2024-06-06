import matplotlib.pyplot as plt
from mhkit import acoustics

# P = acoustics.io.read_iclisten("examples/data/acoustics/RBW_6661_20240601_053114.wav")
# P = acoustics.io.read_hydrophone("examples/data/acoustics/RBW_6661_20240601_053114.wav", peak_V=3, Sf=-177)
P = acoustics.io.read_soundtrap("examples/data/acoustics/6247.230204150508.wav", Sf=-177)
# P = acoustics.io.read_hydrophone("examples/data/acoustics/6247.230204150508.wav", peak_V=1, Sf=-177)

spsd = acoustics.sound_pressure_spectral_density(P, P.fs)

spsdl = acoustics.sound_pressure_spectral_density_level(spsd)

spl = acoustics.sound_pressure_level(spsd, fmin=20, fmax=96000 // 2)

spsdl.T.plot()
plt.show()
print("here")
