import numpy as np
from scipy.io import wavfile


Sf = 177.0  # hydrophone calibration sensitivity


def read_audiofile(filename, Sf):
    Sf = 10 ** (Sf / 20)  # convert calibration from dB into ratio
    fs, P_raw = wavfile.read(filename)
    pressure = P_raw * Sf  # Sound pressure in uPa
    pressure = pressure / 10**6  # Convert sound pressure to Pa

    return pressure
