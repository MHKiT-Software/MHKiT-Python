import numpy as np
from scipy.io import wavfile


def read_audiofile(filename, Sf=177):
    """
    Read .wav file.

    Parameters
    ----------
    filename: string
        Input filename
    Sf: numeric
        Hydrophone calibration sensitivity in dB

    Returns
    -------
    out: numpy.array
        Sound pressure [Pa] indexed by time[s] and frequency [Hz]

    """
    Sf = 10 ** (Sf / 20)  # convert calibration from dB into ratio
    fs, P_raw = wavfile.read(filename)
    pressure = P_raw * Sf  # Sound pressure in uPa
    pressure = pressure / 10**6  # Convert sound pressure to Pa

    return pressure
