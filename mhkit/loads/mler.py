import mhkit.wave.resource as wave
import pandas as pd
import numpy as np
import scipy.interpolate


def readRAO(DOFread, RAO_File_Name, wave_freq):
    """
    Read in the RAO from the specified file and assign it to a dimension

    Parameters
    ----------
    DOFread : int
        1 - 3 (translational DOFs), 4 - 6 (rotational DOFs)
    RAO_File_Name : str
        Path to file. Format of the file must be
        Column 1: period in seconds, Column 2: response amplitude,
        Column 3: response phase
    wave_freq : numpy array
        Array of wave frequencies [Hz]

    Returns
    -------
    RAO : pd.DataFrame
        Response amplitude operator [m/m or rad/m] of chosen DOF indexed
        by frequency [Hz]
    """
    try:
        wave_freq = np.array(wave_freq)
    except:
        pass
    assert isinstance(DOFread, int), 'DOFread must be of type int'
    assert isinstance(RAO_File_Name, str), 'RAO_File_Name must be of type str'
    assert isinstance(
        wave_freq, np.ndarray), 'wave_freq must be of type np.ndarray'

    # - get total number of lines
    with open(RAO_File_Name, 'r') as f:
        for i, _ in enumerate(f.readlines()):
            pass
    nData = i+1
    # - get number of header lines
    dataStart = 0
    with open(RAO_File_Name, 'r') as f:
        for line in f:
            try:
                float(line.split()[0])
                break
            except:
                dataStart += 1
    # - preallocate and read data
    nData -= dataStart
    tmpRAO = np.zeros((nData, 3))
    with open(RAO_File_Name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[dataStart:]):
            tmpRAO[i, :] = [float(val) for val in line.split()]

    # convert from period in seconds to frequency in rad/s
    T = tmpRAO[:, 0]
    tmpRAO[:, 0] = 2*np.pi / T
    tmpRAO = tmpRAO[np.argsort(tmpRAO[:, 0])]  # sort by frequency

    # Add at w=0, amp=0, phase=0
    #### Questionable if the amplitude should be 1 or 0.  If set to 1, we count
    #### on the spectrum having nothing out there.  For dimensions
    #### other than in heave, this should be valid.  Heave should be
    #### set to 1. (ADP)
    if DOFread == 3:  # heave
        tmp = np.array([[0, 1, 0]])
    else:
        tmp = np.array([[0, 0, 0]])
    tmpRAO = np.concatenate((tmp, tmpRAO), axis=0)

    # convert freq to rad/s
    wave_freq = wave_freq*(2*np.pi)

    # Now interpolate to find the values
    Amp = scipy.interpolate.pchip_interpolate(
        tmpRAO[:, 0], tmpRAO[:, 1], wave_freq)
    Phase = scipy.interpolate.pchip_interpolate(
        tmpRAO[:, 0], tmpRAO[:, 2], wave_freq)

    # create the complex value to return
    _RAO = Amp * np.exp(1j*Phase)
    RAO = pd.DataFrame(data={'RAO': _RAO}, index=wave_freq/(2*np.pi))
    return RAO


def MLERcoeffsGen(RAO, wave_spectrum, response_desired):
    """
    This function calculates MLER (most likely extreme response)
    coefficients given a spectrum and RAO.

    Parameters
    ----------
    RAO : numpy array
        Response amplitude operator for a DOF
    wave_spectrum : pd.DataFrame
        Wave spectral density [m^2/Hz] indexed by frequency [Hz]
    response_desired : int or float
        Desired response, units should correspond to DOFtoCalc
        for a motion RAO or units of force for a force RAO

    Returns
    -------
    mler : pd.DataFrame
        DataFrame containing MLERcoeff [-], Conditioned wave spectrum [m^2-s], and Phase [rad]
        indexed by freq [Hz]
    """

    try:
        RAO = np.array(RAO)
    except:
        pass

    assert isinstance(RAO, np.ndarray), 'RAO must be of type np.ndarray'
    assert isinstance(
        wave_spectrum, pd.DataFrame), 'wave_spectrum must be of type pd.DataFrame'
    assert isinstance(response_desired, (int, float)
                      ), 'response_desired must be of type int or float'

    freq = wave_spectrum.index.values * (2*np.pi)  # convert from Hz to rad/s
    # change from Hz to rad/s
    wave_spectrum = wave_spectrum.iloc[:, 0].values / (2*np.pi)
    dw = (2*np.pi - 0.) / (len(freq)-1)  # get delta

    S_R = np.zeros(len(freq))  # [(response units)^2-s/rad]
    _S = np.zeros(len(freq))  # [m^2-s/rad]
    _A = np.zeros(len(freq))  # [m^2-s/rad]
    _CoeffA_Rn = np.zeros(len(freq))  # [1/(response units)]
    _phase = np.zeros(len(freq))

    # Note: waves.A is "S" in Quon2016; 'waves' naming convention matches WEC-Sim conventions (EWQ)
    # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3
    S_R[:] = np.abs(RAO)**2 * (2*wave_spectrum)

    # calculate spectral moments and other important spectral values.
    m0 = (wave.frequency_moment(pd.Series(S_R, index=freq), 0)).iloc[0, 0]
    m1 = (wave.frequency_moment(pd.Series(S_R, index=freq), 1)).iloc[0, 0]
    m2 = (wave.frequency_moment(pd.Series(S_R, index=freq), 2)).iloc[0, 0]
    wBar = m1 / m0

    # calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
    _CoeffA_Rn[:] = np.abs(RAO) * np.sqrt(2*wave_spectrum*dw) * ((m2 - freq*m1) + wBar*(freq*m0 - m1)) \
        / (m0*m2 - m1**2)  # Drummen version.  Dietz has negative of this.

    # save the new spectral info to pass out
    # Phase delay should be a positive number in this convention (AP)
    _phase[:] = -np.unwrap(np.angle(RAO))

    # for negative values of Amp, shift phase by pi and flip sign
    # for negative amplitudes, add a pi phase shift
    _phase[_CoeffA_Rn < 0] -= np.pi
    _CoeffA_Rn[_CoeffA_Rn < 0] *= -1    # then flip sign on negative Amplitudes

    # calculate the conditioned spectrum [m^2-s/rad]
    _S[:] = wave_spectrum * _CoeffA_Rn[:]**2 * response_desired**2
    _A[:] = 2*wave_spectrum * _CoeffA_Rn[:]**2 * \
        response_desired**2  # self.A == 2*self.S

    # if the response amplitude we ask for is negative, we will add
    # a pi phase shift to the phase information.  This is because
    # the sign of self.desiredRespAmp is lost in the squaring above.
    # Ordinarily this would be put into the final equation, but we
    # are shaping the wave information so that it is buried in the
    # new spectral information, S. (AP)
    if response_desired < 0:
        _phase += np.pi

    mler = pd.DataFrame(data={'MLERcoeff': _CoeffA_Rn,
                        'ResponseSpec': _S, 'Phase': _phase}, index=freq)
    mler = mler.fillna(0)
    return mler
