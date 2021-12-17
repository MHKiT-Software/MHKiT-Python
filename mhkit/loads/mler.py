import mhkit.wave.resource as wave
import pandas as pd
import numpy as np
import scipy.interpolate


def readRAO(DOFread,RAO_File_Name,wave_freq):
    """ Read in the RAO from the specified file and assign it to a dimension
    DOFread : 1 - 3 (translational DOFs)
                4 - 6 (rotational DOFs)
    Sets: self._RAO, self._RAOdataReadIn[DOFread], self._RAOdataFileName[DOFread]
    """
    # Format of file to read in:
    # Column 1:    period in seconds
    # Column 2:    response amplitude (m/m for DOF 1-3; or radians/m for DOF 4-6)
    # Column 3:    response phase (radians)
    #

    # - get total number of lines
    with open(RAO_File_Name,'r') as f:
        for i,_ in enumerate(f.readlines()): pass
    nData = i+1
    # - get number of header lines
    dataStart = 0
    with open(RAO_File_Name,'r') as f:
        for line in f:
            try:
                float(line.split()[0])
                break
            except:
                dataStart += 1
    # - preallocate and read data
    nData -= dataStart
    tmpRAO = np.zeros((nData,3))
    with open(RAO_File_Name,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines[dataStart:]):
            tmpRAO[i,:] = [ float(val) for val in line.split() ]

    # convert from period in seconds to frequency in rad/s
    T = tmpRAO[:,0]
    tmpRAO[:,0] = 2*np.pi / T
    tmpRAO = tmpRAO[ np.argsort(tmpRAO[:,0]) ]  # sort by frequency

    # Add at w=0, amp=0, phase=0
    #### Questionable if the amplitude should be 1 or 0.  If set to 1, we count
    #### on the spectrum having nothing out there.  For dimensions
    #### other than in heave, this should be valid.  Heave should be
    #### set to 1. (ADP)
    if DOFread == 3: #heave
        tmp = np.array( [[0,1,0]] )
    else:
        tmp = np.array( [[0,0,0]] )
    tmpRAO = np.concatenate( (tmp,tmpRAO), axis=0 )   

    # Now interpolate to find the values
    Amp   = scipy.interpolate.pchip_interpolate( tmpRAO[:,0], tmpRAO[:,1], wave_freq )
    Phase = scipy.interpolate.pchip_interpolate( tmpRAO[:,0], tmpRAO[:,2], wave_freq )

    # create the complex value to return
    RAO = Amp * np.exp(1j*Phase)
    return RAO

# inputs
RAO_File_Name = 'RAO_heave_RM3float.dat'
DOFread = 3
numFreq = 500
wave_freq = np.linspace( 0.,1,numFreq)
RAO = readRAO(DOFread,RAO_File_Name, wave_freq*(2*np.pi) )

# inputs
DOFtoCalc = 3
response_desired = 1
Hs = 9.0
Tp = 15.1
ws = wave.pierson_moskowitz_spectrum(wave_freq,Tp,Hs)
wave_spectrum = ws / (2*np.pi) # change from Hz to rad/s

# DONE: compare RAO results - match
# DONE: compare spectrum results - results are the same after converting to radians
# TODO: compare coefficient results
# TODO: clean up docstrings
# TODO: use read_csv for readRAO
# TODO: consistency in return formatting (pandas)

# def MLERcoeffsGen(self,DOFtoCalc,response_desired=None,safety_factor=None):
""" This function calculates MLER (most likely extreme response)
coefficients given a spectrum and RAO

DOFtoCalc: 1 - 3 (translational DOFs)
            4 - 6 (rotational DOFs)
response_desired: desired response, units should correspond to DOFtoCalc
    for a motion RAO or units of force for a force RAO
safety_factor: alternative to specifying response_desired;
    non-dimensional scaling factor applied to half the significant wave
    height

Sets self._S, self._A, self._CoeffA_Rn, self._phase
Sets self._Spect containing spectral information
"""
DOFtoCalc -= 1 # convert to zero-based indices (EWQ)

dw = (2*np.pi - 0.) / (numFreq-1)

# check that we specified a response
# if (response_desired is None) and (safety_factor is None):
#     raise ValueError('Specify response_desired or safety_factor.')
# elif safety_factor:
#     #RAO_Tp = np.interp(2.0*np.pi/self.waves.T,self.waves._w,np.abs(self._RAO[:,DOFtoCalc]))
#     RAO_Tp = scipy.interpolate.pchip_interpolate(self.waves._w,
#                                                     np.abs(self._RAO[:,DOFtoCalc]),
#                                                     2.0*np.pi/self.waves.T)
#     response_desired = np.abs(RAO_Tp) * safety_factor*self.waves.H/2
#     print('Target wave elevation         :',safety_factor*self.waves.H/2)
#     print('Interpolated RAO(Tp)          :',RAO_Tp)
#     print('Desired response (calculated) :',response_desired)
desiredRespAmp = response_desired

S_R             = np.zeros(numFreq)  # [(response units)^2-s/rad]
_S         = np.zeros(numFreq)  # [m^2-s/rad]
_A         = np.zeros(numFreq)  # [m^2-s/rad]
_CoeffA_Rn = np.zeros(numFreq)  # [1/(response units)]
_phase     = np.zeros(numFreq)

# calculate the RAO times sqrt of spectrum
# note that we could define:  a_n=(waves.A*waves.dw).^0.5; (AP)
#S_tmp(:)=squeeze(abs(obj.RAO(:,DOFtoCalc))).*2 .* obj.waves.A;          % Response spectrum.
# note: self.A == 2*self.S  (EWQ)
#   i.e. S_tmp is 4 * RAO * calculatedeWaveSpectrum
#S_tmp[:] = 2.0*np.abs(self._RAO[:,DOFtoCalc])*self.waves._A     # Response spectrum.

# Note: waves.A is "S" in Quon2016; 'waves' naming convention matches WEC-Sim conventions (EWQ)
S_R[:] = np.abs(RAO)**2 * (2*wave_spectrum.iloc[:,0])  # Response spectrum [(response units)^2-s/rad] -- Quon2016 Eqn. 3 

# calculate spectral moments and other important spectral values.
#self._Spect = spectrum.stats( S_R, self.waves._w, self.waves._dw )
m0 = (wave.frequency_moment(pd.Series(S_R,index=wave_freq*(2*np.pi)),0)).iloc[0,0]
m1 = (wave.frequency_moment(pd.Series(S_R,index=wave_freq*(2*np.pi)),1)).iloc[0,0]
m2 = (wave.frequency_moment(pd.Series(S_R,index=wave_freq*(2*np.pi)),2)).iloc[0,0]

wBar = m1 / m0
# calculate coefficient A_{R,n} [(response units)^-1] -- Quon2016 Eqn. 8
_CoeffA_Rn[:] = np.abs(RAO) * np.sqrt(2*wave_spectrum.iloc[:,0]*dw) * ( (m2 - wave_freq*m1) + wBar*(wave_freq*m0 - m1) ) \
        / (m0*m2 - m1**2) # Drummen version.  Dietz has negative of this.

# !!!! review above equation again; wave_freq may need adjustment

# self._CoeffA_Rn[:] = np.abs(self._RAO[:,DOFtoCalc]) * np.sqrt(self.waves._A*self.waves._dw) \
#         * ( (self._Spect.M2 - self.waves._w*self._Spect.M1) \
#             + self._Spect.wBar*(self.waves._w*self._Spect.M0 - self._Spect.M1) ) \
#         / (self._Spect.M0*self._Spect.M2 - self._Spect.M1**2) # Drummen version.  Dietz has negative of this.

# save the new spectral info to pass out
_phase[:] = -np.unwrap( np.angle(RAO) ) # Phase delay should be a positive number in this convention (AP)

# for negative values of Amp, shift phase by pi and flip sign
_phase[_CoeffA_Rn < 0] -= np.pi # for negative amplitudes, add a pi phase shift
_CoeffA_Rn[_CoeffA_Rn < 0] *= -1    # then flip sign on negative Amplitudes

# calculate the conditioned spectrum [m^2-s/rad]
_S[:] = wave_spectrum.iloc[:,0] * _CoeffA_Rn[:]**2 * desiredRespAmp**2
_A[:] = 2*wave_spectrum.iloc[:,0] * _CoeffA_Rn[:]**2 * desiredRespAmp**2 # self.A == 2*self.S

# if the response amplitude we ask for is negative, we will add
# a pi phase shift to the phase information.  This is because
# the sign of self.desiredRespAmp is lost in the squaring above.
# Ordinarily this would be put into the final equation, but we
# are shaping the wave information so that it is buried in the
# new spectral information, S. (AP)
if desiredRespAmp < 0:
    _phase += np.pi




print('done')