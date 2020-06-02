import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy import signal
from scipy import fft, fftpack
from scipy.signal import hilbert


#This group of functions are to be used for power quality assessments 

def electrical_angle(frequency,um,sample_rate):
    """
    Calculates a time series of electrical angle of the fundamental of the measured voltage. 
    from IEC 62600-30 Eq. 3. 
    Note: this function could use some improvements in efficency and tests for accuracy
    Note: right now this function can only handle 1 column of data at a time

    Parameters
    ------------
    frequency: pandas DataFrame
        time varying freqiency (Hz) index by time time (s or datetime)
    um: pandas DataFrame
        measured voltage source (V) index by time 
    sample_rate: float
        frequency of the timeseries data [Hz]
        
    Returns
    ---------
    alpha: pandas DataFrame 
        electrical angle index by time (s)
    """
    v0=um.values[0]
    if v0 < 0:
        x = np.where(um.values > 0)[0][0]
        #print(um.values[x-2])
        dt = (x+1)*(1/sample_rate)
        alpha0 = frequency.values[x]*360*dt
    if v0 ==0:
        alpha0 = 0.0
        
    if v0 > 0: 
        x = np.where(um.values < 0)[0][0]
        #print(um.values[x-2])
        dt = (x+1)*(1/sample_rate)
        alpha0 = frequency.values[x]*360*dt
    
    alpha=np.zeros(len(frequency.values))
    i = 1
    j=0
    t = frequency.index.values
    print(alpha0)
    for time in t[1:]:
        #print(time)
        #x = integrate.trapz(frequency.values[0:i],dx=sample_rate,axis = 0)
        #print(x)
        alpha[i] =2*np.pi*(integrate.trapz(frequency.values[0:i],dx=j,axis = 0))+ np.radians(alpha0)  
        #NOTE: this may not be correct, we may want to look into this equation more 
        #alpha[i] = 2*np.pi*np.sum(frequency.values[0:i])+alpha0
        #print(alpha[time])
        #alpha.index=frequency.index
        i = i+1
        j=j+(1/sample_rate)
    alpha= pd.DataFrame(alpha,index = um.index[1:])
    return alpha
    

def apparent_power_fict(sc_ratio,Sr):
    """
    Calculates the three-phase short-circuit apprant power of a fictitious grid according to 
    IEC standard 62600-30. 
     

    Parameters
    ------------
    sc_ratio: intiger
        short-circuit ratio, must be between 20 and 50
        
    Sr: float
        rated apparent power for the MEC (VA)
        
    Returns
    ---------
    Sk_fic: float 
        three phase short-circuit apparent power of fictitious grid (VA)
    """

    return sc_ratio*Sr


def calc_resitance_inductance(sc_ratio,Sr,Un,freq):
    """
    Calculates the short-circuit resistance and inductance
    IEC standard 62600-30 Eq 5 and 4. 
     

    Parameters
    ------------
    sc_ratio: intiger
        short-circuit ratio, must be between 20 and 50
        
    Sr: float
        rated apparent power for the MEC (VA)
    
    Un: float 
        Nominal RMS voltage (V) of the grid

    freq: intiger
        nominal grid freqency (Hz). 50 or 60 Hz, typically
        
    Returns
    ---------
    R_fic: Pandas DataFrame 
        fictitious grid resistance (Ohms) with impedance phase angle as index
    L_fic: Pandas DataFrame
        fictitious grid indictance (H) with impedance phase angle as index
    """

    angle=np.array([30,50,70,85]) #impedance phase angles in degrees
    xfic = np.ones(np.size(angle))
    Skfic=apparent_power_fict(sc_ratio,Sr)
    c=((Un**2/Skfic)**2)*np.tan(np.radians(angle))
    #need to solve the quadratic formula
    d=(np.tan(np.radians(angle))**2)-4*1*(-1*c)
    sol1=(-np.tan(np.radians(angle))-np.sqrt(d))/(-2)
    sol2=(-np.tan(np.radians(angle))+np.sqrt(d))/(-2)
    i = 0
    for s in sol1:
        
        if sol1[i] >= 0: 
            xfic[i]=sol1[i]
        elif sol2[i] >=0:
            xfic[i]=sol2[i]
        else: 
            print('No valid Solution Found')
            return
        i = i+1
    
    L_fic=xfic/(2*np.pi*freq)
    R_fic=xfic/np.tan(np.radians(angle))
    L_fic=pd.DataFrame(L_fic,index=angle)
    R_fic=pd.DataFrame(R_fic,index=angle)
    
    return R_fic, L_fic
    
def ideal_voltage(Un,alpha):
    """
    Calculates the time series of ideal voltage
    IEC standard 62600-30 Eq 2. 
     

    Parameters
    -----------
    Un: float 
        Nominal RMS voltage (V) of the grid

    alpha: pandas DataFrame 
        electrical angle index by time (s)
        
    Returns
    ---------
    uo: pandas DataFrame
        ideal voltage source (V) index by time 
    """

    uo=pd.DataFrame()
    uo=np.sqrt(2/3)*Un*np.sin(alpha)
    uo.index=alpha.index
    return uo

def simulated_voltage(uo,R_fic,L_fic,im):
    """
    Calculates the time series of simulated voltage
    IEC standard 62600-30 Eq 1. 
     
    Parameters
    -----------
    uo: pandas DataFrame
        ideal voltage source (V) index by time 

    R_fic: Pandas DataFrame 
        fictitious grid resistance (Ohms) with impedance phase angle as index

    L_fic: Pandas DataFrame
        fictitious grid indictance (H) with impedance phase angle as index

    im: pandas DataFrame
        measured instantaneous current (A) index by time 
        
    Returns
    ---------
    ufic: pandas DataFrame
        simulated voltage source (V) index by time 
    """

    #ufic=pd.DataFrame()
    ufic=np.ones((np.size(uo.values),np.size(L_fic.values)))
    dt=pd.Series(uo.index).diff()
    #dt=uo.index[2]-uo.index[1]
    i = 0
    k = 0
    print(R_fic)
    for t in uo.values[1:]:
        #print(ufic[i,:])
        for j in R_fic.values: 
            #print(j)
            #x=uo.values[i]+R_fic.values[k]*im.values[i]+L_fic.values[k]*((im.values[i]-im.values[i-1])/dt[i])
            #print(x)
            ufic[i,k]=uo.values[i]+R_fic.values[k]*im.values[i]+L_fic.values[k]*((im.values[i]-im.values[i-1])/dt[i])
            k=k+1
        i = i+1
        k=0
    ufic=pd.DataFrame(ufic,index=uo.index)
    return ufic 


    

def short_term_flicker_severity(P): ### Should not need this function- it should be the output of the flickermeter 
    """
    Calcultes the short term flicker severity based on a 10 min observation period
    
     
    Parameters
    -----------
    P: pandas DataFrame
        measured flicker levels exceeded by the index value of % of the measurment time 
        
    Returns
    ---------
    Pst: Float
        Short Term flicker severity  
    """  

    P1=P.loc['0.1']
    P1s=(P.loc['0.7']+P.loc['1']+P.loc['1.5'])/3
    P3s=(P.loc['2.2'],P.loc['3']+P.loc['4'])/3
    P10s=(P.loc['6']+P.loc['8']+P.loc['10']+P.loc['13']+P.loc['17'])/5
    P50s=(P.loc['30']+P.loc['50']+P.loc['80'])/3

    return np.sqrt(0.0314*P1+0.0525*P1s+0.0657*P3s+0.28*P10s+0.08*P50s)

def flicker_coefficient(Pst_fic,Sr,Sk_fic): 
    """
    Calcultes the flicker coefficient for continuous operation based 
    on IEC 62600-30 
    
     
    Parameters
    -----------
    Pst_fic: float
        flicker emission from the MEC on a fictitious grid
    NOTE: One of these has to be a data frame with the angles!!!
    Sr: float
        the rated apparnet power of the MEC (VA)

    Sk_fic: float
        the short circuit apparent power of a fictitious grid (VA) 
        
    Returns
    ---------
    c: Pandas Dataframe
        flicker coefficient for continuious operation indexed by network impedance pahse angles
          
    """  

    angle=np.array([30,50,70,85]) #impedance phase angles in degrees that should be reported for 

    flicker_coefficient = Pst_fic*(Sk_fic/Sr)

    pass

def harmonics(x,freq):
    """
    Calculates the harmonics from time series of voltage or current based on IEC 61000-4-7. 

    Parameters
    -----------
    x: pandas Series of DataFrame
        timeseries of voltage or current
    
    freq: float
        frequency of the timeseries data [Hz]
    
    Returns
    --------
    harmonics: float? Array? 
        harmonics of the timeseries data
    """
    
    x.to_numpy()
    
    a = np.fft.fft(x,axis=0)
    
    amp = np.abs(a) # amplitude of the harmonics
    #print(len(amp))
    freqfft = fftpack.fftfreq(len(x),d=1./freq)
    ##### NOTE: Harmonic order is fft freq/ number of fundamentals in the sample time period, look int if this is what I should be using
    ### Note: do I need to impliment the digital low pass filter equations ???

    harmonics = pd.DataFrame(amp,index=freqfft)
    
    harmonics=harmonics.sort_index(axis=0)
    #print(harmonics)
    hz = np.arange(0,3000,5)
    
    ind=pd.Index(harmonics.index)
    indn = [None]*np.size(hz)
    i = 0
    for n in hz:
        indn[i] = ind.get_loc(n, method='nearest')
        i = i+1
    
    harmonics = harmonics.iloc[indn]
    
    return harmonics


def harmonic_subgroups(harmonics, frequency): 
    """
    calculates the harmonic subgroups based on IEC 61000-4-7

    Parameters
    ----------
    harmonics: pandas Series or DataFrame 
        RMS harmonic amplitude indexed by the harmonic frequency 
    frequency: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    --------
    harmonic_subgroups: array? Pandas?
        harmonic subgroups 
    """
    #assert isinstance(frequency, {60,50]), 'Frequency must be either 60 or 50'

    #def subgroup(h,ind):
        

    if frequency == 60:
        
        hz = np.arange(1,3000,60)
    elif frequency == 50: 
        
        hz = np.arange(1,2500,50)
    else:
        print("Not a valid frequency")
        pass
    
    j=0
    i=0
    cols=harmonics.columns
    #harmonic_subgroups=[None]*np.size(hz)
    harmonic_subgroups=np.ones((np.size(hz),np.size(cols)))
    for n in hz:

        harmonics=harmonics.sort_index(axis=0)
        ind=pd.Index(harmonics.index)
        
        indn = ind.get_loc(n, method='nearest')
        for col in cols:
            harmonic_subgroups[i,j] = np.sqrt(np.sum([harmonics[col].iloc[indn-1]**2,harmonics[col].iloc[indn]**2,harmonics[col].iloc[indn+1]**2]))
            j=j+1
        j=0
        i=i+1
        #print(harmonic_subgroups)
    
    harmonic_subgroups = pd.DataFrame(harmonic_subgroups,index=hz)

    return harmonic_subgroups

def total_harmonic_current_distortion(harmonics_subgroup,rated_current):    #### might want to rename without current since this can be done for voltage too

    """
    Calculates the total harmonic current distortion (THC) based on IEC 62600-30

    Parameters
    ----------
    harmonics_subgroup: pandas DataFrame or Series
        the subgrouped RMS current harmonics indexed by harmonic order
    
    rated_current: float
        the rated current of the energy device in Amps
    
    Returns
    --------
    THC: float
        the total harmonic current distortion 
    """
    #print(harmonics_subgroup)
    harmonics_sq = harmonics_subgroup.iloc[2:50]**2

    harmonics_sum=harmonics_sq.sum()

    THC = (np.sqrt(harmonics_sum)/harmonics_subgroup.iloc[1])*100

    return THC

def interharmonics(harmonics,frequency):
    """
    calculates the interharmonics ffrom the harmonics of current

    Parameters
    -----------
    harmonics: pandas Series or DataFrame 
        RMS harmonic amplitude indexed by the harmonic frequency 

    frequency: int
        value indicating if the power supply is 50 or 60 Hz. Valid input are 50 and 60

    Returns
    -------
    interharmonics: pandas DataFrame
        interharmonics groups
    """
    #Note: work on the data types, df, Series, numpy to streamline this. Will I ever pass multiple columns of harmonics??
    #assert isinstance(frequency, {60,50]), 'Frequency must be either 60 or 50'

    if frequency == 60:
        
        hz = np.arange(0,3000,60)
    elif frequency == 50: 
        
        hz = np.arange(0,2500,50)
    else:
        print("Not a valid frequency")
        pass
    j=0
    i=0
    cols=harmonics.columns
    interharmonics=np.ones((np.size(hz),np.size(cols)))
    for n in hz: 
        harmonics=harmonics.sort_index(axis=0)
        ind=pd.Index(harmonics.index)
        
        indn = ind.get_loc(n, method='nearest')  
        if frequency == 60:
            subset = harmonics.iloc[indn+1:indn+11]**2
            subset = subset.squeeze()
        else: 
            subset = harmonics.iloc[indn+1:indn+7]**2
            subset = subset.squeeze()
        for col in cols:
            interharmonics[i,j] = np.sqrt(np.sum(subset))
            j=j+1
        j=0
        i=i+1
    
    
    interharmonics = pd.DataFrame(interharmonics,index=hz)

    return interharmonics
