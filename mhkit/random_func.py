# import statements
import pandas as pd 
import numpy as np 

def calculate_TSR(rotor_speed,blade_length,inflow_speed):
    '''
    function used to calculate the tip speed ratio (TSR) of a MH device with rotor

    Parameters:
    ---------------
    rotor_speed : numpy array
        Rotor speed [rpm]
    blade_length : float/int
        Length of blade [m]
    inflow_speed : numpy array
        Velocity of inflow condition [m/s]

    Returns:
    -----------------
    TSR : numpy array
        Calculated tip speed ratio (TSR)
    '''

    try:
        rotor_speed = np.asarray(rotor_speed)
        inflow_speed = np.asarray(inflow_speed)
    except:
        pass

    # assertion statemnts



    # get rotational velocity in m/s
    rotor_velocity = rotor_speed / 60 * 2*np.pi*blade_length

    # calculate TSR
    TSR = rotor_velocity / inflow_speed

    return TSR

def calculate_Cp(power,velocity,blade_swept_area,rho):
    '''DESCRIPTION

    '''
    # check data types

    # calculat power in
    P_in = 0.5 * rho * blade_swept_area * velocity**3

    # calculate Cp ratio
    Cp = power / P_in 

    return Cp


def calculate_blade_moments(blade_matrix,flap_offset,flap_raw,edge_offset,edge_raw):
    '''Description

    '''
    # check data types

    # remove offset from raw signal
    flap_signal = flap_raw - flap_offset
    edge_signal = edge_raw - edge_offset

    # apply matrix to get load signals
    M_flap = blade_matrix(1,1)*flap_signal + blade_matrix(1,2)*edge_signal
    M_edge = blade_matrix(2,1)*flap_signal + blade_matrix(2,2)*edge_signal

    return M_flap, M_edge



