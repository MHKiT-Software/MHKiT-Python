import pandas as pd
import numpy as np
import scipy.io as sio

def load_wecSim_output(file_name):
    """
    Loads the wecSim response class once it's been saved to a *.MAT structure 
    named 'output'. NOTE: Python is unable to import MATLAB objects. 
    MATLAB must be used to save the wecSim object as a structure. 
        
    Parameters
    ------------
    file_name: wecSim output *.mat file saved as a structure
        
        
    Returns
    ---------
    ws_output: pandas DataFrame indexed by time (s)
        
            
    """
    
    ws_data = sio.loadmat(file_name)
    output = ws_data['output']

    ######################################
    ## import wecSim wave class
    #         type: 'irregular'
    #         time: [30001×1 double]
    #    elevation: [30001×1 double]
    ######################################
    wave = output['wave']
    wave_type = wave[0][0][0][0][0][0]
    wave_time = wave[0][0]['time'][0][0].squeeze()#.reshape((len(wave_time), 1))
    wave_elevation = wave[0][0]['elevation'][0][0].squeeze() #.reshape((len(wave_time), 1))
    
    ######################################
    ## create wave output dataframe
    ######################################
    wave_output = pd.DataFrame(data = wave_time,columns=['time'])   
    wave_output = wave_output.set_index('time') 
    wave_output['elevation']=wave_elevation
    blah = {'wave_output' : wave_output}
    
    ######################################
    ## import wecSim body class
    #                       name: 'float'
    #                       time: [30001×1 double]
    #                   position: [30001×6 double]
    #                   velocity: [30001×6 double]
    #               acceleration: [30001×6 double]
    #                 forceTotal: [30001×6 double]
    #            forceExcitation: [30001×6 double]
    #      forceRadiationDamping: [30001×6 double]
    #             forceAddedMass: [30001×6 double]
    #             forceRestoring: [30001×6 double]
    #    forceMorrisonAndViscous: [30001×6 double]
    #         forceLinearDamping: [30001×6 double]
    ######################################    
    bodies = output['bodies']
    num_bodies = len(bodies[0][0]['name'][0])   # number of bodies
    bodies_time = []
    bodies_name = []
    bodies_position = []
    bodies_velocity = []
    bodies_acceleration = []
    bodies_forceTotal = []
    bodies_forceExcitation = []
    bodies_forceRadiationDamping = []
    bodies_forceAddedMass = []
    bodies_forceRestoring = []
    bodies_forceMorrisonAndViscous = []
    bodies_forceLinearDamping = []
    for body in range(num_bodies):
        bodies_name.append(bodies[0][0]['name'][0][body][0])
        bodies_time.append(bodies[0][0]['time'][0][body])
        bodies_position.append(bodies[0][0]['position'][0][body])
        bodies_velocity.append(bodies[0][0]['velocity'][0][body])
        bodies_acceleration.append(bodies[0][0]['acceleration'][0][body])
        bodies_forceTotal.append(bodies[0][0]['forceTotal'][0][body])
        bodies_forceExcitation.append(bodies[0][0]['forceExcitation'][0][body])
        bodies_forceRadiationDamping.append(bodies[0][0]['forceRadiationDamping'][0][body])
        bodies_forceAddedMass.append(bodies[0][0]['forceAddedMass'][0][body])
        bodies_forceRestoring.append(bodies[0][0]['forceRestoring'][0][body])
        bodies_forceMorrisonAndViscous.append(bodies[0][0]['forceMorrisonAndViscous'][0][body])
        bodies_forceLinearDamping.append(bodies[0][0]['forceLinearDamping'][0][body])    
        
    ######################################
    ## create body output dataframe
    ######################################    
    body_output = pd.DataFrame(data = bodies_time[0],columns=['time'])   
    body_output = body_output.set_index('time') 
    for body in range(num_bodies):
        for dof in range(6):
            body_output['body_'+str(body+1)+'_pos_'+str(dof+1)] = bodies_position[body][:,dof]
            body_output['body_'+str(body+1)+'_vel_'+str(dof+1)] = bodies_velocity[body][:,dof]
            body_output['body_'+str(body+1)+'_acc_'+str(dof+1)] = bodies_acceleration[body][:,dof]



    ######################################
    ## create wecSim output dataframe - OPTION 1
    ######################################
#    ws_output = pd.DataFrame(data = wave_time,columns=['time'])   
#    ws_output = ws_output.set_index('time') 
#    ws_output['elevation']=wave_elevation
#    
#    for body in range(num_bodies):
#        for dof in range(6):
#            ws_output['body_'+str(body+1)+'_pos_'+str(dof+1)] = bodies_position[body][:,dof]
#            ws_output['body_'+str(body+1)+'_vel_'+str(dof+1)] = bodies_velocity[body][:,dof]
#            ws_output['body_'+str(body+1)+'_acc_'+str(dof+1)] = bodies_acceleration[body][:,dof]

    ######################################
    ## create wecSim output dataframe - OPTION 2 with Dict
    ######################################
    ws_output = {'wave' : {'wave_output' : wave_output}}

#    ws_output = {'wave' : 'wave_output','body': 'body_output'}

    return ws_output