import pandas as pd
import numpy as np
import scipy.io as sio

def read_wecSim(file_name):
    """
    Loads the wecSim response class once 'output' has been saved to a *.mat 
    structure. 
    
    NOTE: Python is unable to import MATLAB objects. 
    MATLAB must be used to save the wecSim object as a structure. 
        
    Parameters
    ------------
    file_name: string
        Name of wecSim output file saved as a *.mat structure
        
        
    Returns
    ---------
    ws_output: dict 
        Dictionary of pandas DataFrames, indexed by time (s)      
              
    """
    
    ws_data = sio.loadmat(file_name)
    output = ws_data['output']

    ######################################
    ## import wecSim wave class
    #         type: ''
    #         time: [iterations x 1 double]
    #    elevation: [iterations x 1 double]
    ######################################
    try:              
        wave = output['wave']
        wave_type = wave[0][0][0][0][0][0]  
        time = wave[0][0]['time'][0][0].squeeze()
        elevation = wave[0][0]['elevation'][0][0].squeeze()
        
        ######################################
        ## create wave_output DataFrame
        ######################################
        wave_output = pd.DataFrame(data = time,columns=['time'])   
        wave_output = wave_output.set_index('time') 
        wave_output['elevation'] = elevation        
        wave_output.name = wave_type
    
    except:
        print("wave class not used") 
        wave_output = []      
    
    
    ######################################
    ## import wecSim body class
    #                       name: ''
    #                       time: [iterations x 1 double]
    #                   position: [iterations x 6 double]
    #                   velocity: [iterations x 6 double]
    #               acceleration: [iterations x 6 double]
    #                 forceTotal: [iterations x 6 double]
    #            forceExcitation: [iterations x 6 double]
    #      forceRadiationDamping: [iterations x 6 double]
    #             forceAddedMass: [iterations x 6 double]
    #             forceRestoring: [iterations x 6 double]
    #    forceMorrisonAndViscous: [iterations x 6 double]
    #         forceLinearDamping: [iterations x 6 double]
    ######################################    
    try:
        bodies = output['bodies']
        num_bodies = len(bodies[0][0]['name'][0])  
        name = []   
        time = []
        position = []
        velocity = []
        acceleration = []
        forceTotal = []
        forceExcitation = []
        forceRadiationDamping = []
        forceAddedMass = []
        forceRestoring = []
        forceMorrisonAndViscous = []
        forceLinearDamping = []
        for body in range(num_bodies):
            name.append(bodies[0][0]['name'][0][body][0])   
            time.append(bodies[0][0]['time'][0][body])
            position.append(bodies[0][0]['position'][0][body])
            velocity.append(bodies[0][0]['velocity'][0][body])
            acceleration.append(bodies[0][0]['acceleration'][0][body])
            forceTotal.append(bodies[0][0]['forceTotal'][0][body])
            forceExcitation.append(bodies[0][0]['forceExcitation'][0][body])
            forceRadiationDamping.append(bodies[0][0]['forceRadiationDamping'][0][body])
            forceAddedMass.append(bodies[0][0]['forceAddedMass'][0][body])
            forceRestoring.append(bodies[0][0]['forceRestoring'][0][body])
            forceMorrisonAndViscous.append(bodies[0][0]['forceMorrisonAndViscous'][0][body])
            forceLinearDamping.append(bodies[0][0]['forceLinearDamping'][0][body])    
    except:
        num_bodies = 0         
        
    ######################################
    ## create body_output DataFrame
    ######################################            
    if num_bodies == 1:
        body_output = pd.DataFrame(data = time[0],columns=['time'])   
        body_output = body_output.set_index('time') 
        body_output.name = name
        for body in range(num_bodies):
            for dof in range(6):
                body_output[f'position_dof{dof+1}'] = position[body][:,dof]
                body_output[f'velocity_dof{dof+1}'] = velocity[body][:,dof]
                body_output[f'acceleration_dof{dof+1}'] = acceleration[body][:,dof]            
                body_output[f'forceTotal_dof{dof+1}'] = forceTotal[body][:,dof]
                body_output[f'forceExcitation_dof{dof+1}'] = forceExcitation[body][:,dof]
                body_output[f'forceRadiationDamping_dof{dof+1}'] = forceRadiationDamping[body][:,dof]
                body_output[f'forceAddedMass_dof{dof+1}'] = forceAddedMass[body][:,dof]
                body_output[f'forceRestoring_dof{dof+1}'] = forceRestoring[body][:,dof]
                body_output[f'forceMorrisonAndViscous_dof{dof+1}'] = forceMorrisonAndViscous[body][:,dof]
                body_output[f'forceLinearDamping_dof{dof+1}'] = forceLinearDamping[body][:,dof]              
    elif num_bodies > 1:
        body_num_output = {}          
        for body in range(num_bodies):
            tmp2 = pd.DataFrame(data = time[0],columns=['time'])   
            tmp2 = tmp2.set_index('time') 
            tmp2.name = name[body]
            for dof in range(6):                
                tmp2[f'position_dof{dof+1}'] = position[body][:,dof]
                tmp2[f'velocity_dof{dof+1}'] = velocity[body][:,dof]
                tmp2[f'acceleration_dof{dof+1}'] = acceleration[body][:,dof]            
                tmp2[f'forceTotal_dof{dof+1}'] = forceTotal[body][:,dof]
                tmp2[f'forceExcitation_dof{dof+1}'] = forceExcitation[body][:,dof]
                tmp2[f'forceRadiationDamping_dof{dof+1}'] = forceRadiationDamping[body][:,dof]
                tmp2[f'forceAddedMass_dof{dof+1}'] = forceAddedMass[body][:,dof]
                tmp2[f'forceRestoring_dof{dof+1}'] = forceRestoring[body][:,dof]
                tmp2[f'forceMorrisonAndViscous_dof{dof+1}'] = forceMorrisonAndViscous[body][:,dof]
                tmp2[f'forceLinearDamping_dof{dof+1}'] = forceLinearDamping[body][:,dof]                            
            body_num_output[f'body{body+1}'] = tmp2            
        body_output = body_num_output.copy()               
    else:
        print("body class not used") 
        body_output = []    


    ######################################
    ## import wecSim pto class
    #                      name: ''
    #                      time: [iterations x 1 double]
    #                  position: [iterations x 6 double]
    #                  velocity: [iterations x 6 double]
    #              acceleration: [iterations x 6 double]
    #                forceTotal: [iterations x 6 double]
    #            forceActuation: [iterations x 6 double]
    #           forceConstraint: [iterations x 6 double]
    #    forceInternalMechanics: [iterations x 6 double]
    #    powerInternalMechanics: [iterations x 6 double]
    ######################################
    try:
        ptos = output['ptos']
        num_ptos = len(ptos[0][0]['name'][0]) 
        name = []   
        time = []
        position = []
        velocity = []
        acceleration = []
        forceTotal = []
        forceActuation = []
        forceConstraint = []
        forceInternalMechanics = []
        powerInternalMechanics= []
        for pto in range(num_ptos):
            name.append(ptos[0][0]['name'][0][pto][0])  
            time.append(ptos[0][0]['time'][0][pto])
            position.append(ptos[0][0]['position'][0][pto])
            velocity.append(ptos[0][0]['velocity'][0][pto])
            acceleration.append(ptos[0][0]['acceleration'][0][pto])
            forceTotal.append(ptos[0][0]['forceTotal'][0][pto])        
            forceActuation.append(ptos[0][0]['forceActuation'][0][pto])        
            forceConstraint.append(ptos[0][0]['forceConstraint'][0][pto])        
            forceInternalMechanics.append(ptos[0][0]['forceInternalMechanics'][0][pto])        
            powerInternalMechanics.append(ptos[0][0]['powerInternalMechanics'][0][pto])        
    except:
        num_ptos = 0         
        
    ######################################
    ## create pto_output DataFrame
    ######################################      
    if num_ptos == 1:  
        for pto in range(num_ptos):
            pto_output = pd.DataFrame(data = time[0],columns=['time'])   
            pto_output = pto_output.set_index('time') 
            pto_output.name = name[pto]
            for dof in range(6):                
                pto_output[f'position_dof{dof+1}'] = position[pto][:,dof]
                pto_output[f'velocity_dof{dof+1}'] = velocity[pto][:,dof]
                pto_output[f'acceleration_dof{dof+1}'] = acceleration[pto][:,dof]                 
                pto_output[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]            
                pto_output[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]     
                pto_output[f'forceActuation_dof{dof+1}'] = forceActuation[pto][:,dof]                 
                pto_output[f'forceConstraint_dof{dof+1}'] = forceConstraint[pto][:,dof]            
                pto_output[f'forceInternalMechanics_dof{dof+1}'] = forceInternalMechanics[pto][:,dof]     
                pto_output[f'powerInternalMechanics_dof{dof+1}'] = powerInternalMechanics[pto][:,dof]   
    elif num_ptos > 1:
        pto_num_output = {}     
        for pto in range(num_ptos):
            tmp3 = pd.DataFrame(data = time[0],columns=['time'])   
            tmp3 = tmp3.set_index('time') 
            tmp3.name = name[pto]
            for dof in range(6):                
                tmp3[f'position_dof{dof+1}'] = position[pto][:,dof]
                tmp3[f'velocity_dof{dof+1}'] = velocity[pto][:,dof]
                tmp3[f'acceleration_dof{dof+1}'] = acceleration[pto][:,dof]                 
                tmp3[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]            
                tmp3[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]     
                tmp3[f'forceActuation_dof{dof+1}'] = forceActuation[pto][:,dof]                 
                tmp3[f'forceConstraint_dof{dof+1}'] = forceConstraint[pto][:,dof]            
                tmp3[f'forceInternalMechanics_dof{dof+1}'] = forceInternalMechanics[pto][:,dof]     
                tmp3[f'powerInternalMechanics_dof{dof+1}'] = powerInternalMechanics[pto][:,dof]                 
            pto_num_output[f'pto{pto+1}'] = tmp3
        pto_output = pto_num_output.copy()  
    else:
        print("pto class not used") 
        pto_output = []


    ######################################
    ## import wecSim constraint class
    #                       
    #            name: ''
    #            time: [iterations x 1 double]
    #        position: [iterations x 6 double]
    #        velocity: [iterations x 6 double]
    #    acceleration: [iterations x 6 double]
    # forceConstraint: [iterations x 6 double]
    ######################################    
    try:
        constraints = output['constraints']
        num_constraints = len(constraints[0][0]['name'][0])   
        name = []   
        time = []
        position = []
        velocity = []
        acceleration = []
        forceConstraint = []
        for constraint in range(num_constraints):
            name.append(constraints[0][0]['name'][0][constraint][0])   
            time.append(constraints[0][0]['time'][0][constraint])
            position.append(constraints[0][0]['position'][0][constraint])
            velocity.append(constraints[0][0]['velocity'][0][constraint])
            acceleration.append(constraints[0][0]['acceleration'][0][constraint])
            forceConstraint.append(constraints[0][0]['forceConstraint'][0][constraint])        
    except:
        num_constraints = 0 
        
    ######################################
    ## create constraint_output DataFrame
    ######################################    
    if num_constraints == 1:
        for constraint in range(num_constraints):          
            constraint_output = pd.DataFrame(data = time[0],columns=['time'])   
            constraint_output = constraint_output.set_index('time') 
            constraint_output.name = name[constraint]        
            for dof in range(6):
                constraint_output[f'position_dof{dof+1}'] = position[constraint][:,dof]
                constraint_output[f'velocity_dof{dof+1}'] = velocity[constraint][:,dof]
                constraint_output[f'acceleration_dof{dof+1}'] = acceleration[constraint][:,dof]            
                constraint_output[f'forceConstraint_dof{dof+1}'] = forceConstraint[constraint][:,dof]
    elif num_constraints > 1:
        constraint_num_output = {}
        for constraint in range(num_constraints):
            tmp4 = pd.DataFrame(data = time[0],columns=['time'])   
            tmp4 = tmp4.set_index('time') 
            tmp4.name = name[constraint]
            for dof in range(6):                
                tmp4[f'position_dof{dof+1}'] = position[constraint][:,dof]
                tmp4[f'velocity_dof{dof+1}'] = velocity[constraint][:,dof]
                tmp4[f'acceleration_dof{dof+1}'] = acceleration[constraint][:,dof]            
                tmp4[f'forceConstraint_dof{dof+1}'] = forceConstraint[constraint][:,dof]
            constraint_num_output[f'constraint{constraint+1}'] = tmp4         
        constraint_output = constraint_num_output.copy()            
    else:
        print("constraint class not used") 
        constraint_output = []


    ######################################
    ## import wecSim moopring class
    # 
    #         name: ''
    #         time: [iterations x 1 double]
    #     position: [iterations x 6 double]
    #     velocity: [iterations x 6 double]
    # forceMooring: [iterations x 6 double]
    ######################################
    try:
        moorings = output['mooring']
        num_moorings = len(moorings[0][0]['name'][0])   
        name = []   
        time = []
        position = []
        velocity = []
        forceMooring = []
        for mooring in range(num_moorings):
            name.append(moorings[0][0]['name'][0][mooring][0])   
            time.append(moorings[0][0]['time'][0][mooring])
            position.append(moorings[0][0]['position'][0][mooring])
            velocity.append(moorings[0][0]['velocity'][0][mooring])
            forceMooring.append(moorings[0][0]['forceMooring'][0][mooring])    
    except:
        num_moorings = 0 

    ######################################
    ## create mooring_output DataFrame
    ######################################    
    if num_moorings == 1:
        mooring_output = pd.DataFrame(data = time[0],columns=['time'])   
        mooring_output = mooring_output.set_index('time')         
        mooring_output.name = name[mooring]
        for mooring in range(num_moorings):
            for dof in range(6):
                mooring_output[f'position_dof{dof+1}'] = position[mooring][:,dof]
                mooring_output[f'velocity_dof{dof+1}'] = velocity[mooring][:,dof]
                mooring_output[f'forceMooring_dof{dof+1}'] = forceMooring[mooring][:,dof]
        mooring_output
    elif num_moorings > 1:   
        mooring_num_output = {}
        for mooring in range(num_moorings):
            tmp5 = pd.DataFrame(data = time[0],columns=['time'])   
            tmp5 = tmp5.set_index('time') 
            tmp5.name = name[mooring]
            for dof in range(6):                
                tmp5[f'position_dof{dof+1}'] = position[mooring][:,dof]
                tmp5[f'velocity_dof{dof+1}'] = velocity[mooring][:,dof]
                tmp5[f'forceMooring_dof{dof+1}'] = forceMooring[mooring][:,dof]
            mooring_num_output[f'mooring{mooring+1}'] = tmp5   
        mooring_output = mooring_num_output.copy()          
    else:
        print("mooring class not used") 
        mooring_output = []

    
    ######################################
    ## import wecSim moorDyn class
    #
    #    Lines: [1×1 struct]
    #    Line1: [1×1 struct]
    #    Line2: [1×1 struct]
    #    Line3: [1×1 struct]
    #    Line4: [1×1 struct]
    #    Line5: [1×1 struct]
    #    Line6: [1×1 struct]  
    ######################################
    try:
        moorDyn = output['moorDyn']       
        num_lines = len(moorDyn[0][0][0].dtype) - 1    # number of moorDyn lines
      
        Lines =  moorDyn[0][0]['Lines'][0][0][0]      
        signals = Lines.dtype.names
        num_signals = len(Lines.dtype.names)
        data = Lines[0]      
        time = data[0]
        Lines = pd.DataFrame(data = time,columns=['time'])   
        Lines = Lines.set_index('time')       
        for signal in range(1,num_signals):
            Lines[signals[signal]] = data[signal]        
        Lines_output= {'Lines': Lines}

        Line_num_output = {}  
        for line_num in range(1,num_lines+1):
          tmp =  moorDyn[0][0][f'Line{line_num}'][0][0][0]
          signals = tmp.dtype.names
          num_signals = len(tmp.dtype.names)
          data = tmp[0]
          time = data[0]
          tmp = pd.DataFrame(data = time,columns=['time'])   
          tmp = tmp.set_index('time')       
          for signal in range(1,num_signals):
            tmp[signals[signal]] = data[signal]              
          Line_num_output[f'Line{line_num}'] = tmp
        
        moorDyn_output = Lines_output.copy()
        moorDyn_output.update(Line_num_output)
      
    except:
        print("moorDyn class not used") 
        moorDyn_output = []


    ######################################
    ## import wecSim ptosim class
    # 
    #                 name: ''
    #             pistonCF: [1×1 struct]
    #            pistonNCF: [1×1 struct]
    #           checkValve: [1×1 struct]
    #                valve: [1×1 struct]
    #          accumulator: [1×2 struct]
    #       hydraulicMotor: [1×1 struct]
    #      rotaryGenerator: [1×1 struct]
    #    pmLinearGenerator: [1×1 struct]
    #    pmRotaryGenerator: [1×1 struct]
    #      motionMechanism: [1×1 struct]
    ######################################    
    try:
        ptosim = output['ptosim']  
        num_ptosim = len(ptosim[0][0]['name'][0])   # number of ptosim  
        print("ptosim class output not supported at this time") 
    except:
        print("ptosim class not used") 
        ptosim_output = []


    ######################################
    ## create wecSim output DataFrame of Dict
    ######################################
    ws_output = {'wave' : wave_output, 
                 'bodies' : body_output,
                 'ptos' : pto_output,
                 'constraints' : constraint_output,                 
                 'mooring' : mooring_output,
                  'moorDyn': moorDyn_output, 
                  'ptosim' : ptosim_output
                 }
    return ws_output 
