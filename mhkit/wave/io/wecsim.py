import pandas as pd
import numpy as np
import scipy.io as sio

def read_output(file_name):
    """
    Loads the wecSim response class once 'output' has been saved to a `.mat` 
    structure. 
    
    NOTE: Python is unable to import MATLAB objects. 
    MATLAB must be used to save the wecSim object as a structure. 
        
    Parameters
    ------------
    file_name: string
        Name of wecSim output file saved as a `.mat` structure
        
        
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
    def _write_body_output(body):
        for dof in range(6):                
            tmp_body[f'position_dof{dof+1}'] = position[body][:,dof]
            tmp_body[f'velocity_dof{dof+1}'] = velocity[body][:,dof]
            tmp_body[f'acceleration_dof{dof+1}'] = acceleration[body][:,dof]            
            tmp_body[f'forceTotal_dof{dof+1}'] = forceTotal[body][:,dof]
            tmp_body[f'forceExcitation_dof{dof+1}'] = forceExcitation[body][:,dof]
            tmp_body[f'forceRadiationDamping_dof{dof+1}'] = forceRadiationDamping[body][:,dof]
            tmp_body[f'forceAddedMass_dof{dof+1}'] = forceAddedMass[body][:,dof]
            tmp_body[f'forceRestoring_dof{dof+1}'] = forceRestoring[body][:,dof]
            tmp_body[f'forceMorrisonAndViscous_dof{dof+1}'] = forceMorrisonAndViscous[body][:,dof]
            tmp_body[f'forceLinearDamping_dof{dof+1}'] = forceLinearDamping[body][:,dof]                            
        return tmp_body

    if num_bodies >= 1:
        body_output = {}
        for body in range(num_bodies):
            tmp_body = pd.DataFrame(data = time[0],columns=['time'])   
            tmp_body = tmp_body.set_index('time') 
            tmp_body.name = name[body]
            if num_bodies == 1:
                body_output = _write_body_output(body)
            elif num_bodies > 1:
                body_output[f'body{body+1}'] = _write_body_output(body) 
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
    def _write_pto_output(pto):
        for dof in range(6):                
            tmp_pto[f'position_dof{dof+1}'] = position[pto][:,dof]
            tmp_pto[f'velocity_dof{dof+1}'] = velocity[pto][:,dof]
            tmp_pto[f'acceleration_dof{dof+1}'] = acceleration[pto][:,dof]                 
            tmp_pto[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]            
            tmp_pto[f'forceTotal_dof{dof+1}'] = forceTotal[pto][:,dof]     
            tmp_pto[f'forceActuation_dof{dof+1}'] = forceActuation[pto][:,dof]                 
            tmp_pto[f'forceConstraint_dof{dof+1}'] = forceConstraint[pto][:,dof]            
            tmp_pto[f'forceInternalMechanics_dof{dof+1}'] = forceInternalMechanics[pto][:,dof]     
            tmp_pto[f'powerInternalMechanics_dof{dof+1}'] = powerInternalMechanics[pto][:,dof]
        return tmp_pto

    if num_ptos >= 1:
        pto_output = {}     
        for pto in range(num_ptos):
            tmp_pto = pd.DataFrame(data = time[0],columns=['time'])   
            tmp_pto = tmp_pto.set_index('time') 
            tmp_pto.name = name[pto]
            if num_ptos == 1:  
                pto_output = _write_pto_output(pto)
            elif num_ptos > 1:
                pto_output[f'pto{pto+1}'] = _write_pto_output(pto)
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
    def _write_constraint_output(constraint):
        for dof in range(6):                
            tmp_constraint[f'position_dof{dof+1}'] = position[constraint][:,dof]
            tmp_constraint[f'velocity_dof{dof+1}'] = velocity[constraint][:,dof]
            tmp_constraint[f'acceleration_dof{dof+1}'] = acceleration[constraint][:,dof]            
            tmp_constraint[f'forceConstraint_dof{dof+1}'] = forceConstraint[constraint][:,dof]
        return tmp_constraint

    if num_constraints >= 1:
        constraint_output = {}
        for constraint in range(num_constraints):
            tmp_constraint = pd.DataFrame(data = time[0],columns=['time'])   
            tmp_constraint = tmp_constraint.set_index('time') 
            tmp_constraint.name = name[constraint]
            if num_constraints == 1:
                constraint_output = _write_constraint_output(constraint)
            elif num_constraints > 1:
                constraint_output[f'constraint{constraint+1}'] = _write_constraint_output(constraint)         
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
    def _write_mooring_output(mooring):
        for dof in range(6):                
            tmp_mooring[f'position_dof{dof+1}'] = position[mooring][:,dof]
            tmp_mooring[f'velocity_dof{dof+1}'] = velocity[mooring][:,dof]
            tmp_mooring[f'forceMooring_dof{dof+1}'] = forceMooring[mooring][:,dof]
        return tmp_mooring

    if num_moorings >= 1:   
        mooring_output = {}
        for mooring in range(num_moorings):
            tmp_mooring = pd.DataFrame(data = time[0],columns=['time'])   
            tmp_mooring = tmp_mooring.set_index('time') 
            tmp_mooring.name = name[mooring]
            if num_moorings == 1:   
                mooring_output = _write_mooring_output(mooring)
            elif num_moorings > 1:   
                mooring_output[f'mooring{mooring+1}'] = _write_mooring_output(mooring)
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
        moorDyn_output= {'Lines': Lines}

        Line_num_output = {}  
        for line_num in range(1,num_lines+1):
          tmp_moordyn =  moorDyn[0][0][f'Line{line_num}'][0][0][0]
          signals = tmp_moordyn.dtype.names
          num_signals = len(tmp_moordyn.dtype.names)
          data = tmp_moordyn[0]
          time = data[0]
          tmp_moordyn = pd.DataFrame(data = time,columns=['time'])   
          tmp_moordyn = tmp_moordyn.set_index('time')       
          for signal in range(1,num_signals):
            tmp_moordyn[signals[signal]] = data[signal]              
          Line_num_output[f'Line{line_num}'] = tmp_moordyn
        
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
