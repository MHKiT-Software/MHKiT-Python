import pandas as pd
import numpy as np
import scipy.io as sio

def read_NDBC_file(file_name, missing_values=['MM',9999,999,99]):
    """
    Reads a NDBC wave buoy data file (from https://www.ndbc.noaa.gov).
    
    Realtime and historical data files can be loaded with this function.  
    
    Note: With realtime data, missing data is denoted by "MM".  With historical 
    data, missing data is denoted using a variable number of 
    # 9's, depending on the data type (for example: 9999.0 999.0 99.0).
    'N/A' is automatically converted to missing data.
    
    Data values are converted to float/int when possible. Column names are 
    also converted to float/int when possible (this is useful when column 
    names are frequency).
    
    Parameters
    ------------
    file_name : string
        Name of NDBC wave buoy data file
    
    missing_value : list of values
        List of values that denote missing data    
    
    Returns
    ---------
    data: pandas DataFrame 
        Data indexed by datetime with columns named according to header row 
        
    metadata: dict or None
        Dictionary with {column name: units} key value pairs when the NDBC file  
        contains unit information, otherwise None is returned
    """
    assert isinstance(file_name, str), 'file_name must be of type str'
    assert isinstance(missing_values, list), 'missing_values must be of type list'
    
    # Open file and get header rows
    f = open(file_name,"r")
    header = f.readline().rstrip().split()  # read potential headers
    units = f.readline().rstrip().split()   # read potential units
    f.close()
    
    # If first line is commented, remove comment sign #
    if header[0].startswith("#"):
        header[0] = header[0][1:]
        header_commented = True
    else:
        header_commented = False
        
    # If second line is commented, indicate that units exist
    if units[0].startswith("#"):
        units_exist = True
    else:
        units_exist = False
    
    # Check if the time stamp contains minutes, and create list of column names 
    # to parse for date
    if header[4] == 'mm':
        parse_vals = header[0:5]
        date_format = '%Y %m %d %H %M'
        units = units[5:]   #remove date columns from units
    else:
        parse_vals = header[0:4]
        date_format = '%Y %m %d %H'
        units = units[4:]   #remove date columns from units
    
    # If first line is commented, manually feed in column names
    if header_commented:
        data = pd.read_csv(file_name, sep='\s+', header=None, names = header,
                           comment = "#", parse_dates=[parse_vals]) 
    # If first line is not commented, then the first row can be used as header                        
    else:
        data = pd.read_csv(file_name, sep='\s+', header=0,
                           comment = "#", parse_dates=[parse_vals])
                             
    # Convert index to datetime
    date_column = "_".join(parse_vals)
    data['Time'] = pd.to_datetime(data[date_column], format=date_format)
    data.index = data['Time'].values
    # Remove date columns
    del data[date_column]
    del data['Time']
    
    # If there was a row of units, convert to dictionary
    if units_exist:
        metadata = {column:unit for column,unit in zip(data.columns,units)}
    else:
        metadata = None

    # Convert columns to numeric data if possible, otherwise leave as string
    for column in data:
        data[column] = pd.to_numeric(data[column], errors='ignore')
        
    # Convert column names to float if possible (handles frequency headers)
    # if there is non-numeric name, just leave all as strings.
    try:
        data.columns = [float(column) for column in data.columns]
    except:
        data.columns = data.columns
    
    # Replace indicated missing values with nan
    data.replace(missing_values, np.nan, inplace=True)
    
    return data, metadata


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
    #         type: 'irregular'
    #         time: [iterations x 1 double]
    #    elevation: [iterations x 1 double]
    ######################################
    try:              
        wave = output['wave']
        wtype = wave[0][0][0][0][0][0]  # Not sure how to store this
        time = wave[0][0]['time'][0][0].squeeze()
        elevation = wave[0][0]['elevation'][0][0].squeeze()
        
        ######################################
        ## create wave_output DataFrame
        ######################################
        wave_output = pd.DataFrame(data = time,columns=['time'])   
        wave_output = wave_output.set_index('time') 
        wave_output['elevation'] = elevation
        
        wave_type = wtype
        wave_output.name = wave_type
        
        # # Store wave type in DataFrame
        # wave_type = pd.DataFrame(data = [wtype],columns=['type'])
        # wave_output = pd.concat([wave_output,wave_type], axis=1)
        # wave_output.index.name = 'time'
    
    except:
        print("wave class not used") 
        wave_output = []      
    
    
    ######################################
    ## import wecSim body class
    #                       name: 'float'
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
        num_bodies = len(bodies[0][0]['name'][0])   # number of bodies, not stored in DataFrame
        name = []   # Not stored in DataFrame
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
            
        ######################################
        ## create body_output DataFrame
        ######################################    
        body_output = pd.DataFrame(data = time[0],columns=['time'])   
        body_output = body_output.set_index('time') 
        for body in range(num_bodies):
            for dof in range(6):
                body_output['body'+str(body+1)+'_position_dof'+str(dof+1)] = position[body][:,dof]
                # body_output['body'+str(body+1)+'_position_dof'+str(dof+1)] = position[body][:,dof]
                body_output[f'body{body+1}_velocity_dof{dof+1}'] = velocity[body][:,dof]
                body_output['body'+str(body+1)+'_acceleration_dof'+str(dof+1)] = acceleration[body][:,dof]            
                body_output['body'+str(body+1)+'_forceTotal_dof'+str(dof+1)] = forceTotal[body][:,dof]
                body_output['body'+str(body+1)+'_forceExcitation_dof'+str(dof+1)] = forceExcitation[body][:,dof]
                body_output['body'+str(body+1)+'_forceRadiationDamping_dof'+str(dof+1)] = forceRadiationDamping[body][:,dof]
                body_output['body'+str(body+1)+'_forceAddedMass_dof'+str(dof+1)] = forceAddedMass[body][:,dof]
                body_output['body'+str(body+1)+'_forceRestoring_dof'+str(dof+1)] = forceRestoring[body][:,dof]
                body_output['body'+str(body+1)+'_forceMorrisonAndViscous_dof'+str(dof+1)] = forceMorrisonAndViscous[body][:,dof]
                body_output['body'+str(body+1)+'_forceLinearDamping_dof'+str(dof+1)] = forceLinearDamping[body][:,dof]                
        # What about num_bodies and name?

    except:
        print("body class not used") 
        body_output = []    


    ######################################
    ## import wecSim pto class
    #                      name: 'PTO1'
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
        num_ptos = len(ptos[0][0]['name'][0])   # number of ptos, not stored in DataFrame
        name = []   # Not stored in DataFrame
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
        
        ######################################
        ## create pto_output DataFrame
        ######################################      
        pto_output = pd.DataFrame(data = time[0],columns=['time'])   
        pto_output = pto_output.set_index('time') 
        for pto in range(num_ptos):
            for dof in range(6):
                pto_output['pto'+str(pto+1)+'_position_dof'+str(dof+1)] = position[pto][:,dof]
                pto_output['pto'+str(pto+1)+'_velocity_dof'+str(dof+1)] = velocity[pto][:,dof]
                pto_output['pto'+str(pto+1)+'_acceleration_dof'+str(dof+1)] = acceleration[pto][:,dof]                 
                pto_output['pto'+str(pto+1)+'_forceTotal_dof'+str(dof+1)] = forceTotal[pto][:,dof]            
                pto_output['pto'+str(pto+1)+'_forceTotal_dof'+str(dof+1)] = forceTotal[pto][:,dof]     
                pto_output['pto'+str(pto+1)+'_forceActuation_dof'+str(dof+1)] = forceActuation[pto][:,dof]                 
                pto_output['pto'+str(pto+1)+'_forceConstraint_dof'+str(dof+1)] = forceConstraint[pto][:,dof]            
                pto_output['pto'+str(pto+1)+'_forceInternalMechanics_dof'+str(dof+1)] = forceInternalMechanics[pto][:,dof]     
                pto_output['pto'+str(pto+1)+'_powerInternalMechanics_dof'+str(dof+1)] = powerInternalMechanics[pto][:,dof]                 
        pto_output    

    except:
        print("pto class not used") 
        pto_output = []


    ######################################
    ## import wecSim constraint class
    #                       
    #            name: 'Constraint1'
    #            time: [iterations x 1 double]
    #        position: [iterations x 6 double]
    #        velocity: [iterations x 6 double]
    #    acceleration: [iterations x 6 double]
    # forceConstraint: [iterations x 6 double]
    ######################################    
    try:      
        constraints = output['constraints']
        num_constraints = len(constraints[0][0]['name'][0])   # number of constraints, not stored in DataFrame
        name = []   # Not stored in DataFrame
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
    
        ######################################
        ## create constraint_output DataFrame
        ######################################    
        constraint_output = pd.DataFrame(data = time[0],columns=['time'])   
        constraint_output = constraint_output.set_index('time') 
        for constraint in range(num_constraints):
            for dof in range(6):
                constraint_output['constraint'+str(constraint+1)+'_position_dof'+str(dof+1)] = position[constraint][:,dof]
                constraint_output['constraint'+str(constraint+1)+'_velocity_dof'+str(dof+1)] = velocity[constraint][:,dof]
                constraint_output['constraint'+str(constraint+1)+'_acceleration_dof'+str(dof+1)] = acceleration[constraint][:,dof]            
                constraint_output['constraint'+str(constraint+1)+'_forceConstraint_dof'+str(dof+1)] = forceConstraint[constraint][:,dof]
        constraint_output
        
    except:
        print("constraint class not used") 
        constraint_output = []


    ######################################
    ## import wecSim moopring class
    # 
    #         name: 'mooring'
    #         time: [iterations x 1 double]
    #     position: [iterations x 6 double]
    #     velocity: [iterations x 6 double]
    # forceMooring: [iterations x 6 double]
    ######################################
    try:
      moorings = output['mooring']
      num_moorings = len(moorings[0][0]['name'][0])   # number of mooring, not stored in DataFrame
      
      # constraints = output['constraints']
      # num_constraints = len(constraints[0][0]['name'][0])   # number of constraints, not stored in DataFrame
      name = []   # Not stored in DataFrame
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
      ######################################
      ## create mooring_output DataFrame
      ######################################    
      mooring_output = pd.DataFrame(data = time[0],columns=['time'])   
      mooring_output = mooring_output.set_index('time') 
      for mooring in range(num_moorings):
          for dof in range(6):
              mooring_output['mooring'+str(mooring+1)+'_position_dof'+str(dof+1)] = position[mooring][:,dof]
              mooring_output['mooring'+str(mooring+1)+'_velocity_dof'+str(dof+1)] = velocity[mooring][:,dof]
              mooring_output['mooring'+str(mooring+1)+'_forceMooring_dof'+str(dof+1)] = forceMooring[mooring][:,dof]
      mooring_output
          
    except:
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
    #                 name: 'Non-Compressible Fluid Hydraulic'
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
        ptosim = output['ptosim']  #TEMP FIX
        num_ptosim = len(ptosim[0][0]['name'][0])   # number of ptosim  
        ## This needs work...
    except:
      print("ptosim class not used") 
      ptosim_output = []



    ######################################
    ## create wecSim output DataFrame of Dicts - OPTION 1
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


    ######################################
    ## create wecSim output with multiple DataFrames - OPTION 2
    ######################################
    # return wave_output, body_output

    ######################################
    ## create wecSim output DataFrame - OPTION 3 with concat
    ######################################
    # ws_output = pd.concat([wave_output, body_output], axis=1, sort=False)
    # return ws_output

    ######################################
    ## create wecSim output with DataFrame of DataFrames - OPTION 4 
    ######################################
    # ws_output = [wave_output, body_output]    
    # return ws_output   



