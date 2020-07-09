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
    # wave_type = wave[0][0][0][0][0][0]
    wave_time = wave[0][0]['time'][0][0].squeeze()
    wave_elevation = wave[0][0]['elevation'][0][0].squeeze()
    
    ######################################
    ## create wave output dataframe
    ######################################
    wave_output = pd.DataFrame(data = wave_time,columns=['time'])   
    wave_output = wave_output.set_index('time') 
    wave_output['elevation']=wave_elevation
    
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

    ######################################
    ## create wecSim output dataframe - OPTION 3 with concat
    ######################################
    ws_output = pd.concat([wave_output, body_output], axis=1, sort=False)
    


    return ws_output