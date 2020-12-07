from scipy.io import loadmat
import pandas as pd
import numpy as np
import re 

def parse_input(input_file):
    '''
    Parses Inputfile to define variables
    
    Parameters
    ----------
    input_file: str
        Name of SWAN input file
    
    Returns
    -------
    inputs: Dict
        Dictionary of model inputs
    '''
    f = open(input_file,'r')
    import ipdb; ipdb.set_trace()
    

def read_output(output_file):
    '''
    Reads in SWAN output
    
    Parameters
    ----------
    output_file: str
        filename to import
        
    Returns
    -------
    swan_data: DataFrame
        Dataframe of swan output
    '''
    f = open(output_file,'r')
    header_line_number = 5
    for i in range(header_line_number):
        line = f.readline()
    header = re.split("\s+",line.rstrip().strip('%').lstrip())
    
    swan_data = pd.read_csv(output_file, sep='\s+', comment='%', 
                            names=header) 
    return swan_data    


def _parse_line_metadata(line):
    '''
    Parses the variable meta data into a dictionary
    
    Parameters
    ----------
    line: str
        line from block swan data to parse
        
    Returns
    -------
    metaDict: Dictionary
        Dictionary of variable metadata
    '''
    metaDict={}        
    meta=re.sub('\s+', " ", line.replace(',', ' ').strip('% \n').replace('**', 'vars:'))
    mList = meta.split(':')
    elms = [elm.split(' ') for elm in mList]
    for elm in elms:
        try:
            elm.remove('')
        except:
            pass                
    for i in range(len(elms)-1):
        elm = elms[i]
        key = elm[-1]
        val = ' '.join(elms[i+1][:-1])
        metaDict[key] = val
    metaDict[key] = ' '.join(elms[-1]) 
    metaDict['unitMultiplier'] = float(metaDict['Unit'].split(' ')[0])
        
    return metaDict    


def block_to_table(data, name='values'):
    '''
    Converts structured 2D grid to Table format
    
    Parameters
    ----------
    data: DataFrame
        DataFrame in with columns as X indicie and Y as index.
    name: string (Optional)
        Name of data column in returned table. Default='values'
    Returns
    -------
    table: DataFrame
        DataFrame with columns x,y,values           
    '''
    table = data.unstack().reset_index(name=name)
    table = table.rename(columns={'level_0':'x', 'level_1': 'y'})
    return table
    
def read_block_output_txt(output_file):
    '''
    Reads in SWAN output and creates a dictionary of DataFrames
    for each SWAN output variable.
    
    Parameters
    ----------
    output_file: str
        filename to import
        
    Returns
    -------
    data: Dictionary
        Dictionary of DataFrame of swan output variables
    '''
    f = open(output_file) 
    runLines=[]
    metaDict = {}
    column_position = None
    dataDict={}
    for position, line in enumerate(f):
        
        if line.startswith('% Run'):
            varPosition = position
            runLines.extend([position])
            column_position = position + 5                       
            varDict = _parse_line_metadata(line)            
            
            metaDict[varPosition] = varDict            
            variable = varDict['vars']
            dataDict[variable] = {}
            
        if position==column_position and column_position!=None:
           columns = line.strip('% \n').split()
           metaDict[varPosition]['cols'] = columns
           N_columns = len(columns)
           columns_position = None
           
        
        if not line.startswith('%'):
            raw_data = ' '.join(re.split(' |\.', line.strip(' \n'))).split()
            index_number = int(raw_data[0])
            columns_data = raw_data[1:]
            data=[]
            possibleNaNs = ['****']
            NNaNsTotal = sum([line.count(nanVal) for nanVal in possibleNaNs])
            
            if NNaNsTotal>0:
                for vals in columns_data:
                    NNaNs = 0                                      
                    for nanVal in possibleNaNs:
                        NNaNs += vals.count(nanVal)
                    if NNaNs > 0:
                        for i in range(NNaNs):
                            data.extend([np.nan]) 
                    else:
                        data.extend([float(vals)])
            else:                
                data.extend([float(val) for val in columns_data])             
                
            dataDict[variable][index_number] = data
                
    metaData = pd.DataFrame(metaDict).T        
    
    for var in metaData.vars.values: 
        df = pd.DataFrame(dataDict[var]).T        
        varCols =  metaData[metaData.vars == var].cols.values.tolist()[0]
        colsDict = dict(zip(df.columns.values.tolist(), varCols))
        df.rename(columns=colsDict)
        unitMultiplier = metaData[metaData.vars == var].unitMultiplier.values[0]
        dataDict[var] = df * unitMultiplier
        #import ipdb;ipdb.set_trace()
    
    
    return dataDict, metaData        
    #import ipdb; ipdb.set_trace()

    
    

def read_block_output_mat(output_file):
    '''
    Reads in SWAN matlab output and creates a dictionary of DataFrames
    for each swan output variable.
    
    Parameters
    ----------
    output_file: str
        filename to import
        
    Returns
    -------
    data: Dictionary
        Dictionary of DataFrame of swan output variables
    '''
    data = loadmat(output_file, struct_as_record=False, squeeze_me=True)
    removeKeys = ['__header__', '__version__', '__globals__']
    for key in removeKeys:
        data.pop(key, None)
    for key in data.keys():
        data[key] = pd.DataFrame(data[key])
    return data
    
    
def swan_units(variable=None)    :
    '''
    Returns a dictionary of swan units if none is supplied. If a specific
    variable is supplied then only those units are returned.
    
    SWAN expects all quantities that are given by the user to be 
    expressed in S.I. units: m, kg, s and composites of these with 
    accepted compounds, such as Newton (N) and Watt(W). Consequently, 
    the wave height and water depth are in m, wave period in s, etc. 
    For wind and wave direction both the Cartesian and a nautical 
    convention can be used (see below). Directions and spherical 
    coordinates are in degrees (0) and not in radians.
    
    For the output of wave energy the user can choose between 
    variance (m2) or energy (spatial) density (Joule/m2, i.e. energy 
    per unit sea surface) and the equivalents in case of energy 
    transport (m3/s or W/m, i.e. energy transport per unit length) 
    and spectral energy density (m2/Hz/Degr or Js/m2/rad, i.e. 
    energy per unit frequency and direction per unit sea surface area). 
    The wave−induced stress components (obtained as spatial derivatives 
    of wave-induced radiation stress) are always expressed in N/m2 even
    if the wave energy is in terms of variance. Note that the energy 
    density is also in Joule/m2 in the case of spherical coordinates.

    SWAN operates either in a Cartesian coordinate system or in a 
    spherical coordinate system, i.e. in a flat plane or on a spherical 
    Earth. In the Cartesian system, all geographic locations and 
    orientations in SWAN, e.g. for the bottom grid or for output points, 
    are defined in one common Cartesian coordinate system with 
    origin (0,0) by definition. This geographic origin may be chosen 
    totally arbitrarily by the user. However, be careful, the numbers 
    for the origin should not be chosen too large; the user is advised 
    to translate the coordinates with an offset. In the spherical 
    system, all geographic locations and orientations in SWAN, e.g. 
    for the bottom grid or for output points, are defined in geographic
    longitude and latitude. Both coordinate systems are designated in 
    this manual as the problem coordinate system.
    
    In the input and output of SWAN the direction of wind and waves 
    are defined according to either
        • the Cartesian convention, i.e. the direction to where the 
          vector points, measured counterclockwise from the positive 
          x−axis of this system (in degrees) or
        • a nautical convention (there are more such conventions), 
        i.e. the direction where the wind or the waves come from, 
        measured clockwise from geographic North.

    All other directions, such as orientation of grids, are according 
    to the Cartesian convention!
    
    For regular grids, i.e. uniform and rectangular, Figure 4.1 
    (in Section 4.5) shows how the locations of the various grids are 
    determined with respect to the problem coordinates. All grid points 
    of curvilinear and unstructured grids are relative to the problem 
    coordinate system.
    
    
    Parameters
    ----------
    variable: string (optional)
        If supplied only returns the requested units

    Returns
    -------
    units: Dictionary
        If variable is none the dictionary is returned
    unit: String
        If Variable is specified the units of that variable are returned
    '''        
    
    units = { 'Xp' : 'm',
              'Yp' : 'm',
              'Hsig' : 'm',
              'Dir' : 'degr',
              'RTpeak' : 's',
              'TDir' : 'degr',
             
            }
    
    
    