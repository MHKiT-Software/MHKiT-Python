import pandas as pd
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