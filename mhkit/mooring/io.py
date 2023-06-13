import pandas as pd


def moordyn(filepath, input_file=None):
    """Reads in MoorDyn OUT files such as "FAST.MD.out" and "FAST.MD.Line1.out" and stores inside
    xarray. Also allows for parsing and storage of MoorDyn input file as attributes inside the xarray.

    Parameters
    ----------
    filepath : str
        Path to MoorDyn OUT file
    inputfile : str (optional)
        Path to MoorDyn input file

    Returns
    -------
    xr.Dataset
        Dataset containing parsed MoorDyn OUT file
    """
    assert isinstance(filepath, str), 'filepath must be of type str'
    if input_file: assert isinstance(input_file, str), 'inputfile must be of type str'
    
    data = pd.read_csv(filepath, header=0, skiprows=[1], sep=' ', skipinitialspace=True, index_col=0)
    data = data.dropna(axis=1)
    ds = data.to_xarray()

    if input_file:
        ds = _moordyn_input(input_file, ds)

    return ds


def _moordyn_input(input_file, ds):
    """Internal function used to parse MoorDyn input file and write attributes

    Parameters
    ----------
    input_file : str
        Path to moordyn input file
    ds : xr.Dataset
        xarray Dataset to be written to

    Returns
    -------
    xr.Dataset
        return Dataset that includes input file parameters as attributes
    """
    f = open(input_file, 'r')
    for line in f:          # loop through each line in the file
        # get line type property sets
        if line.count('---') > 0 and (line.upper().count('LINE DICTIONARY') > 0 or line.upper().count('LINE TYPES') > 0):
            linetypes = dict()
            line = next(f) # skip this header line, plus channel names and units lines
            variables = line.split()
            line = next(f)
            units = line.split()
            line = next(f)
            while line.count('---') == 0:
                entries = line.split()
                linetypes[entries[0]] = dict()
                for x in range(1,len(entries)):
                    linetypes[entries[0]][variables[x]] = entries[x]
                line = next(f)
            linetypes['units'] = units[1:]
            ds.attrs['LINE_TYPES'] = linetypes
        
        # TODO: get rod type property sets
        if line.count('---') > 0 and (line.upper().count('ROD DICTIONARY') > 0 or line.upper().count('ROD TYPES') > 0):
            line = next(f) # skip this header line, plus channel names and units lines
            # line = next(f)
            # line = next(f)
            # RodDict = dict()
            # while line.count('---') == 0:
            #     entries = line.split()
            #     #RodTypesName.append(entries[0]) # name string
            #     #RodTypesD.append(   entries[1]) # diameter
            #     RodDict[entries[0]] = entries[1] # add dictionary entry with name and diameter
            #     line = next(f)
            # #ds.attrs['ROD_TYPES'] = RodDict

        # TODO: get properties of each Body
        if line.count('---') > 0 and (line.upper().count('BODIES') > 0 or line.upper().count('BODY LIST') > 0 or line.upper().count('BODY PROPERTIES') > 0):
            line = next(f) # skip this header line, plus channel names and units lines
            # line = next(f)
            # line = next(f)
            # while line.count('---') == 0:
            #     entries = line.split()                    
            #     entry0 = entries[0].lower() 
                
            #     num = np.int("".join(c for c in entry0 if not c.isalpha()))  # remove alpha characters to identify Body #
                
            #     if ("fair" in entry0) or ("coupled" in entry0) or ("ves" in entry0):       # coupled case
            #         bodyType = -1                        
            #     elif ("con" in entry0) or ("free" in entry0):                              # free case
            #         bodyType = 0
            #     else:                                                                      # for now assuming unlabeled free case
            #         bodyType = 0
            #         # if we detected there were unrecognized chars here, could: raise ValueError(f"Body type not recognized for Body {num}")
            #     #bodyType = -1   # manually setting the body type as -1 for FAST.Farm SM investigation
                
            #     r6  = np.array(entries[1:7], dtype=float)   # initial position and orientation [m, rad]
            #     r6[3:] = r6[3:]*np.pi/180.0                 # convert from deg to rad
            #     rCG = np.array(entries[7:10], dtype=float)  # location of body CG in body reference frame [m]
            #     m = np.float_(entries[10])                   # mass, centered at CG [kg]
            #     v = np.float_(entries[11])                   # volume, assumed centered at reference point [m^3]
                
            #     #self.bodyList.append( Body(self, num, bodyType, r6, m=m, v=v, rCG=rCG) )
                            
            #     line = next(f)
                
        # TODO: get properties of each rod
        if line.count('---') > 0 and (line.upper().count('RODS') > 0 or line.upper().count('ROD LIST') > 0 or line.upper().count('ROD PROPERTIES') > 0):
            line = next(f) # skip this header line, plus channel names and units lines
            # line = next(f)
            # line = next(f)
            # while line.count('---') == 0:
            #     entries = line.split()              
            #     entry0 = entries[0].lower() 
                
            #     num = np.int("".join(c for c in entry0 if not c.isalpha()))  # remove alpha characters to identify Rod #
            #     lUnstr = 0 # not specified directly so skip for now
            #     #dia = RodDict[entries[2]] # find diameter based on specified rod type string
            #     nSegs = np.int(entries[9])
                
            #     # additional things likely missing here <<<
                
            #     #RodList.append( Line(dirName, num, lUnstr, dia, nSegs, isRod=1) )
            #     line = next(f)
                
        # get properties of each Point
        if line.count('---') > 0 and (line.upper().count('POINTS') > 0 or line.upper().count('POINT LIST') > 0 or line.upper().count('POINT PROPERTIES') > 0):
            line = next(f) # skip this header line, plus channel names and units lines
            variables = line.split()
            line = next(f)
            units = line.split()
            line = next(f)
            points = dict()
            while line.count('---') == 0:
                entries = line.split()
                points[entries[0]] = dict()
                for x in range(1,len(entries)):
                    points[entries[0]][variables[x]] = entries[x]
                line = next(f)
            points['units'] = units[1:]
            ds.attrs['POINTS'] = points
                
        # get properties of each line
        if line.count('---') > 0 and (line.upper().count('LINES') > 0 or line.upper().count('LINE LIST') > 0 or line.upper().count('LINE PROPERTIES') > 0):
            line = next(f) # skip this header line, plus channel names and units lines
            variables = line.split()
            line = next(f)
            units = line.split()
            line = next(f)
            lines = {}
            while line.count('---') == 0:
                entries = line.split()
                lines[entries[0]] = dict()
                for x in range(1,len(entries)):
                    lines[entries[0]][variables[x]] = entries[x]
                line = next(f)
            lines['units'] = units[1:]
            ds.attrs['LINES'] = lines
                
        # get options entries
        if line.count('---') > 0 and "options" in line.lower():
            #print("READING OPTIONS")
            line = next(f) # skip this header line
            options = {}
            while line.count('---') == 0:
                entries = line.split()
                options[entries[1]] = entries[0]
                line = next(f)
            ds.attrs['OPTIONS'] = options

    f.close()  # close data file

    return ds