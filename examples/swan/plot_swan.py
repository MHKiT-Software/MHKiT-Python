# Updated to Python 3.7

from scipy.io import loadmat
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

###########################################################################################

def clean_line(line):
    line = line.strip()           # remove leading and trailing whitespace, carriage returns
    line = line.replace(',', ' ') # change commas to spaces
    line = line.split()           # split on spaces
    
    return line

###########################################################################################

def GetStruc(InpFile):
    ## determine whether run is structured/unstructured
    iunst=-1    #0 = structured, 1 = unstructured, -1 = undetermined
    with open(inputfile,'r') as fin:
        for line in fin:
            linein = line.split()
            ## read until 'CGRID' colorfound
            if len(linein) == 0:
                continue # cycle this loop
            if linein[0].upper().find('''CGRID''') >= 0:
                if linein[1].upper().find('''UNS''') >= 0:
                    iunst=1
                    break
                if linein[1].upper().find('''REG''') >= 0:
                    iunst=0
                    break
   
    return iunst

###########################################################################################

def GetMat(InpFile):
    ## get the matlab output file
    with open(InpFile,'r') as fin:
        linein = fin.readline()
        ## read until 'BLOCK COMPGRID' output line found
        ifound=1
        while linein.upper().find("""BLOCK 'COMPGRID'""") < 0:
            linein = fin.readline()
            if linein == '':  #EOF
                ifound=0
                break
    if ifound == 0:
        matfile = ''
        data = ''
    else:
        matfile = linein.split()[3].replace("'","")
        print('Loading ' + matfile)
        data = loadmat(matfile, struct_as_record=False, squeeze_me=True)    
    
    return matfile, data

###########################################################################################

def GetNodeEleFile(InpFile):
    ## get node and element files
    with open(InpFile,'r') as fin:
        for line in fin:
            linein = line.split()
            ##  find the line defining the unstructure grid files
            if linein[0].upper().find("""READ""") >= 0 and linein[1].upper().find("""UNSTRUC""") >= 0:
                casename = linein[3].replace("'","")
                break
    
    return casename+'.node', casename+'.ele'

###########################################################################################

def ReadNodeLocs(NodeFile):
    with open(NodeFile,'r') as fin:
        nnode = int(float(fin.readline().split()[0]))
        # for each line in file, split it and extract x,y, convert to float
        # first line is nverts, ndim, nattr (???), nbmark (value of boundary mark)
        # other lines are node#, xloc, yloc, vmark (mark value?)
        nodelocs = np.asarray( [ [float(val) for val in dataline.split()[1:3]] for dataline in fin.readlines()] ) # just take columns 2 and 3 (x and y)
    
    return nnode, nodelocs

###########################################################################################

def ReadElems(EleFile):
    with open(EleFile,'r') as fin:
        nele = int(fin.readline().split()[0])
        # for each line in file, split it and extract x,y, convert to float
        # subtract one because python counts from 0
        elenodes = np.asarray( [ [int(val)-1 for val in dataline.split()[1:4]] for dataline in fin.readlines()] )

    return nele, elenodes

###########################################################################################

def GetStrucGrid(InpFile):
    ## get the grid coordinates
    with open(InpFile,'r') as fin:
        linein = fin.readline()
        #
        # read until 'CGRID REGULAR' found
        #
        while linein.upper().find('''CGRID REG''') < 0:
            linein = fin.readline()

    GridInfo=linein.split()[2:]  # chop off keywords in the front

    xo = float(GridInfo[0])
    yo = float(GridInfo[1])
#    rot = float(GridInfo[2])
    xmax = float(GridInfo[3])
    ymax = float(GridInfo[4])
    nx = int(GridInfo[5]) +1
    ny = int(GridInfo[6]) +1

    xvalues = np.asarray(np.linspace(xo,xmax,nx))
    yvalues = np.asarray(np.linspace(yo,ymax,ny))

    # TODO: add rotation functionality
    return xvalues, yvalues

###########################################################################################

def GetObs(InpFile):
    ## get the obstacle locations
    obslist=[]
    with open(InpFile,'r') as fin:
        for linein in fin:
            ## read each 'OBSTACLE' line found
            if linein.upper().find('''OBSTACLE''') >= 0:
                obsinfo = [float(val) for val in clean_line(linein)[-4:] ]  # chop off initial keywords, float the rest
                obslist.append(obsinfo)    # append the obstacle info to the list
    
    return obslist

###########################################################################################
## Begin body of the script
###########################################################################################

#import sys
## define input filename
inputfile = 'INPUT'

## read in grid structure format 0 = structured, 1 = unstructured
iunst = GetStruc(inputfile)
if iunst == -1:
    print('ERROR structure not found')
    exit()

## get the matlab dataarray
matfile, mat = GetMat(inputfile)
if matfile == '':
    print('ERROR matfile not found')
    exit()

## load mat variables
plot_vars = list(mat.keys())
plot_vars.remove('__globals__')
plot_vars.remove('__header__')
plot_vars.remove('__version__')
plot_unit = ['[m]', '[s]', '[s]', '[deg]']
num_vars = len(plot_vars)
len_vars = range(num_vars)

## loop through each variable
if num_vars > 1:
    for i in len_vars:
        num_vars = len_vars[i]
        
        ## get obstacle locations
        obslist = GetObs(inputfile)
        
        ## set up the plotting
        fig, ax=plt.subplots()
        fig.set_facecolor("white")
        
        ## choose color map
#        colormap = 'jet'
#        colormap = 'copper'
#        colormap = 'plasma'
        colormap = 'viridis'
        
        ## plot the data
        if iunst == 1:
            nodefile, elefile  = GetNodeEleFile(inputfile)
            nnode, nodelocs = ReadNodeLocs(nodefile)
            nele, elenodes = ReadElems(elefile)
        
            x = nodelocs[:,0]
            y = nodelocs[:,1]
            z = mat[plot_vars[num_vars]]
        
            import matplotlib.tri as tri
            grid = tri.Triangulation(x, y, triangles=elenodes, mask=None)
        
            CSF = plt.tricontourf(grid,z,256,cmap=colormap)
            #gridlines = plt.triplot(grid,color='k')
        else:
            x,y = GetStrucGrid(inputfile)
            xi, yi = np.meshgrid(x,y)
            z = mat[plot_vars[num_vars]]
            CSF = plt.contourf(xi,yi,z,256,cmap=colormap)
                        
        ## square axes
        plt.axis('equal')
        
        ## plot axes
        plt.xlim(min(x),max(x))
        plt.ylim(min(y),max(y))
        plt.xlabel('Cross-shore [m]')
        plt.ylabel('Along-shore [m]')
        plt.title(plot_vars[num_vars])
        cbar = plt.colorbar(fraction=0.15, pad=0.05, format =  '%0.4f')
        cbar.set_label(plot_vars[num_vars] + ' ' + plot_unit[num_vars])
        
        ##  plot the obstacles
        for obs in obslist:
            x = [ obs[0], obs[2] ]
            y = [ obs[1], obs[3] ]
            plt.plot(x,y, color='k', linewidth=2)
            #plt.show()
        
        ## split off the extension
        import os.path
        pngFile = os.path.splitext(matfile)[0]
        ## add variable plotted
        pngFile += '_'+plot_vars[num_vars]
        ## add png extension
        pngFile += '.png'
        plt.savefig(pngFile)

