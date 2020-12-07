import pandas as pd
import matplotlib.pyplot as plt
from mhkit.wave.io import swan

# User manual
#http://swanmodel.sourceforge.net/download/zip/swanuse.pdf


#data = swan.read_block_output_mat('SWANOUT.mat')

dataDict, metaData = swan.read_block_output_txt('SWANOUTBlock.DAT')

variables = [var for var in  dataDict.keys() ]
#import ipdb;ipdb.set_trace()
var0 = variables[0]
data = swan.block_to_table(dataDict[var0], name=var0)
for var in variables[1:]:
    
    tmp_dat = swan.block_to_table(dataDict[var], name=var)
    data[var] = tmp_dat[var]
import ipdb;ipdb.set_trace()

#inputFile='INPUT'
#input_params = swan.parse_input(inputFile)



#df = swan.read_output('SWANOUT.DAT')




plt.tricontourf(df.Xp, df.Yp, df.Hsig, levels=256, cmap='jet')
plt.colorbar()

plt.figure()
plt.tricontourf(df.Xp, df.Yp, df.Dir, levels=256, cmap='jet')
plt.colorbar()

plt.figure()
plt.tricontourf(df.Xp, df.Yp, df.RTpeak, levels=256, cmap='jet')
plt.colorbar()

plt.figure()
plt.tricontourf(df.Xp, df.Yp, df.TDir, levels=256, cmap='jet')
plt.colorbar()

import ipdb;ipdb.set_trace()
