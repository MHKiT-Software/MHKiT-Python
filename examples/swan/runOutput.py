import pandas as pd
import matplotlib.pyplot as plt
from mhkit.wave.io import swan

# User manual
#http://swanmodel.sourceforge.net/download/zip/swanuse.pdf

swanTable = swan.read_table('SWANOUT.DAT')

swanBlockTxt, metaData = swan.read_block('SWANOUTBlock.DAT')
swanBlockMat, metaDataMat = swan.read_block('SWANOUT.mat')

swanBlockTxtDf = swan.dictionary_of_grid_to_table(swanBlockTxt)
swanBlockMatDf = swan.dictionary_of_grid_to_table(swanBlockMat)

import ipdb;ipdb.set_trace()

#inputFile='INPUT'
#input_params = swan.parse_input(inputFile)


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
