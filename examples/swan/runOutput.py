import pandas as pd
import matplotlib.pyplot as plt
from mhkit.wave.io import swan

inputFile='INPUT'
input_params = parse_input(inputFile)

df = swan.read_output('SWANOUT.DAT')

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
