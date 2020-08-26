import mhkit
import scipy.io as sio
import matplotlib.pyplot as plt

######################################
# Example loading WEC-Sim Output
######################################
    

# ## loading wec-sim output 
# file_name = './data/wecsim_output.mat'
# ws_output = mhkit.wave.io.read_wecSim(file_name)

# # # no Mooring
# file_name = './data/RM3_matlabWorkspace_structure.mat'
# ws_output = mhkit.wave.io.read_wecSim(file_name)

# # with Mooring
# file_name = './data/RM3MooringMatrix_matlabWorkspace_structure.mat'
# ws_output = mhkit.wave.io.read_wecSim(file_name)

# with moorDyn
file_name = './data/RM3MooringMatrix_matlabWorkspace_structure.mat'
ws_output = mhkit.wave.io.read_wecSim(file_name)



ws_output.keys()

######################################
# Wave
######################################
ws_output['wave'].head()
wave = ws_output['wave']
plt.figure()
wave.plot()

######################################
# Bodies
######################################
ws_output['bodies'].keys()
bodies = ws_output['bodies']
bodies.keys()

plt.figure()
bodies['body1'].keys()
bodies['body1'].name
bodies['body1'].position_dof3.max()
bodies['body1'].position_dof3.plot()


plt.figure()
df = bodies['body1']
filter_col = [col for col in df if col.startswith('position')]
# filter_col = [col for col in df if col.endswith('dof3')]
df[filter_col].plot()

plt.figure()
bodies['body2'].name
bodies['body2'].position_dof3.max()
bodies['body2'].position_dof3.plot()

######################################
# PTO
######################################
ptos=  ws_output['ptos']
ptos.head()
plt.figure()
(-1*ptos.powerInternalMechanics_dof3).plot()

######################################
# Constraints
######################################
constraints = ws_output['constraints']
constraints.head()
plt.figure()
constraints['forceConstraint_dof4'].plot()

######################################
# Mooring
######################################
if len(ws_output['mooring']) > 0:
    ws_output['mooring'].head()
    mooring = ws_output['mooring']
    mooring.forceMooring_dof5.plot()

######################################
# MoorDyn
######################################
elif len(ws_output['mooring']) > 0:
    ws_output['moorDyn']['Lines'].head()
    moorDyn = ws_output['moorDyn']
    moorDyn['Lines'].plot()

else:
    print("no mooring used") 


######################################
## Run MHKiT modules
######################################
ws_spectrum = mhkit.wave.resource.elevation_spectrum(wave,60,1000)

plt.figure()
ws_spectrum.plot()
Tp = mhkit.wave.resource.peak_period(ws_spectrum)
Hs = mhkit.wave.resource.significant_wave_height(ws_spectrum)
print(Tp)
print(Hs)




