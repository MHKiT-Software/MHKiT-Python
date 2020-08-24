import mhkit
import scipy.io as sio
import matplotlib.pyplot as plt
# import h5py  --> load MATLAB objec?

### loading wec-sim output using io function### --> NOT WORKING RIGHT NOW
# file_name = './data/wecsim_output.mat'
# ws_output = mhkit.wave.io.load_wecSim_output(file_name)

# # # no Mooring
# file_name = './data/RM3_matlabWorkspace_structure.mat'
# ws_output = mhkit.wave.io.load_wecSim_output(file_name)

# # with Mooring
# file_name = './data/RM3MooringMatrix_matlabWorkspace_structure.mat'
# ws_output = mhkit.wave.io.load_wecSim_output(file_name)

# with moorDyn
file_name = './data/RM3MooringMatrix_matlabWorkspace_structure.mat'
ws_output = mhkit.wave.io.read_wecSim(file_name)
ws_output.keys()

ws_output['wave'].head()
wave = ws_output['wave']
plt.figure()
wave.plot()
wave.elevation.plot()

ws_output['bodies'].head()
bodies = ws_output['bodies']
plt.figure()
# bodies['body1_position_dof3'].plot()
bodies.body1_position_dof3.plot()

## Plot all body outputs
# num_signals = bodies.columns.size
# for col in range(num_signals):
#         plt.figure()
#         signal = bodies.columns[col]
#         # bodies[signal].plot()
#         plt.plot(bodies[signal])
#         plt.title(signal)
#         plt.xlabel('Time (s)')


ws_output['ptos'].head()
ptos=  ws_output['ptos']
# ptos.pto1_powerInternalMechanics_dof3.plot()
plt.figure()
(-1*ptos['pto1_powerInternalMechanics_dof3']).plot()

ws_output['constraints'].head()
constraints = ws_output['constraints']
plt.figure()
constraints['constraint1_forceConstraint_dof4'].plot()

ws_output['mooring'].head()
mooring = ws_output['mooring']


# ws_output['moorDyn']['Lines'].head()
moorDyn = ws_output['moorDyn']
moorDyn['Lines'].plot()




## Running MHKiT modules
try:
    # If wave type is *not* stored in DataFrame
    ws_spectrum = mhkit.wave.resource.elevation_spectrum(wave,60,1000)
except:
    # If wave type is stored in DataFrame
    eta = wave.drop(columns='type')
    ws_spectrum = mhkit.wave.resource.elevation_spectrum(eta,60,1000)

plt.figure()
ws_spectrum.plot()
Tp = mhkit.wave.resource.peak_period(ws_spectrum)
Hs = mhkit.wave.resource.significant_wave_height(ws_spectrum)
print(Tp)
print(Hs)


