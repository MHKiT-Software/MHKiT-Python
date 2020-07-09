import mhkit
import scipy.io as sio
import matplotlib.pyplot as plt

### loading wec-sim output using io function### --> NOT WORKING RIGHT NOW
file_name = './data/wecsim_output.mat'
ws_output = mhkit.wave.io.load_wecSim_output(file_name)
ws_output.head()
ws_output['elevation'].plot()
ws_output['body_1_pos_1'].plot()

mhkit.wave.resource.elevation_spectrum(ws_output[['elevation']],60,1000).T.plot()
S = mhkit.wave.resource.elevation_spectrum(ws_output[['elevation']],60,1000)
Tp = mhkit.wave.resource.peak_period(S)
Hs = mhkit.wave.resource.significant_wave_height(S)


######################################
######################################
### Manually loading wec-sim output ###
## import wecSim response class, saved as a data structure
ws_data = sio.loadmat('./data/wecsim_output.mat')
output = ws_data['output']

######################################
## import wecSim wave class
#         type: 'irregular'
#         time: [30001×1 double]
#    elevation: [30001×1 double]
######################################
wave = output['wave']
wave_type = wave[0][0][0][0][0][0]
wave_time = wave[0][0]['time'][0][0].squeeze()
wave_elevation = wave[0][0]['elevation'][0][0].squeeze()
#plot wave
plt.plot(wave_time,wave_elevation)


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
    #plot response
    for dof in range(6):
        plt.figure()
        plt.plot(bodies_time[body],bodies_position[body][:,dof])
        plt.plot(bodies_time[body],bodies_velocity[body][:,dof])
        plt.plot(bodies_time[body],bodies_acceleration[body][:,dof])
        plt.title(bodies_name[body]+' response, dof = ' + str(dof+1))
        plt.figure()
        plt.plot(bodies_time[body],bodies_forceTotal[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceExcitation[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceRadiationDamping[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceAddedMass[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceRestoring[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceMorrisonAndViscous[body][:,dof])
        plt.plot(bodies_time[body],bodies_forceLinearDamping[body][:,dof])
        plt.title(bodies_name[body]+' loads, dof = '+str(dof+1))
        
        
######################################
## import wecSim pto class
#                      name: 'PTO1'
#                      time: [30001×1 double]
#                  position: [30001×6 double]
#                  velocity: [30001×6 double]
#              acceleration: [30001×6 double]
#                forceTotal: [30001×6 double]
#            forceActuation: [30001×6 double]
#           forceConstraint: [30001×6 double]
#    forceInternalMechanics: [30001×6 double]
#    powerInternalMechanics: [30001×6 double]
######################################
ptos = output['ptos']
num_ptos = len(ptos[0][0]['name'][0])   # number of ptos
ptos_name = []
ptos_time = []
ptos_position = []
ptos_velocity = []
ptos_acceleration = []
ptos_forceTotal = []
ptos_forceActuation = []
ptos_forceConstraint = []
ptos_forceInternalMechanics = []
ptos_powerInternalMechanics = []
for pto in range(num_ptos):
    ptos_name.append(ptos[0][0]['name'][0][pto][0])
    ptos_time.append(ptos[0][0]['time'][0][pto])
    ptos_position.append(ptos[0][0]['position'][0][pto])
    ptos_velocity.append(ptos[0][0]['velocity'][0][pto])
    ptos_acceleration.append(ptos[0][0]['acceleration'][0][pto])
    ptos_forceTotal.append(ptos[0][0]['forceTotal'][0][pto])
    ptos_forceActuation.append(ptos[0][0]['forceActuation'][0][pto])
    ptos_forceConstraint.append(ptos[0][0]['forceConstraint'][0][pto])
    ptos_forceInternalMechanics.append(ptos[0][0]['forceInternalMechanics'][0][pto])
    ptos_powerInternalMechanics.append(ptos[0][0]['powerInternalMechanics'][0][pto])
    #plot pto
    for dof in range(6):
        plt.figure()
        plt.plot(bodies_time[pto],ptos_position[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_velocity[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_acceleration[pto][:,dof])
        plt.title(ptos_name[pto]+' response, dof = ' + str(dof+1))
        plt.figure()
        plt.plot(bodies_time[pto],ptos_forceTotal[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_forceActuation[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_forceConstraint[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_forceInternalMechanics[pto][:,dof])
        plt.plot(bodies_time[pto],ptos_powerInternalMechanics[pto][:,dof])
        plt.title(ptos_name[pto]+' loads, dof = ' + str(dof+1))


######################################
## import wecSim constraint class
#               name: 'Constraint1'
#               time: [30001×1 double]
#           position: [30001×6 double]
#           velocity: [30001×6 double]
#       acceleration: [30001×6 double]
#    forceConstraint: [30001×6 double]
######################################
constraints = output['constraints']
num_constraints = len(constraints[0][0]['name'][0])   # number of ptos
constraints_name = []
constraints_time = []
constraints_position = []
constraints_position = []
constraints_velocity = []
constraints_acceleration = []
constraints_forceConstraint = []
for constraint in range(num_constraints):
    constraints_name.append(constraints[0][0]['name'][0][constraint][0])
    constraints_time.append(constraints[0][0]['time'][0][constraint])
    constraints_position.append(constraints[0][0]['position'][0][constraint])
    constraints_velocity.append(constraints[0][0]['velocity'][0][constraint])
    constraints_acceleration.append(constraints[0][0]['acceleration'][0][constraint])
    constraints_forceConstraint.append(constraints[0][0]['forceConstraint'][0][constraint])
    #plot constraint
    for dof in range(6):
        plt.figure()
        plt.plot(constraints_time[constraint],constraints_position[constraint][:,dof])
        plt.plot(constraints_time[constraint],constraints_velocity[constraint][:,dof])
        plt.plot(constraints_time[constraint],constraints_acceleration[constraint][:,dof])
        plt.title(constraints_name[constraint]+' response, dof = ' + str(dof+1))
        plt.figure()
        plt.plot(constraints_time[constraint],constraints_forceConstraint[constraint][:,dof])
        plt.title(constraints_name[constraint]+' loads, dof = ' + str(dof+1))


######################################
## import wecSim ptosim class
## ptosim
#                 name: 'Non-Compressible Fluid Hydraulic'
#             pistonCF: [1×1 struct]
#            pistonNCF: [1×1 struct]
#           checkValve: [1×1 struct]
#                valve: [1×1 struct]
#          accumulator: [1×2 struct]
#       hydraulicMotor: [1×1 struct]
#      rotaryGenerator: [1×1 struct]
#    pmLinearGenerator: [1×1 struct]
#    pmRotaryGenerator: [1×1 struct]
#      motionMechanism: [1×1 struct]
## output.ptosim
#                  pistonNCF: [1×1 struct]
#              valve: [1×1 struct]
#        accumulator: [1×2 struct]
#     hydraulicMotor: [1×1 struct]
#    rotaryGenerator: [1×1 struct]
#    motionMechanism: [1×1 struct]
#               time: [4001×1 double]
#ptosim = output['ptosim']   #DEFAULT METHOD
######################################
try:
    ptosim = output['ptosim']  #TEMP FIX
    num_ptosim = len(ptosim[0][0]['name'][0])   # number of ptosim  
    ## This needs work...
except:
  print("ptosim class not used") 

        
######################################
## import wecSim moopring class
#mooring = output['mooring']  #DEFAULT METHOD
######################################
try:
  mooring = output['mooring']  #TEMP FIX
  num_mooring = len(output[0][0]['name'][0])   # number of mooring
  ## This needs work...
except:
  print("mooring class not used") 


######################################
## import wecSim moorDyn class
## output.moorDyn
#    Lines: [1×1 struct]
#    Line1: [1×1 struct]
#    Line2: [1×1 struct]
#    Line3: [1×1 struct]
#    Line4: [1×1 struct]
#    Line5: [1×1 struct]
#    Line6: [1×1 struct]  
#moorDyn = output['moorDyn']   #DEFAULT METHOD
######################################
try:
  moorDyn = output['moorDyn']  #TEMP FIX
  num_moorDyn = len(output[0][0]['name'][0])   # number of moorDyn  
  ## This needs work...
except:
  print("moorDyn class not used") 

