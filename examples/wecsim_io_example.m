close all; clc; clear all;

%% Load wec-sim and convert output
%% no Mooring
% load('./data/RM3_matlabWorkspace.mat')
% % Convert WEC-Sim output object to structure for MHKiT
% output = struct(output);
% save(['./data/RM3_matlabWorkspace_structure.mat'],'output')

%% with Mooring
load('./data/RM3MooringMatrix_matlabWorkspace.mat')
output = struct(output);
save(['./data/RM3MooringMatrix_matlabWorkspace_structure.mat'],'output')

%% with moorDyn
% load('./data/RM3MoorDyn_matlabWorkspace.mat')
% output = struct(output);
% save(['./data/RM3MoorDyn_matlabWorkspace_structure.mat'],'output')

%% Plot WEC-Sim data

% wave 
wave = output.wave;
plot(wave.time,wave.elevation)

%bodies 
bodies = output.bodies;
% figure; plot(output.bodies(1).position(:,3))
figure; 
plot(bodies(1).time,bodies(1).position(:,3))

%ptos 
ptos = output.ptos;
figure; 
plot(-output.ptos(1).powerInternalMechanics(:,3))


%constraints 
constraints = output.constraints;
figure; 
% plot(constraints(1).time,constraints(1).forceConstraint(:,:))
plot(constraints(1).time,constraints(1).forceConstraint(:,4))
